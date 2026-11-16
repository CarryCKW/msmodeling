# 新增设备性能数据库目录指南（以 ATLAS_950_486T_96G 为例）

本文档说明 `tensor_cast/performance_model/profiling_database/data/<DEVICE>/` 目录（例如
`ATLAS_800_A3_752T_128G_DIE`）是如何产生的、框架为新增设备预留了哪些接口，以及如果要为
`ATLAS_950_486T_96G` 生成同样的目录内容，应当如何操作。

---

## 1. 这个目录是怎么产生的

`data/<DEVICE>/` 不是手写或静态拷贝的，而是由 `tools/perf_data_collection/` 下的采集链路
按"解析 → 扩样 → 回写 → 通信采集"四个阶段在真实 NPU 环境（CANN、`msprof`、`torchrun`、
vLLM-Ascend）中逐步生成的一组 CSV + `op_mapping.yaml`。

### 1.1 目录结构约定

目录名 `<DEVICE>` 就是 `DeviceProfile.name`（设备画像的唯一名称）。标准布局如下：

```text
tensor_cast/performance_model/profiling_database/data/<DEVICE>/
├── vllm_ascend/
│   └── <vllm>_torch<torch>_cann<cann>/        # 例: vllm0.18.0_torch2.9.0_cann8.5
│       ├── op_mapping.yaml                    # 算子→NPU kernel 映射表（必含 device 字段）
│       ├── <KernelType>.csv                    # 每个算子的实测耗时数据，如 MatMulV2.csv
│       └── reports/                            # replay 回写报告（start_microbench.py 生成）
└── hccl/
    └── <cann>/                                 # 例: v9.0.1
        ├── hcom_allReduce_.csv                 # 通信算子实测数据
        ├── hcom_allGather_.csv
        ├── hcom_reduceScatter_.csv
        └── hcom_alltoallv_.csv
```

关键约定：

- `vllm_ascend/<version>/op_mapping.yaml` 中的 `device:` 字段（如
  `device: ATLAS_800_A3_752T_128G_DIE`）**必须与一个已注册的 `DeviceProfile.name` 一致**，
  否则校准工具（`gmm_build.py` / `attention_build.py`）会拒绝构建。
- `op_mapping.yaml` 中的 `communication_data_ref` 是**相对路径**，指向同设备下的 hccl 目录。
  例如 `communication_data_ref: "../../hccl/v8.5/"`，相对于
  `vllm_ascend/<version>/` 解析为 `data/<DEVICE>/hccl/v8.5/`。
- 运行时（`tensor_cast/performance_model/profiling_database/profiling_data_source.py` 的
  `ProfilingDataSource`）只接收一个**显式目录路径** `data_dir`，并不把设备名自动映射到目录；
  设备名 → 目录的关联只存在于采集工具中（见下文 3.2）。

### 1.2 生成流程（四个阶段）

详见 `tools/perf_data_collection/readme.md` 与
`spec/.../profiling_database_lifecycle.md`（对应 `profiling-database-lifecycle` Skill）。

| 阶段 | 工具 | 作用 |
| --- | --- | --- |
| 解析 | `parsers/parse_kernel_details.py` | 把 NPU `kernel_details*.csv` profiling 产物拆成 per-operator CSV |
| 扩样 | `generate_shape_grid.py` | 从模型/优化器场景采样补充 theory shape 行，扩大覆盖 |
| 回写 | `start_microbench.py` → `op_replay/run_all_op.py` | 在 NPU 上 replay 算子，用 `msprof` 聚合并把耗时写回 CSV |
| 通信采集 | `comm_bench/generate_comm_microbench.py`、`comm_bench/run_comm_bench.sh` | 采集 HCCL 通信算子耗时写 `hcom_*.csv` |

简化示例（以 `ATLAS_800_A3_752T_128G_DIE` 为例）：

```bash
# 1) 解析真实 profiling
python tools/perf_data_collection/parsers/parse_kernel_details.py \
  --profiling-path ./PROF_001 \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5

# 2) 扩样 theory shape
python tools/perf_data_collection/generate_shape_grid.py \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5_shape_generated \
  --target-models deepseek-ai/DeepSeek-V3 Qwen/Qwen3-32B --rows 2000 --seed 20260409

# 3) 回写耗时（需 NPU + CANN + msprof）
python tools/perf_data_collection/start_microbench.py \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5_shape_generated \
  --repeat-count 1 --update-mode missing-only

# 4) 采集 HCCL（752T DIE 固定 48 8 2 网格）
bash tools/perf_data_collection/comm_bench/run_comm_bench.sh ./hccl_bench_data
```

> 注意：这份数据依赖真实硬件实测，**无法靠拷贝/占位直接"生成"**。新设备必须有对应的
> NPU 硬件、CANN 和软件栈才能完成 replay 与通信采集。

---

## 2. 框架是否为新增设备预留了接口

是的。新增设备分三处明确的扩展点，框架已预留接口，无需改动核心逻辑：

### 2.1 设备画像注册接口（运行时 + 校准）

`tensor_cast/device.py` 中的 `DeviceProfile` 在 `__post_init__` 会自动注册到
`DeviceProfile.all_device_profiles`，`name` 字段全局唯一。新增设备只需追加一条
`DeviceProfile` 定义（或放到 `tensor_cast/device_profiles/` 下的独立文件，由该目录
`__init__.py` 自动导入）。

- 引导式生成：`device_config` Skill 可把自然语言硬件规格转换为可运行的 `DeviceProfile`。
- 命名约束：`<DEVICE>` 目录名 **必须** 等于 `DeviceProfile.name`。

### 2.2 采集工具的设备清单接口

采集链路集中维护一份设备白名单：
`tools/perf_data_collection/op_replay/common.py`

```text
DATA_DIR = .../profiling_database/data          # 第 44 行：所有设备目录的根
SUPPORTED_DEVICES = [                            # 第 45-54 行：支持的设备名清单
    "TEST_DEVICE",
    "ATLAS_800_A2_376T_64G",
    ...
    "ATLAS_800_A3_752T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE",
]
DEFAULT_DEVICE = "ATLAS_800_A3_752T_128G_DIE"    # 第 55 行
```

- `get_target_data_dir()`（第 537-554 行）按 `DATA_DIR / device / "vllm_ascend" / <version>`
  拼装目录，设备名即目录名，没有额外的硬编码映射。
- CLI 的 `--device` 参数通过 `choices=SUPPORTED_DEVICES`（第 935-940 行）限制可选值。
- 若直接用 `--database-path` 显式传目录，则**不**经过 `SUPPORTED_DEVICES` 校验。

> 因此：要让采集工具以 `--device` 自动拼路径的方式支持新设备，必须把它加入
> `SUPPORTED_DEVICES`；若每次都显式传 `--database-path`，则无需改清单也能工作。

### 2.3 目录/映射约定接口

- 目录层级约定：`data/<DEVICE>/vllm_ascend/<version>/` 与 `data/<DEVICE>/hccl/<cann>/`。
- `op_mapping.yaml` 必含 `device:` 字段且等于 `DeviceProfile.name`；必含
  `communication_data_ref` 指向 hccl 子目录。
- 这些均为"约定 + 配置驱动"，框架通过读取目录与 YAML 自动适配，不依赖硬编码设备清单。

---

## 3. 为 ATLAS_950_486T_96G 生成同样内容的操作步骤

要点：**先在 2.1/2.2 两处登记设备，再跑 1.2 的采集流水线**，并把硬件实测数据落到对应目录。

### 步骤 1：注册 DeviceProfile（运行时/校准可用）

在 `tensor_cast/device.py` 中追加（或写入 `tensor_cast/device_profiles/atlas_950_486t_96g.py`）：

```python
class ATLAS_950:
    _STATIC_COST = StaticCost(mma_op_cost_s=5e-6, gp_op_cost_s=2e-6, comm_op_cost_s=10e-6)
    _INTERCONNECT = CommGrid(
        grid=torch.arange(96).reshape(48, 2),          # 按真实硬件网格调整
        topologies={
            0: InterconnectTopology(bandwidth_bytes_ps=..., latency_s=..., type=InterconnectType.CLOS),
            1: InterconnectTopology(bandwidth_bytes_ps=..., latency_s=..., type=InterconnectType.FULL_MESH),
        },
    )
    ATLAS_950_486T_96G = DeviceProfile(
        name="ATLAS_950_486T_96G",
        vendor="HUAWEI",
        mma_ops={torch.bfloat16: 486e12, ...},
        gp_ops={...},
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=...,
        compute_efficiency=...,
        memory_efficiency=...,
        comm_grid=_INTERCONNECT,
        static_cost=_STATIC_COST,
    )
```

> 数值需来自官方规格或实测；不确定项按 `device_config` Skill 标记为 `needs calibration`。
> 建议先走 `device_config` Skill 生成并校验。

验证注册成功：

```bash
python - <<'PY'
from tensor_cast.device import DeviceProfile
print(DeviceProfile.all_device_profiles["ATLAS_950_486T_96G"])
PY
```

### 步骤 2：把设备加入采集白名单

在 `tools/perf_data_collection/op_replay/common.py` 的 `SUPPORTED_DEVICES` 列表追加：

```python
SUPPORTED_DEVICES = [
    ...
    "ATLAS_800_A3_560T_128G_DIE",
    "ATLAS_950_486T_96G",          # 新增
]
```

### 步骤 3：准备目录并采集数据

按真实软件栈创建版本目录，例如：

```text
tensor_cast/performance_model/profiling_database/data/ATLAS_950_486T_96G/
├── vllm_ascend/
│   └── vllm0.18.0_torch2.9.0_cann8.5/      # 实际版本号以目标环境为准
│       └── op_mapping.yaml                 # device: ATLAS_950_486T_96G
└── hccl/
    └── v8.5/
```

然后执行 1.2 的四个阶段（`--device ATLAS_950_486T_96G` 会自动拼出上面的目录），并把
`op_mapping.yaml` 中的 `device` 与 `communication_data_ref` 改为新设备的值：

```yaml
device: ATLAS_950_486T_96G
communication_data_ref: "../../hccl/v8.5/"
```

> ATLAS_950 的网格形状可能与 752T DIE 不同（752T 默认 `48 8 2`），通信采集时请按真实拓扑
> 传入 `--grid-shape`，或用 `generate_comm_microbench.py` 手动配置，不要照抄 752T 的网格。

### 步骤 4：发布前校验

- 确保 `op_mapping.yaml` 的 `device` 字段在 `DeviceProfile.all_device_profiles` 中已注册；
- 运行 profiling database 回归测试：
  ```bash
  pytest -q tests/benchmark/ops/perf_database/test_profiling_data_source.py
  pytest -q tests/benchmark/ops/perf_database/test_profiling_interpolation_non_regression.py
  ```
- 按 `profiling-database-lifecycle` Skill 做异常审计与 query replay 后再合入。

---

## 4. 小结

| 组件 | 是否需改动 | 说明 |
| --- | --- | --- |
| `tensor_cast/device.py`（或 `device_profiles/`） | 是 | 注册 `name="ATLAS_950_486T_96G"` 的 `DeviceProfile` |
| `tools/perf_data_collection/op_replay/common.py` 的 `SUPPORTED_DEVICES` | 是（若用 `--device` 自动拼路径） | 加入新设备名 |
| 目录 `data/ATLAS_950_486T_96G/...` | 是（由采集工具生成） | 需真实 NPU 实测，无法占位生成 |
| `op_mapping.yaml` 的 `device` / `communication_data_ref` | 是 | 必须与新设备一致 |
| 核心查询/插值逻辑 | 否 | 框架按目录+YAML 自适应，已预留接口 |

结论：框架**已预留明确接口**（DeviceProfile 注册 + `SUPPORTED_DEVICES` 清单 + 目录/映射约定）。
新增 `ATLAS_950_486T_96G` 属于"登记设备 + 跑采集流水线"，不需要修改 TensorCast 仿真核心代码；
但目录内的 CSV/映射数据本身必须由对应硬件实测产生，不能凭空生成。
