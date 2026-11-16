# 性能数据库设备目录（`data/<DEVICE>/`）产生机制与新设备接入说明

本文回答三个问题：`tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/`
是如何产生的、如何为 `ATLAS_800_A2_376T_64G` 产生同样目录、仓库里是否遗留了可复用的脚本和接口。

> 相关文档：[tools/perf_data_collection/readme.md](../../../tools/perf_data_collection/readme.md)（采集工具权威指南）、
> `.agents/skills/profiling_database_lifecycle/SKILL.md`（生命周期规范）。
>
> 注意：同目录的 `ADD_NEW_DEVICE_PROFILING_DATABASE_zh.md` 由其他模型生成，其"目录产生机制/扩展点"描述
> 基本正确，但"新增设备操作步骤"（尤其 `DeviceProfile` 示例代码与构建校验归因）与当前代码不符，
> 请以本文及代码实证为准；差异详见下方附录。

---

## 附录：`ADD_NEW_DEVICE_PROFILING_DATABASE_zh.md` 验证结果（基于当前分支代码）

| 该文档声明 | 核验结果 |
| --- | --- |
| 四阶段生成流水线、目录结构约定、`communication_data_ref` 相对路径、`ProfilingDataSource` 只接收显式目录 | ✅ 正确 |
| `DeviceProfile.__post_init__` 自动注册、`name` 唯一；`SUPPORTED_DEVICES` 白名单、`--device choices`、`--database-path` 绕过校验 | ✅ 正确（`device.py` L133-138、`common.py` L45-54/L537-554/L935-940） |
| 步骤 1 的 `DeviceProfile` 示例代码（`class ATLAS_950`、`reshape(48,2)`、`mma_ops={bf16:486e12}`、`comm_op_cost_s=10e-6`） | ❌ 错误。`ATLAS_950_486T_96G` **已注册**（`device.py` L815-825，位于 `A5` 类），重复注册会触发 L134-135 的 `already exists`；真实网格为 `POD_1K`（`reshape(16,8,8)`，L612-614）；bf16 算力为 `432e12`（`Chip.C486T`，L454）；`comm_op_cost_s=5e-6`（L448）。 |
| "`op_mapping.yaml` 的 `device` 不一致会被 `gmm_build.py`/`attention_build.py` 拒绝构建" | ⚠️ 部分错误。`gmm_build.py` L346-351 确实读 mapping 的 `device` 并校验，但可用 `--device` 覆盖；`attention_build.py` **不读** `op_mapping.yaml`，其 `device_name` 来自必填 `--device`（L340）并校验（L224-225）。 |
| `spec/.../profiling_database_lifecycle.md`（L48-49） | ❌ 该路径不存在；生命周期规范实际位于 Wiki 外链与 `.agents/skills/profiling_database_lifecycle/SKILL.md`。 |
| 扩样命令输出目录 `..._shape_generated`（L68、L73） | ❌ `generate_shape_grid.py` 原地读写 `--database-path`（L134），不存在 `_shape_generated` 后缀逻辑；该目录需预先存在且含 `op_mapping.yaml`。 |
| `--device ATLAS_950_486T_96G` 会自动拼出版本目录（L202） | ⚠️ 不完整。还需 `--vllm-version/--torch-version/--cann-version`，否则用默认版本（`common.py` L293-310/L537-554）。 |
| 测试文件路径 `tests/benchmark/ops/perf_database/test_profiling_*.py` | ✅ 正确（两文件均存在）。 |
| "框架已预留接口、新增设备无需改仿真核心逻辑" | ✅ 正确。 |

结论：该文档可用于理解"目录如何产生"和"扩展点在哪"，但其第 3 节操作步骤不可照抄——`ATLAS_950_486T_96G`
的画像已注册且仓库已存在模板骨架（`data/ATLAS_950_486T_96G/`，含 `collect_atlas950.sh`），请按本文第 3 节执行。

---

## 0. 结论速览

`data/<DEVICE>/` **不是由某个脚本一次性生成的静态目录**，而是采集工具链在真实昇腾 NPU 上，按
「解析 → 扩样 → replay 回写 → HCCL 通信采集」四个阶段逐步落盘的一组 CSV + `op_mapping.yaml`。

- 框架侧只提供三个扩展点：`DeviceProfile` 注册、`SUPPORTED_DEVICES` 白名单、目录/YAML 约定。
- 数据本身**无法脱离硬件凭空生成**：必须有目标设备的 NPU、CANN（含 `msprof`）、`torch`/`torch_npu`、vLLM-Ascend。
- 对 `ATLAS_800_A2_376T_64G`：设备画像和白名单**已就位**，采集脚本/接口/模板**全部齐备**，但仓库中
  **没有该设备的数据目录**——缺的是真机实测结果，不是代码。

---

## 1. `ATLAS_800_A3_752T_128G_DIE` 目录是怎么产生的

### 1.1 目录内容与格式

```
tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/
├── vllm_ascend/                                  # 计算算子，按软件栈版本分目录
│   ├── vllm0.13.0_torch2.8.0_cann8.3/            # 71 个 CSV + op_mapping.yaml
│   ├── vllm0.15.0_torch2.9.0_cann8.5/            # 69 个 CSV + op_mapping.yaml
│   ├── vllm0.18.0_torch2.9.0_cann8.5/            # 108 个 CSV + op_mapping.yaml
│   └── vllm0.23.0rc1_torch2.10.0_cann9.0.1/      # 79 个 CSV + op_mapping.yaml
└── hccl/                                         # 通信算子，按 CANN 版本分目录
    ├── v8.5/
    └── v9.0.1/
```

| 数据文件 | schema / 含义 |
| --- | --- |
| `vllm_ascend/<version>/<KernelType>.csv` | 每个 kernel 一个文件，每行是一个 shape/dtype/format 组合的实测耗时与微架构指标。表头含 `Input Shapes`、`Input Data Types`、`Average Duration(us)`、`Profiling Average Duration(us)`、`MicroBench aicore_time(us)`、`cube_utilization(%)` 等约 60 列。 |
| `hccl/<cann>/hcom_*.csv` | 通信算子实测，列为 `message_bytes,num_devices,dtype,topology_tier,Duration(us),bandwidth_gbps`。`topology_tier`：0 跨 pod / 1 pod 内 / 2 die 级。 |
| `op_mapping.yaml` | 算子 → NPU kernel 映射；必含 `device:`（等于 `DeviceProfile.name`）、`cann_version`、`pytorch_version`、`communication_data_ref`（相对路径，指向 hccl 目录）。实测样例 `vllm0.18.0.../op_mapping.yaml` L196-203 为 `device: ATLAS_800_A3_752T_128G_DIE`、`communication_data_ref: "../../hccl/v8.5/"`。 |

这些 CSV 由 `.gitattributes` 配置为 **Git LFS** 管理。

运行时消费者 `tensor_cast/performance_model/profiling_database/profiling_data_source.py` 的
`ProfilingDataSource` 只接收**显式目录** `data_dir`，不做设备名→目录映射；`hcom_` 前缀 kernel 通过
`communication_data_ref` 解析到 hccl 目录读取。

### 1.2 生成流程（四个阶段）

命令与参数以 `tools/perf_data_collection/readme.md` 和当前分支 `--help` 为准：

```bash
# 1) 解析：NPU kernel_details*.csv -> per-operator CSV
python tools/perf_data_collection/parsers/parse_kernel_details.py \
  --profiling-path ./PROF_001 \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5

# 2) 扩样：从模型/优化器场景补充 theory shape 行
python tools/perf_data_collection/generate_shape_grid.py \
  --database-path <DB> --target-models deepseek-ai/DeepSeek-V3 Qwen/Qwen3-32B \
  --rows 2000 --seed 20260409

# 3) 回写：msprof replay 算子，写回 Average Duration(us) 等列
python tools/perf_data_collection/start_microbench.py \
  --database-path <DB> --repeat-count 1 --update-mode missing-only --fail-fast

# 4) 通信采集：生成 hcom_*.csv（752T DIE 默认网格 48 8 2）
bash tools/perf_data_collection/comm_bench/run_comm_bench.sh ./hccl_bench_data
```

阶段 3 由 `start_microbench.py` 调用 `op_replay/run_all_op.py` 调度约 45 个 `*_run.py` 单算子 replay 脚本，
聚合 `op_summary_*.csv` 后写回 CSV，并在 `<db>/reports/` 生成 `profile_update_report_<ts>.md`。

### 1.3 命名规则

设备名**不做解析/映射，就是目录名本身**，经白名单校验后字符串拼接：

- 目录名 `<DEVICE>` = `DeviceProfile.name`，例如 `tensor_cast/device.py` L388 定义
  `A3_752T_128G_DIE = DeviceProfile(name="ATLAS_800_A3_752T_128G_DIE", ...)`。
- 采集侧 `tools/perf_data_collection/op_replay/common.py` L44-55：`DATA_DIR` 指向 `.../profiling_database/data`，
  `SUPPORTED_DEVICES` 为白名单，`DEFAULT_DEVICE = "ATLAS_800_A3_752T_128G_DIE"`。
- `get_target_data_dir()`（L537-554）按 `DATA_DIR / device / "vllm_ascend" / <version>` 拼装；
  `resolve_version_dir_name()` 把 vLLM/torch/CANN 版本拼成 `vllm<v>_torch<v>_cann<v>`。
- `normalize_device_name()` 仅 `strip()`，无别名映射。显式传 `--database-path` 可绕过白名单。

---

## 2. 仓库中遗留的脚本、接口与模板（均已核验存在）

### 2.1 采集脚本（`tools/perf_data_collection/`）

| 脚本 | 作用 | 关键参数 |
| --- | --- | --- |
| `parsers/parse_kernel_details.py` | 解析 profiling，拆 per-operator CSV | `--profiling-path`（必填）、`--database-path`、`--device`、`--vllm-version/--torch-version/--cann-version` |
| `generate_shape_grid.py` | 扩样 theory shape / 优化器精确 query 行 | `--database-path`、`--target-models`、`--optimizer-args-file`、`--ops`、`--rows`、`--seed` |
| `start_microbench.py` | NPU replay + 耗时回写 | `--database-path`、`--device`、`--ops`、`--repeat-count`、`--update-mode {all,missing-only}`、`--fail-fast` |
| `op_replay/run_all_op.py` | 自动发现并运行 `*_run.py` | `--execution-mode {inprocess,subprocess}` |
| `op_replay/*_run.py`（约 45 个） | 单算子 replay 入口 | 由调度器调用 |
| `comm_bench/generate_comm_microbench.py` | 生成 `hcom_*.csv` | `--database-path`、`--ops`、`--grid-shape`（默认 48 8 2）、`--num-devices` |
| `comm_bench/run_comm_bench.sh` | 752T DIE 一键 HCCL 采集封装 | 支持 `NNODES/NODE_RANK/MASTER_ADDR` 多机 |
| `find_database_anomalies.py` | 只读异常审计 | `--database-path`、`--output-dir`、`--residual-threshold` |
| `run_op_microbench.py` | 独立单算子测试（不写库） | 见 `README_op_microbench.md` |

### 2.2 设备注册与约定接口

- `tensor_cast/device.py`：`DeviceProfile.__post_init__` 自动注册到 `all_device_profiles`，`name` 全局唯一。
- `tensor_cast/device_profiles/__init__.py`：自动导入目录内全部 `.py`；`README.md` 给出最小模板。
- `tools/perf_data_collection/op_replay/common.py` 的 `SUPPORTED_DEVICES` / `get_target_data_dir()`。
- `op_mapping.yaml` 的 `device` + `communication_data_ref` 配置约定。

### 2.3 Skill 与文档

| 路径 | 用途 |
| --- | --- |
| `.agents/skills/profiling_database_lifecycle/SKILL.md` | 端到端 5 阶段生命周期（冻结基线 → 密度/Shape → 采集回写 → 质量检查 → PR → 运行反馈） |
| `.agents/skills/profiling_database_axis_density/SKILL.md` | 轴密度定标 |
| `.agents/skills/microbench/SKILL.md` | 从 profiling CSV 生成 `op_replay/<KernelType>_run.py` |
| `.agents/skills/op-mapping/SKILL.md` | 生成 `op_mapping.yaml` |
| `.agents/skills/device_config/SKILL.md` | 自然语言硬件规格 → `DeviceProfile` |
| `docs/perf_database/tutorial/ADD_NEW_DEVICE_PROFILING_DATABASE_zh.md` | 新设备接入分步指南 |
| `tools/perf_data_collection/readme.md` | 采集工具完整参数与输出说明 |

### 2.4 可直接复用的模板骨架

`tensor_cast/performance_model/profiling_database/data/ATLAS_950_486T_96G/` 是一个**完整的新设备骨架样例**：

- `MANIFEST.json`：`status: TEMPLATE_NOT_MEASURED`、`cloned_from: ATLAS_800_A3_752T_128G_DIE/...`、
  79 个计算 CSV / 4 个 hccl CSV、`shape_rows_preserved: 19359`、`latency_policy: all measured columns = 0`，
  并明确 `prohibited_use`（不得当实测性能发布）。
- `collect_atlas950.sh`：四步端到端采集脚本（扩样 → replay 回写 → HCCL → 命中率验证），可用
  `REPEAT` / `NPROC` 覆盖参数。
- `vllm_ascend/vllm0.23.0rc1_torch2.10.0_cann9.0.1/` 下 79 个 CSV + `op_mapping.yaml`（全零 latency，结构完整），
  `hccl/v9.0.1/` 下 4 个 `hcom_*.csv`。

### 2.5 仅外部文档描述 / 缺失的部分

- 生命周期顶层规范不在仓库内，指向 Wiki：`Ascend/msmodeling.wiki` 的
  `design/profiling_database_lifecycle.md`。
- Skill 中若干 PR 合入顺序、`--max-hbm-gb` 等描述滞后于当前代码，应以当前分支 `--help` 为准。
- **没有任何设备的数据可以靠脚本凭空生成**——脚本只处理真实 profiling 输入或在 NPU 上实测。

---

## 3. 为 `ATLAS_800_A2_376T_64G` 产生同样目录

### 3.1 现状判断

| 项 | 状态 | 证据 |
| --- | --- | --- |
| `DeviceProfile` 注册 | 已就位 | `tensor_cast/device.py` L257-258：`A2_376T_64G = DeviceProfile(name="ATLAS_800_A2_376T_64G", ...)` |
| 采集白名单 | 已就位 | `op_replay/common.py` L47：`SUPPORTED_DEVICES` 含 `ATLAS_800_A2_376T_64G` |
| readme 支持清单 | 已就位 | `tools/perf_data_collection/readme.md` L29 |
| 数据目录 | **不存在** | `data/` 下目前只有 `ATLAS_800_A3_752T_128G_DIE/` 与 `ATLAS_950_486T_96G/` |
| 采集脚本/接口/模板 | 全部齐备 | 见第 2 节 |

因此无需修改任何核心代码，只需「准备目录骨架 + 在目标设备上实机采集」。

### 3.2 操作步骤

**步骤 1：确认设备注册（已满足，可跳过）**

```bash
python - <<'PY'
from tensor_cast.device import DeviceProfile
print(DeviceProfile.all_device_profiles["ATLAS_800_A2_376T_64G"])
PY
```

**步骤 2：建立目录骨架并配置 `op_mapping.yaml`**

按目标环境真实版本创建 `data/ATLAS_800_A2_376T_64G/vllm_ascend/<vllm>_torch<torch>_cann<cann>/` 与
`data/ATLAS_800_A2_376T_64G/hccl/<cann>/`。可从 A3 对应软件栈版本复制 `op_mapping.yaml` 结构，
然后修改：

```yaml
device: ATLAS_800_A2_376T_64G
communication_data_ref: "../../hccl/<cann>/"
```

> `device` 必须等于已注册的 `DeviceProfile.name`，否则校准工具拒绝构建。

**步骤 3：在真机执行四阶段采集**

```bash
DB=tensor_cast/performance_model/profiling_database/data/ATLAS_800_A2_376T_64G/vllm_ascend/<version>
HCCL_DB=tensor_cast/performance_model/profiling_database/data/ATLAS_800_A2_376T_64G/hccl/<cann>

# 1) 解析（输入：A2 真机 profiling 产物）
python tools/perf_data_collection/parsers/parse_kernel_details.py \
  --profiling-path ./PROF_A2 --database-path "$DB" --device ATLAS_800_A2_376T_64G

# 2) 扩样
python tools/perf_data_collection/generate_shape_grid.py \
  --database-path "$DB" --rows 2000 --seed 0

# 3) replay 回写（需 NPU + CANN + msprof）
python tools/perf_data_collection/start_microbench.py \
  --database-path "$DB" --repeat-count 3 --update-mode missing-only --fail-fast

# 4) 通信采集：按 A2 真实拓扑设置 --grid-shape，不要照抄 752T 的 48 8 2
torchrun --nproc_per_node=8 tools/perf_data_collection/comm_bench/generate_comm_microbench.py \
  --database-path "$HCCL_DB" \
  --ops all_reduce all_gather reduce_scatter all_to_all \
  --grid-shape <按真实拓扑> --num-devices <按真实拓扑>
```

前置条件：Linux + 昇腾 A2 NPU + CANN Toolkit（含 `msprof`）+ `torch`/`torch_npu` + vLLM-Ascend，
并加载环境 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。本采集链路无法在纯 CPU/Windows 环境完成。

**步骤 4：校验与发布**

```bash
pytest -q tests/benchmark/ops/perf_database/test_profiling_data_source.py
pytest -q tests/benchmark/ops/perf_database/test_profiling_interpolation_non_regression.py
python tools/perf_data_collection/find_database_anomalies.py --database-path "$DB" --output-dir reports/a2_audit
python -m cli.inference.text_generate <model> --device ATLAS_800_A2_376T_64G \
  --performance-model profiling --profiling-database-path "$DB" \
  --dump-input-shapes --export-empirical-metrics-file reports/a2_hit.json
```

通过后按 `profiling-database-lifecycle` Skill 走异常审计、query replay 和 PR 流程。

---

## 4. 关键结论

| 问题 | 答案 |
| --- | --- |
| 这个文件夹如何产生？ | `tools/perf_data_collection/` 四阶段采集流水线在真实 NPU 上落盘 CSV + `op_mapping.yaml`，不是静态拷贝。 |
| 目录名如何映射设备？ | 目录名直接等于 `DeviceProfile.name`，采集侧用 `SUPPORTED_DEVICES` 白名单校验，无别名映射。 |
| A2_376T_64G 如何产生？ | 注册已就位，建目录骨架 + 配 `op_mapping.yaml` + 按真实拓扑跑四阶段采集；数据必须来自 A2 真机实测。 |
| 是否有遗留脚本/接口？ | 有且完整：4 个阶段脚本、45 个单算子 replay、HCCL 工具、审计工具、3 个扩展点、5 个 Skill、`ATLAS_950_486T_96G` 模板骨架。 |
| 能否直接复制生成？ | 不能。骨架可复制，但 latency 必须实机采集，复制的零值/他机数据不得作为实测性能发布。 |
