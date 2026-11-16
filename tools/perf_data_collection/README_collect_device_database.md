# 通用设备性能数据库采集脚本说明

`collect_device_database.sh` 提供一条命令，在**目标昇腾 NPU 的 Linux Docker 环境**中完成
`data/<DEVICE>/` 目录的四阶段采集，产出与
`tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/` 同构的目录。

> 项目此前没有 Dockerfile。采集环境镜像定义见同目录 [`Dockerfile.collect`](Dockerfile.collect)。

## 1. 适用范围与前提

- 只接受**已注册**的 `DeviceProfile.name`。用 `--list-devices` 查看当前仓库已注册设备
  （如 `ATLAS_800_A2_376T_64G`、`ATLAS_800_A2_280T_64G`、`ATLAS_950_486T_96G`）。
- 必须在带昇腾 NPU 的容器/主机运行；加载 CANN 环境后确认 `npu-smi info` 可见卡。
- 依赖 `python3`（可 `import torch` / `torch_npu`）、`msprof`（阶段 3）、`torchrun`（阶段 4）。
- 采集数据来自真实硬件实测，**无法靠拷贝/占位生成**。

### 与白名单的关系

采集工具 `op_replay/common.py` 的 `SUPPORTED_DEVICES` **不含** `ATLAS_950_486T_96G`。
本脚本全程使用显式 `--database-path` 传递目录，因此不受该白名单 `choices` 限制；
只要设备是已注册的 `DeviceProfile`，即可采集。

## 2. 产出目录结构

```
tensor_cast/performance_model/profiling_database/data/<DEVICE>/
├── vllm_ascend/<vllm>_torch<torch>_cann<cann>/
│   ├── op_mapping.yaml          # 需预先准备（见第 4 节）
│   ├── <KernelType>.csv         # 阶段 1/2 生成，阶段 3 回写耗时
│   └── reports/                 # start_microbench.py 生成
└── hccl/<cann>/                 # 如 v8.5 / v9.0.1
    └── hcom_*.csv               # 阶段 4 生成
```

## 3. 快速使用

```bash
# 0) 加载 CANN 环境（镜像未自动加载时）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1) 查看可用设备
bash tools/perf_data_collection/collect_device_database.sh --list-devices

# 2) 先看命令、不实际跑（强烈建议）
bash tools/perf_data_collection/collect_device_database.sh \
  --device ATLAS_950_486T_96G \
  --version-dir vllm0.23.0rc1_torch2.10.0_cann9.0.1 \
  --hccl-cann-version v9.0.1 \
  --profiling-path ./PROF_001 \
  --optimizer-args-file tensor_cast/performance_model/profiling_database/data/ATLAS_950_486T_96G/workloads_glm52.yaml \
  --dry-run

# 3) 确认无误后去掉 --dry-run 正式采集
```

### 只跑某一段

```bash
# compute 数据已就绪，只补采 HCCL（A2 需显式 --grid-shape）
bash tools/perf_data_collection/collect_device_database.sh \
  --device ATLAS_800_A2_376T_64G \
  --version-dir vllm0.18.0_torch2.9.0_cann8.5 \
  --grid-shape 16 8 \
  --skip-parse --skip-shape --skip-replay
```

## 4. 关键前提：op_mapping.yaml

阶段 2/3 要求 compute 目录已有 `op_mapping.yaml`，且其中 `device:` 必须等于 `--device`。
`parse_kernel_details.py` **不会生成**它。准备方式二选一：

1. 用 `op-mapping` Skill 基于 profiling CSV + 目标模型 + 软件栈版本生成（推荐，见
   `.agents/skills/op-mapping/SKILL.md`）；
2. 从仓库内同软件栈的已有版本目录拷贝一份，把 `device:` 改为目标设备名，
   并核对 `communication_data_ref` 指向本设备的 `hccl/<cann>/`。

若缺失或 `device` 不匹配，脚本会明确报错退出，不会写出无效数据库。

## 5. 版本目录与 HCCL 网格

- 版本目录名格式固定为 `vllm<v>_torch<torch>_cann<v>`，可用 `--version-dir` 一次给定，
  或分别传三个 `--*-version`；都不传时脚本尝试从运行时探测，失败则要求显式指定。
- HCCL `--grid-shape` 默认值：`ATLAS_800_A3_752T_128G_DIE` = `48 8 2`，
  `ATLAS_950_486T_96G` = `16 8 8`（POD_1K）。其它设备必须显式传 `--grid-shape`，
  值应与 `tensor_cast/device.py` 中该 profile 的 `comm_grid` 一致。

## 6. 采集后处理（发布前）

```bash
# 只读异常审计
python tools/perf_data_collection/find_database_anomalies.py \
  --database-path <DB_DIR> --output-dir <report_dir>

# query 命中率验证
python -m cli.inference.text_generate <model> --device <DEVICE> \
  --performance-model profiling --profiling-database-path <DB_DIR> --dump-input-shapes

# 回归测试
pytest -q tests/benchmark/ops/perf_database/test_profiling_data_source.py
```

通过后再按 `profiling-database-lifecycle` Skill 走 PR 流程。

## 7. 参数速查

`bash tools/perf_data_collection/collect_device_database.sh --help` 获取完整参数列表。
常用：`--device` / `--version-dir` / `--hccl-cann-version` / `--profiling-path` /
`--optimizer-args-file` / `--target-models` / `--grid-shape` / `--num-devices` /
`--ops` / `--repeat-count` / `--nproc` / `--skip-*` / `--dry-run` / `--python-bin`。

## 8. 背景说明

- 目录产生机制的完整论证见
  `docs/perf_database/tutorial/PROFILING_DATA_DIR_ORIGIN_AND_NEW_DEVICE_zh.md`。
- 采集工具权威参数见 `tools/perf_data_collection/readme.md`。
- 端到端采集模板（ATLAS_950）见
  `data/ATLAS_950_486T_96G/collect_atlas950.sh`；本脚本是其通用化版本。
