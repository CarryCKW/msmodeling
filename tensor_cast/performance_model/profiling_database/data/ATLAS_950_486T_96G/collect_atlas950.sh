#!/bin/bash
# ============================================================
#  ATLAS_950_486T_96G 实测算子库采集入口
#
#  当前数据库状态: TEMPLATE_NOT_MEASURED（见 MANIFEST.json）
#  所有 latency 列为 0，本脚本负责在真机上把它们填成实测值。
#
#  运行环境（必须）:
#    Linux + 昇腾 NPU + CANN Toolkit(含 msprof) + torch/torch_npu
#    + vllm-ascend 0.23.0 系列；在仓库根目录执行
#    source /usr/local/Ascend/ascend-toolkit/set_env.sh
#
#  用法:
#    bash tensor_cast/performance_model/profiling_database/data/ATLAS_950_486T_96G/collect_atlas950.sh
#    REPEAT=5 NPROC=8 bash ...   # 覆盖重复次数 / torchrun 进程数
#
#  采集完成后:
#    - 用 find_database_anomalies.py 做只读审计
#    - 用 text_generate --performance-model profiling 验证 query 命中率
#    - 把 MANIFEST.json 的 status 改成 COLLECTED 并填 collection_date
# ============================================================

set -euo pipefail

HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(cd "${HERE}/../../../../.." && pwd)"
DB="${HERE}/vllm_ascend/vllm0.23.0rc1_torch2.10.0_cann9.0.1"
HCCL_DB="${HERE}/hccl/v9.0.1"
REPORT_DIR="${REPO_ROOT}/reports/atlas950_profiling"
mkdir -p "$REPORT_DIR"

REPEAT="${REPEAT:-3}"
NPROC="${NPROC:-8}"

# ---------- 0. 前置检查 ----------
command -v msprof >/dev/null 2>&1 || {
  echo "[ERROR] msprof 未找到，请先加载 CANN 环境: source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
  exit 1
}
command -v torchrun >/dev/null 2>&1 || {
  echo "[ERROR] torchrun 未找到，请激活带 torch / torch_npu 的环境" >&2
  exit 1
}
[ -f "${DB}/op_mapping.yaml" ] || { echo "[ERROR] 数据库骨架缺失: ${DB}" >&2; exit 1; }

echo "[0/4] 设备与软件栈"
npu-smi info | head -n 20 || true
echo "  DB        = ${DB}"
echo "  HCCL_DB   = ${HCCL_DB}"
echo "  repeat    = ${REPEAT}"
echo "  报告目录  = ${REPORT_DIR}"

# ---------- 1. 按真实寻优场景生成待采 shape ----------
echo "[1/4] generate_shape_grid.py: 生成 workload 精确查询的 shape"
python "${REPO_ROOT}/tools/perf_data_collection/generate_shape_grid.py" \
  --database-path "$DB" \
  --optimizer-args-file "${HERE}/workloads_glm52.yaml" \
  --rows 2000 \
  --seed 0

# ---------- 2. NPU 上 replay 算子并回写 latency ----------
# missing-only 只填 0/空 latency 的行，不会覆盖任何已有实测值
echo "[2/4] start_microbench.py: msprof replay + 回写耗时"
python "${REPO_ROOT}/tools/perf_data_collection/start_microbench.py" \
  --database-path "$DB" \
  --repeat-count "$REPEAT" \
  --update-mode missing-only \
  --fail-fast

# ---------- 3. HCCL 通信微基准 ----------
# grid 16 8 8 = POD_1K（16 pods x 8 unions x 8 dies），必须按 950 实际拓扑填
# num-devices 8 -> tier2(同 Union FullMesh 840GB/s)
# num-devices 16 -> tier1(跨 Union 448GB/s)
echo "[3/4] generate_comm_microbench.py: 采集 hcom_*.csv (grid 16 8 8)"
torchrun --nproc_per_node="$NPROC" \
  "${REPO_ROOT}/tools/perf_data_collection/comm_bench/generate_comm_microbench.py" \
  --database-path "$HCCL_DB" \
  --ops all_reduce all_gather reduce_scatter all_to_all \
  --grid-shape 16 8 8 \
  --num-devices 8 16 2

# ---------- 4. 命中率验证 ----------
# 参数以当前分支 `python -m cli.inference.text_generate --help` 为准
echo "[4/4] text_generate: 验证 profiling query 命中率"
python -m cli.inference.text_generate zai-org/GLM-5.2 \
  --device ATLAS_950_486T_96G \
  --num-devices 16 \
  --performance-model profiling \
  --profiling-database-path "$DB" \
  --dump-input-shapes \
  --export-empirical-metrics-file "${REPORT_DIR}/glm52_hit_report.json" \
  --num-queries 18 --query-length 4096 --context-length 0 \
  2>&1 | tee "${REPORT_DIR}/glm52_hit_report.log"

echo
echo "采集完成。下一步:"
echo "  1) 检查 ${REPORT_DIR}/glm52_hit_report.json 的 hit / partial / miss"
echo "  2) miss 多的算子回到 generate_shape_grid.py 补 shape 后重跑步骤 2"
echo "  3) 只读审计: python tools/perf_data_collection/find_database_anomalies.py --database-path ${DB} --output-dir ${REPORT_DIR}/anomaly"
echo "  4) 全部通过后把 MANIFEST.json 的 status 改为 COLLECTED"
