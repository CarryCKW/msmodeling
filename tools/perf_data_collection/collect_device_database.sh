#!/usr/bin/env bash
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# ============================================================
# 通用设备性能数据库采集入口（Linux + 昇腾 NPU + Docker）
#
# 目标：在目标 NPU 设备上执行本脚本，按“解析 -> 扩样 -> 回写 -> HCCL 通信”
#       四个阶段采集，产出标准目录：
#         tensor_cast/performance_model/profiling_database/data/<DEVICE>/
#           ├── vllm_ascend/<vllm>_torch<torch>_cann<cann>/   （计算算子 CSV + op_mapping.yaml）
#           └── hccl/<cann>/                                   （hcom_*.csv）
#
# 设备范围：仅接受已在 tensor_cast/device.py 注册的 DeviceProfile.name，
#           例如 ATLAS_800_A2_376T_64G / ATLAS_800_A2_280T_64G / ATLAS_950_486T_96G。
#           注意：脚本全程用显式 --database-path 传递目录，因此不受
#           op_replay/common.py 中 SUPPORTED_DEVICES 白名单限制（ATLAS_950 不在白名单也可用）。
#
# 重要前提：
#   1) 必须在带昇腾 NPU 的 Linux 容器/主机运行，并已加载 CANN 环境
#      （source /usr/local/Ascend/ascend-toolkit/set_env.sh）；
#   2) 每个阶段是否需要由开关控制，可只跑其中一段；
#   3) op_mapping.yaml 不由 parse 脚本生成：目标 compute 目录必须已有
#      op_mapping.yaml（用 op-mapping Skill 生成，或从已有同软件栈版本目录拷贝后
#      修改 device 字段）。缺失时会明确报错，不会静默产出无效库。
#
# 详见同目录 README_collect_device_database.md 与 tools/perf_data_collection/readme.md。
# ============================================================

set -euo pipefail

# ---------- 路径解析（脚本位于 tools/perf_data_collection/） ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "${TOOLS_DIR}/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/tensor_cast/performance_model/profiling_database/data"

# ---------- 默认参数 ----------
DEVICE=""
PROFILING_PATH=""
VLLM_VERSION=""
TORCH_VERSION=""
CANN_VERSION=""
VERSION_DIR=""
HCCL_CANN_VERSION=""
ROWS=2000
SEED=0
REPEAT_COUNT=3
NPROC=16
GRID_SHAPE=""            # 例如 "48 8 2"，留空则按设备推断，推断不出则 HCCL 阶段报错
NUM_DEVICES=""           # 例如 "16 2"，留空则 generate_comm_microbench.py 用默认 16
TARGET_MODELS=""
OPTIMIZER_ARGS_FILE=""
OPS=""
DO_PARSE=1
DO_SHAPE=1
DO_REPLAY=1
DO_HCCL=1
DRY_RUN=0
LIST_DEVICES=0
VERBOSE=0

usage() {
    cat <<'EOF'
Usage: bash tools/perf_data_collection/collect_device_database.sh [options]

通用昇腾 NPU 性能数据库采集入口。产出 data/<DEVICE>/{vllm_ascend/<version>,hccl/<cann>}/。

设备与软件栈：
  --device NAME              已注册 DeviceProfile.name（必填，除非 --list-devices）
                             例: ATLAS_800_A2_376T_64G / ATLAS_800_A2_280T_64G / ATLAS_950_486T_96G
  --list-devices             列出当前仓库已注册的设备画像后退出
  --version-dir NAME         完整版本目录名，如 vllm0.18.0_torch2.9.0_cann8.5
                             给定后忽略 --vllm/--torch/--cann-version
  --vllm-version V           vLLM-Ascend 版本（如 0.18.0）
  --torch-version V          PyTorch 版本（如 2.9.0）
  --cann-version V           CANN 版本（如 8.5）；用于 compute 版本目录名
  --hccl-cann-version V      hccl 子目录名（如 v8.5 / v9.0.1）；缺省取 "v${CANN_VERSION}"

阶段输入：
  --profiling-path PATH      NPU profiling 目录或 kernel_details*.csv（阶段 1 解析输入）
  --target-models "M1 M2"    扩样用的 HuggingFace 模型 ID（空格分隔）
  --optimizer-args-file PATH 扩样用的吞吐优化器场景 YAML/JSON（优先于 --target-models）
  --rows N                   扩样每算子最多新增行数（默认 2000）
  --seed N                   扩样随机种子（默认 0）

阶段控制：
  --ops "OP1 OP2"            只处理指定 kernel type（空格分隔）
  --repeat-count N           replay 重复次数（默认 3）
  --nproc N                  HCCL torchrun 每节点进程数（默认 16）
  --grid-shape A B C         HCCL 硬件网格（默认: A3_752T=48 8 2, ATLAS_950=16 8 8；其它设备需显式给）
  --num-devices "N1 N2"      HCCL 通信组设备数（空格分隔）
  --skip-parse               跳过阶段 1（已有 per-operator CSV 时）
  --skip-shape               跳过阶段 2（扩样）
  --skip-replay              跳过阶段 3（NPU replay 回写）
  --skip-hccl                跳过阶段 4（HCCL 通信采集）

运行方式：
  --python-bin PATH          解释器（默认 python3），需能 import torch/torch_npu
  --dry-run                  只打印将执行的命令，不实际运行
  -v, --verbose              打印详细信息
  -h, --help                 显示本帮助

示例：
  # ATLAS_950：先 source CANN 环境，再一次性跑四阶段
  bash tools/perf_data_collection/collect_device_database.sh \
    --device ATLAS_950_486T_96G \
    --version-dir vllm0.23.0rc1_torch2.10.0_cann9.0.1 \
    --hccl-cann-version v9.0.1 \
    --profiling-path ./PROF_001 \
    --optimizer-args-file tensor_cast/performance_model/profiling_database/data/ATLAS_950_486T_96G/workloads_glm52.yaml

  # A2：只补采 HCCL（compute 数据已就绪）
  bash tools/perf_data_collection/collect_device_database.sh \
    --device ATLAS_800_A2_376T_64G \
    --version-dir vllm0.18.0_torch2.9.0_cann8.5 \
    --grid-shape 8 16 \
    --skip-parse --skip-shape --skip-replay
EOF
}

log()  { echo "[collect] $*"; }
warn() { echo "[collect][WARN] $*" >&2; }
die()  { echo "[collect][ERROR] $*" >&2; exit 1; }

# 通用命令执行器：dry-run 下只打印
run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# ---------- 参数解析 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)              DEVICE="${2:?missing value for --device}"; shift 2 ;;
        --list-devices)        LIST_DEVICES=1; shift ;;
        --version-dir)         VERSION_DIR="${2:?missing value for --version-dir}"; shift 2 ;;
        --vllm-version)        VLLM_VERSION="${2:?missing value for --vllm-version}"; shift 2 ;;
        --torch-version)       TORCH_VERSION="${2:?missing value for --torch-version}"; shift 2 ;;
        --cann-version)        CANN_VERSION="${2:?missing value for --cann-version}"; shift 2 ;;
        --hccl-cann-version)   HCCL_CANN_VERSION="${2:?missing value for --hccl-cann-version}"; shift 2 ;;
        --profiling-path)      PROFILING_PATH="${2:?missing value for --profiling-path}"; shift 2 ;;
        --target-models)       TARGET_MODELS="${2:?missing value for --target-models}"; shift 2 ;;
        --optimizer-args-file) OPTIMIZER_ARGS_FILE="${2:?missing value for --optimizer-args-file}"; shift 2 ;;
        --rows)                ROWS="${2:?missing value for --rows}"; shift 2 ;;
        --seed)                SEED="${2:?missing value for --seed}"; shift 2 ;;
        --ops)                 OPS="${2:?missing value for --ops}"; shift 2 ;;
        --repeat-count)        REPEAT_COUNT="${2:?missing value for --repeat-count}"; shift 2 ;;
        --nproc)               NPROC="${2:?missing value for --nproc}"; shift 2 ;;
        --grid-shape)
            shift
            GRID_SHAPE=""
            while [[ $# -gt 0 && "$1" != --* ]]; do GRID_SHAPE="${GRID_SHAPE:+$GRID_SHAPE }$1"; shift; done
            ;;
        --num-devices)         NUM_DEVICES="${2:?missing value for --num-devices}"; shift 2 ;;
        --python-bin)          PYTHON_BIN="${2:?missing value for --python-bin}"; shift 2 ;;
        --skip-parse)          DO_PARSE=0; shift ;;
        --skip-shape)          DO_SHAPE=0; shift ;;
        --skip-replay)         DO_REPLAY=0; shift ;;
        --skip-hccl)           DO_HCCL=0; shift ;;
        --dry-run)             DRY_RUN=1; shift ;;
        -v|--verbose)          VERBOSE=1; shift ;;
        -h|--help)             usage; exit 0 ;;
        *)                     die "unknown option: $1 (use --help)" ;;
    esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---------- 列设备模式 ----------
if [[ "$LIST_DEVICES" -eq 1 ]]; then
    "$PYTHON_BIN" - <<'PY'
from tensor_cast.device import DeviceProfile
from tools.perf_data_collection.op_replay.common import SUPPORTED_DEVICES
print("Registered DeviceProfiles:")
for name in sorted(DeviceProfile.all_device_profiles):
    flag = " [in SUPPORTED_DEVICES]" if name in SUPPORTED_DEVICES else ""
    print(f"  - {name}{flag}")
PY
    exit 0
fi

[[ -n "$DEVICE" ]] || die "--device is required (see --list-devices / --help)"

# ---------- 前置检查 ----------
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python not found: $PYTHON_BIN"

# 设备必须是已注册的 DeviceProfile.name
"$PYTHON_BIN" - "$DEVICE" <<'PY'
import sys
from tensor_cast.device import DeviceProfile
name = sys.argv[1]
if name not in DeviceProfile.all_device_profiles:
    raise SystemExit(
        f"device {name!r} is not a registered DeviceProfile.name; "
        "register it in tensor_cast/device.py first"
    )
PY

# ---------- 推导版本目录 ----------
if [[ -z "$VERSION_DIR" ]]; then
    if [[ -n "$VLLM_VERSION" && -n "$TORCH_VERSION" && -n "$CANN_VERSION" ]]; then
        VERSION_DIR="vllm${VLLM_VERSION}_torch${TORCH_VERSION}_cann${CANN_VERSION}"
    else
        log "版本未显式给出，尝试从运行时探测 vLLM-Ascend / torch / CANN 版本..."
        DETECTED="$("$PYTHON_BIN" - <<'PY' 2>/dev/null || true
from tools.perf_data_collection.op_replay.common import detect_runtime_stack_versions
v, t, c = detect_runtime_stack_versions()
print(f"{v or ''} {t or ''} {c or ''}")
PY
)"
        read -r dv dt dc <<< "$DETECTED"
        VLLM_VERSION="${VLLM_VERSION:-$dv}"
        TORCH_VERSION="${TORCH_VERSION:-$dt}"
        CANN_VERSION="${CANN_VERSION:-$dc}"
        [[ -n "$VLLM_VERSION" && -n "$TORCH_VERSION" && -n "$CANN_VERSION" ]] \
            || die "cannot auto-detect stack versions; pass --version-dir or the three --*-version options"
        VERSION_DIR="vllm${VLLM_VERSION}_torch${TORCH_VERSION}_cann${CANN_VERSION}"
    fi
fi

if [[ -z "$HCCL_CANN_VERSION" ]]; then
    if [[ -n "$CANN_VERSION" ]]; then
        HCCL_CANN_VERSION="v${CANN_VERSION#v}"
    else
        # 从 VERSION_DIR 中解析 cann 段，如 ..._cann8.5 -> v8.5
        raw="${VERSION_DIR##*_cann}"
        HCCL_CANN_VERSION="v${raw#v}"
    fi
fi

DB_DIR="${DATA_ROOT}/${DEVICE}/vllm_ascend/${VERSION_DIR}"
HCCL_DIR="${DATA_ROOT}/${DEVICE}/hccl/${HCCL_CANN_VERSION}"

log "DEVICE      = ${DEVICE}"
log "VERSION_DIR = ${VERSION_DIR}"
log "DB_DIR      = ${DB_DIR}"
log "HCCL_DIR    = ${HCCL_DIR}"
log "stages      = parse:${DO_PARSE} shape:${DO_SHAPE} replay:${DO_REPLAY} hccl:${DO_HCCL} dry-run:${DRY_RUN}"

# ---------- 推断 HCCL grid-shape ----------
if [[ -z "$GRID_SHAPE" ]]; then
    case "$DEVICE" in
        ATLAS_800_A3_752T_128G_DIE) GRID_SHAPE="48 8 2" ;;
        ATLAS_950_486T_96G)         GRID_SHAPE="16 8 8" ;;
        *)
            if [[ "$DO_HCCL" -eq 1 ]]; then
                die "--grid-shape is required for ${DEVICE}; set it to the real hardware topology (see tensor_cast/device.py comm_grid)"
            fi
            ;;
    esac
    [[ -n "$GRID_SHAPE" ]] && log "grid-shape  = ${GRID_SHAPE} (inferred)"
fi

# ---------- 创建目录骨架 ----------
mkdir -p "$DB_DIR" "$HCCL_DIR"

# op_mapping.yaml 必须预先存在（parse 脚本不会生成它）
if [[ "$DO_REPLAY" -eq 1 || "$DO_SHAPE" -eq 1 ]]; then
    if [[ ! -f "${DB_DIR}/op_mapping.yaml" ]]; then
        die "missing ${DB_DIR}/op_mapping.yaml. Generate it with the op-mapping Skill, or copy from an existing stack dir and set 'device: ${DEVICE}'. See README_collect_device_database.md"
    fi
    "$PYTHON_BIN" - "${DB_DIR}/op_mapping.yaml" "$DEVICE" <<'PY'
import sys, yaml
path, device = sys.argv[1], sys.argv[2]
mapping = yaml.safe_load(open(path, encoding="utf-8")) or {}
if mapping.get("device") != device:
    raise SystemExit(
        f"op_mapping.yaml device={mapping.get('device')!r} != --device {device!r}; "
        "fix the device field before collecting"
    )
PY
fi

# ---------- 阶段 1：解析 profiling ----------
if [[ "$DO_PARSE" -eq 1 ]]; then
    [[ -n "$PROFILING_PATH" ]] || die "--profiling-path is required for the parse stage (or use --skip-parse)"
    [[ -e "$PROFILING_PATH" ]] || die "profiling path not found: $PROFILING_PATH"
    log "[1/4] parse_kernel_details.py -> ${DB_DIR}"
    run_cmd "$PYTHON_BIN" "${TOOLS_DIR}/parsers/parse_kernel_details.py" \
        --profiling-path "$PROFILING_PATH" \
        --database-path "$DB_DIR"
fi

# ---------- 阶段 2：扩样 theory shape ----------
if [[ "$DO_SHAPE" -eq 1 ]]; then
    log "[2/4] generate_shape_grid.py -> ${DB_DIR}"
    shape_args=(--database-path "$DB_DIR" --rows "$ROWS" --seed "$SEED")
    if [[ -n "$OPTIMIZER_ARGS_FILE" ]]; then
        [[ -e "$OPTIMIZER_ARGS_FILE" ]] || die "optimizer args file not found: $OPTIMIZER_ARGS_FILE"
        shape_args+=(--optimizer-args-file "$OPTIMIZER_ARGS_FILE")
    elif [[ -n "$TARGET_MODELS" ]]; then
        # shellcheck disable=SC2206
        model_arr=($TARGET_MODELS)
        shape_args+=(--target-models "${model_arr[@]}")
    else
        die "扩样阶段需要 --optimizer-args-file 或 --target-models（或用 --skip-shape 跳过）"
    fi
    if [[ -n "$OPS" ]]; then
        # shellcheck disable=SC2206
        op_arr=($OPS)
        shape_args+=(--ops "${op_arr[@]}")
    fi
    run_cmd "$PYTHON_BIN" "${TOOLS_DIR}/generate_shape_grid.py" "${shape_args[@]}"
fi

# ---------- 阶段 3：NPU replay 回写 ----------
if [[ "$DO_REPLAY" -eq 1 ]]; then
    log "[3/4] start_microbench.py (msprof replay + 回写) -> ${DB_DIR}"
    replay_args=(--database-path "$DB_DIR" --repeat-count "$REPEAT_COUNT"
                 --update-mode missing-only --fail-fast)
    if [[ -n "$OPS" ]]; then
        # shellcheck disable=SC2206
        op_arr=($OPS)
        replay_args+=(--ops "${op_arr[@]}")
    fi
    if [[ "$DRY_RUN" -eq 0 ]]; then
        command -v msprof >/dev/null 2>&1 || die "msprof not found; load CANN env: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    fi
    run_cmd "$PYTHON_BIN" "${TOOLS_DIR}/start_microbench.py" "${replay_args[@]}"
fi

# ---------- 阶段 4：HCCL 通信采集 ----------
if [[ "$DO_HCCL" -eq 1 ]]; then
    log "[4/4] generate_comm_microbench.py -> ${HCCL_DIR}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        command -v torchrun >/dev/null 2>&1 || die "torchrun not found; activate an env with torch/torch_npu"
    fi
    # shellcheck disable=SC2206
    grid_arr=($GRID_SHAPE)
    hccl_args=(--database-path "$HCCL_DIR"
               --ops all_reduce all_gather reduce_scatter all_to_all
               --grid-shape "${grid_arr[@]}")
    if [[ -n "$NUM_DEVICES" ]]; then
        # shellcheck disable=SC2206
        nd_arr=($NUM_DEVICES)
        hccl_args+=(--num-devices "${nd_arr[@]}")
    fi
    run_cmd torchrun --nproc_per_node="$NPROC" \
        "${TOOLS_DIR}/comm_bench/generate_comm_microbench.py" "${hccl_args[@]}"
fi

echo
log "DONE. 采集目录："
log "  compute : ${DB_DIR}"
log "  hccl    : ${HCCL_DIR}"
log "下一步（发布前）："
log "  1) python tools/perf_data_collection/find_database_anomalies.py --database-path ${DB_DIR} --output-dir <report_dir>"
log "  2) python -m cli.inference.text_generate <model> --device ${DEVICE} --performance-model profiling --profiling-database-path ${DB_DIR} --dump-input-shapes"
log "  3) 按 profiling-database-lifecycle Skill 审计 + 命中率验证后再合入"
