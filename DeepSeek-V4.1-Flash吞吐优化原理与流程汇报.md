# DeepSeek-V4.1-Flash 吞吐优化原理与流程汇报（以 msmodeling throughput_optimizer 为例）

> 对象：懂技术的领导
> 目的：用一条真实命令串联 msmodeling（MindStudio-Modeling）中 ServingCast 的吞吐优化器，讲清"原理—框架—计算流程—时延—细节"。
> 参考命令：
>
> ```bash
> python -m cli.inference.throughput_optimizer \
>   model_configs/msmodeling/DeepSeek-V4.1-Flash \
>   --device ATLAS_950_486T_96G \
>   --num-devices 16 \
>   --input-length 4096 --output-length 1024 \
>   --enable-optimize-prefill-decode-ratio \
>   --prefill-devices-per-instance 8 --decode-devices-per-instance 8 \
>   --quantize-linear-action MXFP4 \
>   --quantize-non-expert-linear-action DISABLED \
>   --quantize-attention-action DISABLED \
>   --tpot-limit 100 --ttft-limit 20000 \
>   --batch-range 4 128
> ```

---

## 一、这套工具在解决什么问题

部署一个大模型做在线推理服务时，工程上最关心两个问题：**"能扛多少并发（吞吐/QPS）"** 和 **"用户等多久（时延/SLO）"**。二者往往冲突——把批大小（batch）调大，单卡算力利用率高、吞吐大，但每个请求的排队与计算时延也变长。

msmodeling 的 `throughput_optimizer` 是一个**不依赖真实硬件、纯靠性能仿真（Performance Simulation）**的工具：

- 它用"性能模型"在软件里算出每一层、每一种并行/批大小配置下的**单步延迟**（ms）；
- 再用公式把延迟换算成**吞吐（token/s、req/s）**；
- 最后在给定 **SLO 时延上限（tpot-limit / ttft-limit）** 和 **硬件规模（--num-devices）** 下，自动搜索出**最优部署配置**（并行切分、批大小、PD 配比等）。

一句话：**给定模型 + 硬件 + 时延红线，自动给出"怎么摆卡、怎么调度最划算"的建议。** 它属于 msmodeling 三大组件中的 **ServingCast（系统级推理服务仿真与吞吐优化）**。

---

## 二、整体原理（三层核心思想）

### 2.1 原理一：Roofline 性能模型（算得快不快，由"算力 vs 带宽"决定）

每一层、每一个算子，工具都按 Roofline 模型拆成 **四个时间分量**，取最长（瓶颈）作为该算子耗时：

| 分量 | 含义 | 估算方式（等价公式） |
|---|---|---|
| **Cube（计算/矩阵乘）** | 矩阵乘法类算力 | `mma_ops / (设备峰值mma算力 × 计算效率)` |
| **Vec（逐元素/向量）** | 激活、归一化、SwiGLU 等 | `gp_ops / (设备峰值gp算力 × 计算效率)` |
| **Mem（访存/HBM）** | 权重、激活、KV 读写 | `字节数 / (HBM带宽 × 内存效率)` |
| **Comm（通信）** | 张量/专家并行集合通信 | `通信量 / 互联拓扑有效带宽` |

最终单算子时间 = `max(计算时间, 访存时间, 通信时间)`（再叠加算子 launch 的静态开销）。
代码位置：`tensor_cast/performance_model/__init__.py:2543-2625`（`_estimate_default_without_static_cost` / `_estimate_default`）、`bound_analyzer.py:120-145`（`dominant()` 取 max 并分类为 Cube/Vec/Comm/Mem 四桶）。

> 这与教科书 Roofline 数学等价：**时间 = max(运算量/峰值算力, 数据量/内存带宽)**。算子"算术强度"（ops/byte）高于机器平衡点则计算受限，反则访存受限。代码没有显式写 `ai = ops/bytes`，而是分别算 compute_time 与 memory_time 再取 max，本质一样。

### 2.2 原理二：Prefill / Decode 分阶段建模（两类任务时延模型完全不同）

LLM 推理分两个阶段，时延结构截然不同：

- **Prefill（预处理）**：一次性把整段输入（4096 token）喂进去，是**计算密集型大矩阵乘**，产出第一个 token。其时延近似为"整批输入 token 的总计算量 / 算力"，对应指标 **TTFT（Time To First Token，首 token 时延）**。
- **Decode（解码）**：自回归逐 token 生成（共 1024 个输出 token），每步只算 1 个新 token，是**访存/带宽受限**（权重反复从 HBM 读），对应指标 **TPOT（Time Per Output Token，每输出 token 时延）**。

因为二者瓶颈不同（prefill 偏算力、decode 偏带宽），**把它们放在同一批卡上会互相拖累**，于是引出下一层原理。

### 2.3 原理三：PD 分离 + 配比寻优（Prefill-Decode Disaggregation）

把 Prefill 和 Decode 拆到**不同的实例（不同卡组）**上独立部署与独立寻优，再用一个 **PD Ratio（解码吞吐 / 预处理吞吐）** 把两边"对齐"：

- 给定总卡数（16 张），按 `PD Ratio = D_QPS / P_QPS` 决定**几张卡做 Prefill、几张做 Decode**，使两端都不闲置（木桶短板 = `min(P_QPS, D_QPS)`）。
- 本案用 `--enable-optimize-prefill-decode-ratio`，并显式指定每实例 8 卡做 P、8 卡做 D，因此 16 卡正好拆成 **1 个 Prefill 实例 + 1 个 Decode 实例**。

---

## 三、框架内容（代码组成）

| 模块 | 文件 | 职责 |
|---|---|---|
| 命令入口 | `cli/inference/throughput_optimizer.py` | 参数解析、校验、分发（普通寻优 / 容量规划 / PD 分离） |
| 参数注册 | `cli/registry/modules/throughput_optimizer.py` | 所有 CLI 参数与默认值声明 |
| 基类搜索 | `serving_cast/service/base_throughput_optimizer.py` | 二分搜索 batch/并发、early-stop、逐行延迟计算 |
| 聚合模式 | `serving_cast/service/agg_throughput_optimizer.py` | P+D 同实例（不分离）的吞吐/QPS |
| 分离模式 | `serving_cast/service/disagg_throughput_optimizer.py` | P、D 各自独立的 TTFT/TPOT 公式 |
| PD 配比 | `serving_cast/service/pd_ratio_throughput_optimizer.py` | cross join P/D 结果、算 pd_ratio 与 balanced_qps |
| 并行候选 | `serving_cast/parallel_runner.py` | 枚举 TP/EP/PP/MOE-DP 可行组合、并发计算 |
| 结果汇总 | `serving_cast/service/optimizer_summary.py` | best row 选取、TopN 表格、硬件对比、SLO 过滤 |
| 性能模型 | `tensor_cast/performance_model/` | Roofline（analytic）逐算子 Cube/Vec/Comm/Mem |
| 设备画像 | `tensor_cast/device.py` | 芯片算力、显存带宽、互联拓扑、静态开销 |
| 量化配置 | `tensor_cast/core/quantization/config.py` | 把量化参数映射到各层 Linear 的 dtype |

---

## 四、本案参数逐项解读

| 参数 | 取值 | 含义与影响 |
|---|---|---|
| `model_configs/.../DeepSeek-V4.1-Flash` | 模型配置目录 | 模型结构来源（`config.json`：`deepseek_v41`，40 层，隐藏 5120，384 路由专家 + 1 共享专家，topk=6，MLA 注意力） |
| `--device ATLAS_950_486T_96G` | A950 芯片 | 单卡：C486T 算力（bf16 432 / fp8 865 / fp4 1730 TFLOPS）、96GB HBM@4.0TB/s、POD_1K 互联 |
| `--num-devices 16` | 16 卡 | 集群总卡数；PD 分离下拆给 P/D 实例 |
| `--input-length 4096` | 输入 4096 token | Prefill 阶段处理长度，决定 TTFT 计算 |
| `--output-length 1024` | 输出 1024 token | Decode 阶段步数，决定 TPOT 乘子与 D_QPS |
| `--enable-optimize-prefill-decode-ratio` | 开 | 启用 PD 分离 + 配比寻优模式 |
| `--prefill-devices-per-instance 8` | P 实例 8 卡 | 每个 Prefill 实例占 8 张卡 |
| `--decode-devices-per-instance 8` | D 实例 8 卡 | 每个 Decode 实例占 8 张卡 |
| `--quantize-linear-action MXFP4` | 线性层 MXFP4 | **广覆盖**：所有 Linear 层权重/激活用 FP4（MXFP4），group=32，PER_GROUP |
| `--quantize-non-expert-linear-action DISABLED` | 非专家不单独覆盖 | **注意**：这里"DISABLED"只是"不单独覆盖非专家层"，非专家层仍落入上面的广覆盖 MXFP4（K3 专属的"DISABLED 即排除"逻辑不适用于 DSV4.1）。结果：**注意力投影、稠密 MLP、共享专家、路由专家全部 MXFP4** |
| `--quantize-attention-action DISABLED` | 注意力不量化 | KV Cache 保持 bf16，不做注意力/KV 量化 |
| `--tpot-limit 100` | 解码时延上限 100ms | 每行结果要求 `TPOT ≤ 100ms`，否则被 SLO 过滤 |
| `--ttft-limit 20000` | 首 token 上限 20s | 每行结果要求 `TTFT ≤ 20000ms` |
| `--batch-range 4 128` | 批大小搜索区间 [4,128] | 在此区间二分搜索最优 batch（每实例并发 = batch × DP） |

> **量化要点澄清**：MXFP4（FP4）相比 fp8、bf16，在 C486T 上 **MMA 峰值算力翻倍**（1730 vs 865 vs 432 TFLOPS）。对 decode（访存受限、权重反复读）尤其关键——权重从 fp8 的 1 字节降到 fp4 的 0.5 字节，**显存占用与带宽需求减半**，直接缓解 decode 瓶颈。本案把全部 Linear 层都压成 MXFP4，正是为最大化 16 卡下的吞吐。

---

## 五、以 DeepSeek-V4.1-Flash 为例的端到端计算流程

### 5.1 设备画像（ATLAS 950 486T 96G）

来自 `tensor_cast/device.py:815-825`（`A5.A950_486T_96G`）：

- **芯片 C486T**（`device.py:451-466`）：
  - MMA 峰值：fp32(HF32) 216T、bf16/half **432T**、fp8/int8 **865T**、**fp4(MXFP4) 1730T** FLOPS
  - GP（向量）峰值：bf16 54T
  - 计算效率 `compute_efficiency = 0.9`
- **显存 M96G_4_0T**（`device.py:504-508`）：容量 **96 GB**，带宽 **4.0 TB/s**，内存效率 `0.8`
- **互联**：`Interconnect.POD_1K`（整柜级 1K 互联拓扑，用于张量/专家并行通信估算）
- **静态开销**：`mma_op_cost_s = 5e-6 s`（每次大算子 launch 的固定时延，叠加在 roofline 之上）

### 5.2 模型结构（DSV4.1，来自 `config.json`）

- 文本主干 `deepseek_v41`：40 层，隐藏维度 5120，词表 129280
- **注意力 MLA**：`num_key_value_heads=1`（极致压缩 KV），`head_dim=512`，带 `q_lora_rank=1280 / o_lora_rank=1024` 的低秩投影
- **MoE**：`n_routed_experts=384`、`n_shared_experts=1`、`num_experts_per_tok=6`（每 token 激活 6 个路由专家 + 1 共享专家）
- 还有 DSpark 投机解码（`dspark_block_size=5`）等加速结构（本案未启用 `--speculative-method`，走基线）

每层计算量 ≈ 注意力（MLA 投影 + 注意力）+ 稠密 MLP（gate/up/down）+ MoE（6/384 专家的门控与 FFN）+ 共享专家。性能模型会对**每一类 Linear 按 dtype（本案全 MXFP4→fp4→1730T）**估算 mma_ops_time，并叠加访存时间。

### 5.3 量化 → 算力映射（为什么选 MXFP4）

性能模型在估算矩阵乘时，按 **tensor 的 dtype** 去查设备峰值（`tensor_cast/performance_model/__init__.py:2554` + `_get_device_ops_for_dtype`）：

- bf16 Linear → 432T；fp8 Linear → 865T；**MXFP4 Linear（以 `int4` 载体承载 fp4 语义）→ 1730T**
- 因此本案把全部 Linear 设为 MXFP4 后，**所有矩阵乘的可达算力上限翻到 1730T**（相对 fp8 的 2 倍、bf16 的 4 倍）。
- 同时 fp4 权重只占 fp8 一半字节，decode 阶段的 HBM 读写量减半，访存时间也随之下降。

### 5.4 PD 分离寻优主流程

1. **参数解析与校验**（`throughput_optimizer.py:50-237`）：确认 PD 模式两参数齐全（8/8），校验并行组合能整除 16，估算搜索组合数。
2. **两阶段独立寻优**（`parallel_runner.py:_run_pd_ratio`）：
   - **P 阶段**：以 8 卡为实例，设 `tpot_limit=None, ttft_limit=20000`，只关心首 token 时延；
   - **D 阶段**：以 8 卡为实例，设 `tpot_limit=100, ttft_limit=inf`，只关心每 token 时延。
   - 每个阶段内部，对**可行的并行组合**（TP/EP/MOE-DP，需整除 8 卡且 EP token 域守恒 `ep×moe_dp = tp×dp`）和 **batch ∈ [4,128]** 做**二分搜索**（`base_throughput_optimizer.py:285-299`），用 early-stop 在超出 SLO/OOM 时收缩上界。
3. **逐配置算延迟**：每对 (batch, 并行) 调 `_get_forward_info` → 性能模型算出 `execution_time_s`，拆成 Cube/Vec/Comm/Mem 占比（`base_throughput_optimizer.py:543,701-742`）。
4. **算 QPS**（见 5.5），得到 P 阶段结果表 `p_df` 与 D 阶段结果表 `d_df`。

### 5.5 时延与吞吐公式（核心，来自 `disagg_*` / `pd_ratio_*`）

- **TTFT（Prefill）**：`TTFT ≈ 整批输入总时延`（首组无排队）。P 阶段 QPS：
  `P_QPS = p_concurrency / prefill_phase_makespan_ms × 1000`（回退用 TTFT）
- **TPOT（Decode）**：`TPOT = 单步解码延迟(ms)`（已含投机解码 fold）。D 阶段 QPS：
  `D_QPS = d_concurrency / (TPOT × max(output_length − 1, 1)) × 1000`
  （本案 `output_length=1024` ⇒ 分母约 `TPOT × 1023`）
- **并发关系**：`concurrency = batch_size × DP`（PP 只切分单次执行，不乘；`agg/disagg_optimizer` 中 `concurrency = batch_size * self.dp`）
- **PD Ratio 与平衡吞吐**（`pd_ratio_throughput_optimizer.py:64-153`）：
  `pd_ratio = D_QPS / P_QPS`，`balanced_qps = min(P_QPS, D_QPS)`
  （木桶效应：整体吞吐受较慢一端限制）

### 5.6 SLO 约束（tpot-limit / ttft-limit）

三道关卡（`optimizer_summary.py`）：
1. **搜索边界剪枝**：用 TTFT/TPOT 斜率 × 1.5 松弛预测安全 batch 上限；
2. **逐行 early-stop**：若某 batch 下 `TPOT > 100ms` 或 `TTFT > 20000ms`，直接收缩二分上界；
3. **结果过滤与排序**：`mask = (ttft ≤ 20000) & (tpot ≤ 100)` 过滤后，组内以"吞吐降序、SLO 余量（`max(ttft/limit, tpot/limit)`）最大"选 best row。

### 5.7 PD Ratio 与实例分配（总 16 卡）

给定 `pd_ratio = D_QPS / P_QPS`，供需平衡要求 `P实例数 × P_QPS ≈ D实例数 × D_QPS` ⇒ `P实例数 = D实例数 × pd_ratio`（`optimizer_summary.py:_calculate_instance_distribution`）。
本案每实例固定 8 卡（P=8、D=8）且总卡 16，枚举结果只能是 **1 个 Prefill 实例 + 1 个 Decode 实例**（8+8=16），工具据此报出：
`allocated_p_qps / allocated_d_qps / balanced_qps / 实际占用卡数 = 16`。
（若改用 `--target-qps` 自动模式，则会反过来自动搜索最优的 `(p_dev, d_dev)` 配比。）

### 5.8 输出报告（`optimizer_summary.py`）

- **Overall Best Configuration**：最佳吞吐、TTFT、TPOT、并发、并行标签（`TP=x|EP=x|MOE-TP=x|MOE-DP=x|...`）、batch、显存占用（weight/kv/activation/avail GB）；
- **Top N 配置表**：多组可行 (P,D) 配置的 PD Ratio / Balanced QPS / TTFT / TPOT / 并行 / 设备 / 批 / 并发；
- **硬件画像对比**：PrettyTable 输出 Cube/Vector TFLOPS（峰值×效率）、HBM 带宽（×效率）、显存、互联带宽。

---

## 六、时延 / 吞吐公式速查表

| 指标 | 公式 | 本案取值/约束 |
|---|---|---|
| 单算子时间 | `max(计算, 访存, 通信) + 静态开销` | roofline，按 dtype 查峰值 |
| MXFP4 算力 | `1730 TFLOPS × 0.9` | 全部 Linear 层 |
| TTFT | `prefill 整批时延(ms)` | `≤ 20000 ms` |
| TPOT | `decode 单步时延(ms)` | `≤ 100 ms` |
| P_QPS | `p_concurrency / makespan_ms × 1000` | P 阶段（8 卡） |
| D_QPS | `d_concurrency / (TPOT × 1023) × 1000` | D 阶段（8 卡） |
| pd_ratio | `D_QPS / P_QPS` | 配比依据 |
| balanced_qps | `min(P_QPS, D_QPS)` | 整体有效吞吐 |
| 并发 | `batch_size × DP` | batch ∈ [4,128] 二分 |
| 卡分配 | `P实例×8 + D实例×8 ≤ 16` | ⇒ 1P + 1D |

---

## 七、价值与结论

1. **不需要真卡就能算部署方案**：纯仿真在复现"算力/带宽/通信"瓶颈的同时，给出可落地的并行切分与批大小建议，节省大量真机调试成本。
2. **MXFP4 是本案吞吐杠杆**：把全部 Linear 压到 FP4，C486T 上 MMA 峰值翻 2 倍、权重带宽减半，重点缓解 decode（访存受限）瓶颈；KV Cache 保持 bf16（注意力不量化）以保精度。
3. **PD 分离避免相互拖累**：prefill（算力受限）与 decode（带宽受限）拆到不同 8 卡实例，按 `pd_ratio` 对齐两端，16 卡得到 1P+1D 的平衡部署。
4. **SLO 红线内自动寻优**：tpot 100ms / ttft 20s 作为硬约束贯穿搜索与过滤，保证给出的配置"既快又满足时延体验"。

> 备注：本汇报基于 msmodeling 源码（`tensor_cast/` 与 `serving_cast/`）静态梳理。具体数值（最佳吞吐、TTFT、TPOT、显存占用）需执行该命令在真实仿真中得出；命令默认 `--performance-model analytic`（Roofline 解析模型），如需更贴近实测可叠加 `--performance-model profiling`（基于实测算子库）或 `calibrated`（标定模型）。
