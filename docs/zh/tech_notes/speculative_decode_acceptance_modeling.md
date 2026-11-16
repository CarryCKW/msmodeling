# 投机解码 Decode 吞吐收益模拟原理与 DSpark 接受期望建模

> 适用范围：`--speculative-method mtp` / `dspark`（以及 `dflash`）在 TensorCast / ServingCast 中的 decode 吞吐折叠逻辑。
> 目标：说明底层运行原理与数学思想，并回答"MTP 的接受率 α_i 是位置条件的、DSpark 是块级的，若用 MTP 的 α_i 拟合'等效块存活概率'会把两种粒度混为一谈，应如何建模 DSpark 的接受期望、内部是否有相关框架"。

---

## 1. 仿真入口与统一折叠公式

三类投机方法在 ServingCast 吞吐优化器里最终都汇聚到同一个折叠函数：

```612:652:serving_cast/service/base_throughput_optimizer.py
@staticmethod
def _fold_decode_latency_ms(latency_ms: float, optimizer_data: OptimizerData) -> float:
    """Apply speculative decode fold (mutually exclusive; never combine).

    - DSpark / Dflash: unified clamp to n (= block-1), ``latency / (accept + 1)``.
    - New MTP (``speculative_method == 'mtp'``): ``latency / (clamp(accept, 0, n) + 1)``.
    - Legacy MTP (no speculative_method): ``latency / (sum(rates[:N]) + 1)``.
    """
    ...
    # DSpark / DFlash: latency / (accept + 1), accept clamped to [0, block-1]
    # New MTP: latency / (accept + 1), accept clamped to [0, n]
    # Legacy MTP: latency / (sum(mtp_acceptance_rate[:N]) + 1)
```

**核心数学思想**：一个投机 step 的成本是"1 次 target 前向 + 1 次 draft 前向"，但平均产出 `1 + E[accepted]` 个 token。单位输出 token 的 decode 延迟为

\[
\text{latency\_per\_token} = \frac{\text{latency\_ms}}{1 + E[\text{accepted}]}
\]

因此所有折叠只是把 `latency_ms` 除以 `1 + E[accepted]` 的不同取值。`accept` 这个标量就是 `E[accepted]` 的某种近似。MTP 与 DSpark 的根本差异，就藏在 `E[accepted]` 该怎么算上。

参数解析与默认值的来源：

```519:535:tensor_cast/model_config.py
# DsparkConfig:
dspark_acceptance_length: float = 5.0          # 默认 5.0
# __post_init__ 中 clamp 到 [0, block_size - 1]
# 解析时 resolve_draft_block_and_acceptance 也会 clamp_acceptance_length
```

```637:647:serving_cast/service/utils.py
# 标签/默认 also shows default accept = 5.0 when None (base_optimizer._fold_decode_latency_ms)
```

---

## 2. MTP 的底层模拟原理

### 2.1 计算建模（前向）

MTP 的计算建模在 `tensor_cast/layers/mtp.py`：`MultiTokenPredictor` 叠加 N 个 MTP 层（每个 = 一个 decoder block + lm_head），`MtpWrapper.forward` 在 target 产出 anchor 后，逐层把"上一步输出 token 当作下一步输入"前向 N 次，每层产出一个 proposal logits。

```352:409:tensor_cast/layers/mtp.py
def forward(self, input_ids, position_ids, inputs_embeds=None, **kwargs):
    ...
    logits, hidden_states = self._inner(..., output_intermediate_hidden_states=True, **kwargs)
    next_tokens = self.sampler(logits, sampling_metadata)
    # 假设所有投机 token 都被接受，把最后一个返回 token 喂给下一步
    output = torch.empty([next_tokens.size(0), self.mtp_config.num_mtp_layers + 1], ...)
    output[:, 0] = next_tokens[:, -1]
    for i in range(self.mtp_config.num_mtp_layers):
        logits, hidden_states = self.mtp.forward(input_ids, position_ids, hidden_states, ...)
        next_tokens = self.sampler(logits, sampling_metadata, mode="proposal")
        output[:, i + 1] = next_tokens[:, -1]
    return output
```

关键点：MTP 的**计算**是逐位置（per-position）独立的——每一层 MTP 块只负责预测"下一个位置"的 token，逐位串行。这决定了它的接受校验天然是"逐 token 校验"。

### 2.2 吞吐折叠（接受率建模）

MTP 在两处折叠，粒度不同：

- **Legacy MTP**（无 `speculative_method`）：`accept = sum(mtp_acceptance_rate[:N])`，即把每个位置的接受率 α_i 直接求和：

```648:652:serving_cast/service/base_throughput_optimizer.py
# Legacy MTP path
num_mtp_tokens = optimizer_data.num_mtp_tokens or 0
mtp_acceptance_rate = optimizer_data.mtp_acceptance_rate or []
average_tokens = sum(mtp_acceptance_rate[:num_mtp_tokens]) + 1
return latency_ms / average_tokens
```

```78:78:tools/perf_data_collection/grid_generator/optimizer_spec.py
DEFAULT_MTP_ACCEPTANCE_RATES = (0.9, 0.6, 0.4, 0.2)
```

- **New MTP**（`--speculative-method mtp`）：`accept = clamp(acceptance_length, 0, n)`，n 个投机 token 退化成单一标量。

### 2.3 数学：位置条件接受率 α_i

MTP 的 α_i 在物理上定义为**条件概率**：

\[
\alpha_i = P(\text{第 } i \text{ 个投机 token 被接受} \mid \text{前 } i-1 \text{ 个都已被接受})
\]

因此"恰好接受 k 个"的概率为（链式条件）：

\[
P(\text{accept}=k) = \Bigl(\prod_{i=1}^{k}\alpha_i\Bigr)\,(1-\alpha_{k+1}),\quad k<N;\qquad
P(\text{accept}=N)=\prod_{i=1}^{N}\alpha_i
\]

期望接受数（精确值）是：

\[
E[\text{accepted}]_{\text{seq}} = \sum_{k=1}^{N}\prod_{i=1}^{k}\alpha_i
\]

注意：`code` 中 legacy 用的是 `Σ α_i`（**各位置边际接受率之和**），它只在 α_i 被当作"第 i 位被接受的边际概率（不要求前缀）"时才精确等于 `E[accepted]`。如果 α_i 是真实 MTP 跑出来的"给定前缀条件的接受率"，那么 `Σ α_i` **高估**了 `E[accepted]`（因为 `P(全对) ≤ min α_i < Π α_i` 的乘积链衰减更快）。这是 MTP 这一侧本身已存在的一个近似选择，但要注意它和 DSpark 混用时会放大误差。

由于 α_i 随位置快速衰减（默认 0.9→0.6→0.4→0.2），乘积链 `α_1·α_2·...` 远比 `Σα_i` 小——这正是"位置条件 + 快速衰减"带来的本质特性。

---

## 3. DSpark 的底层模拟原理

### 3.1 计算建模（块内串行 Markov/Confidence 采样）

DSpark 复用 DFlash 主干，但额外叠加 `MarkovHead` 与 `ConfidenceHead`，并在**块内**做串行采样：

```136:167:tensor_cast/layers/dspark.py
class DsparkDraftModel(DflashDraftModel):
    # 叠加 MarkovHead（vanilla/gated/rnn）与 ConfidenceHead
    ...
class DsparkWrapper(DflashWrapper):
    # 采样所有 block_size 个 slot（sample_from_anchor=True），与 DFlash 不同
```

```215:263:tensor_cast/layers/dspark.py
def _sample_sequential(self, draft_hidden, base_logits, batch_size, block, next_tokens):
    """Block-internal serial Markov / Confidence / ArgMax for compute modeling."""
    prev = next_tokens[:, -1]
    for i in range(block):
        logits_i = base_logits[:, i]
        markov_embed, bias, markov_state = markov_head(prev, head_i, markov_state)
        logits_i = logits_i + bias                      # Markov 偏置注入
        conf_i = confidence_head(conf_features)         # 置信度（仅计 compute）
        draft_i = torch.argmax(logits_i, dim=-1)
        draft_steps.append(draft_i)
        prev = draft_i                                  # 块内串行：prev 依赖上一步
    ...
```

关键差异：DSpark 的块内 token 是**串行相关**的——`prev = draft_i` 把上一步 argmax 结果喂给下一步的 Markov head。块内 token 之间不是独立采样，而是一条 Markov 链。DeepSeek-V4.1 的原生 DSpark（`deepseek_v41_dspark.py`）也是这样：整条提案块由 `anchor → markov_head → argmax` 串行生成。

### 3.2 吞吐折叠（块级接受）

DSpark 的折叠与 DFlash 完全统一：

```620:628:serving_cast/service/base_throughput_optimizer.py
# DSpark: unified clamp to n (block-1)
dspark_block_size = optimizer_data.dspark_block_size or 0
if dspark_block_size >= 2:
    accept = optimizer_data.dspark_acceptance_length
    if accept is None:
        accept = 5.0
    n = dspark_block_size - 1
    accept = min(max(float(accept), 0.0), float(n))
    return latency_ms / (accept + 1.0)
```

即 DSpark 用一个**标量** `dspark_acceptance_length ∈ [0, n]` 作为 `E[accepted]` 的代理，折叠为 `latency/(accept+1)`。

### 3.3 数学：块级（block-level）存活

DSpark 的接受是**整块联合**事件：要么整块被 verifier 接受（块存活），要么整块回退到 anchor + 1 个 bonus。其块存活概率记为 `p_block`，则

\[
E[\text{accepted}]_{\text{block}} = n \cdot p_{\text{block}} \quad (\text{块存活则拿满 } n \text{ 个})
\]

若 verifier 支持块内部分接受（如 tree / 部分验证），接受数 A ∈ {0,1,…,n} 是一个**联合分布**，更一般地

\[
E[\text{accepted}]_{\text{block}} = \sum_{k=0}^{n} k\,P(A=k)
\]

无论哪种，DSpark 的随机性来源是"整块的联合正确性"，而**不是**逐位条件概率的乘积链。

---

## 4. 两种粒度为什么不能混

| 维度 | MTP（位置条件） | DSpark（块级） |
|---|---|---|
| 提案结构 | 逐层独立预测下一位置 | 块内由 Markov head 串行生成（相关） |
| 校验粒度 | 逐 token 校验，第 i 位条件于前缀 | 整块联合校验（块存活） |
| 随机性来源 | 逐位条件不确定性 α_i，位置衰减 | 块内相关链 + 整块联合不确定性 |
| 期望形式 | 乘积链 `Σ Π α_i`（精确）/ `Σα_i`（code 近似） | `n·p_block` 或 `Σ k·P(A=k)` |

**问题本质**：若直接拿 MTP 的 α_i 去拟合一个"等效块存活概率"，等于把"逐位条件不确定性"误当作"块联合不确定性"：

1. **忽略了块内相关结构**。DSpark 的块内 token 由同一条 Markov 链串行生成（`prev → draft_i`），它们高度相关；而 MTP 的 α_i 是 verifier 逐位独立校验的条件率。把 α_i 当成分量去"拼"块存活，等于假设块内 token 像 MTP 那样逐位独立可校验，低估了块内相关带来的"一错全错"风险。

2. **把位置衰减结构强加给本没有位置索引的块模型**。MTP 的 α_i 快速衰减来自"越靠后越难在正确前缀上继续预测"。DSpark 的块存活是单一联合事件，没有"第 i 个位置"的概念；用一组 α_i 拟合等价存活，会把"位置衰减"这一 MTP 专属结构错误地映射进 DSpark 的块产量随块大小变化的曲线里。

3. **块产量对块大小是非线性饱和的，且与 MTP 的衰减形态不同**。DSpark 增大 block_size 时，块存活 `p_block` 并非按 MTP 的 `Π α_i` 方式衰减——它的衰减来自"相关链越长、整块出错概率越高"，形态是另一类饱和/衰减曲线。用 MTP 的 α_i 拟合会把错误的非线性形态带进来。

结论：**两种粒度（token-level conditional vs block-level joint）必须由各自的校准源独立建模，不能相互借用 α_i 拟合"等效块存活"。**

---

## 5. 较优的 DSpark 接受期望建模思路（按成熟度排序）

### A. 解耦校准（最小改动，先止血）
把 `dspark_acceptance_length` 当作**独立标定量**，直接从 DSpark 实测的"块接受率"标定（整块验证通过的比例），**不借 MTP 的 α_i**。接口层其实已经分离（`dspark_acceptance_length` vs `mtp_acceptance_rate`），只差"拟合程序不能去读 MTP 的 α_i"。折依然 `latency/(accept+1)`，但 `accept = n · p_block`（满块产量），而非 MTP 拟合出的标量。

### B. 经验块接受分布 P(A=k)（最贴近真相）
直接从真实 DSpark 回放/采样收集接受数 A 的经验分布（A ∈ {0,…,n}），则 `E[accepted] = Σ k·P(A=k)` 精确成立，彻底绕开条件/联合的混淆。TensorCast 的 `text_generate` 已经能产出 draft token（`_sample_sequential`），扩展一个"与 target replay 对拍、统计接受数"的探针即可产出 P(A=k)。折叠仍是 `1 + E[A]`。

### C. 用 ConfidenceHead 驱动块存活（in-repo 已有原材料）
`ConfidenceHead` 在每个位置已产出标量置信度 `conf_i`（见 `dspark.py` 的 `_sample_sequential` 返回 `confidence`），但代码中明确标注：

```125:133:tensor_cast/layers/dspark.py
class ConfidenceHead(torch.nn.Module):
    """Confidence projection; counted for compute only (does not drive acceptance)."""
```

即置信度目前**只计入计算量、不参与接受决策**。最自然的扩展：把块存活建模为块内置信度的联合事件，例如

\[
p_{\text{block}} = P\Bigl(\bigwedge_{i} (c_i \ge \tau_i)\Bigr)
\]

用 DSpark **自身**的置信信号（而非 MTP 的 α_i）来标定 `τ_i` 与 `p_block`。这把"块联合不确定性"建立在 DSpark 原生信号上。

### D. Markov 相关块接受模型（principled）
既然块内 token 是 Markov 链（`prev → draft_i`，由 MarkovHead 偏置），可显式建模"正确/错误"态上的转移链：

\[
p_{\text{block}} = P(\text{链上 } n \text{ 步都停留在"正确"态} \mid \text{起始正确})
\]

转移强度由 MarkovHead / ConfidenceHead 质量决定，**与 MTP 的逐位条件接受 α_i 完全分离**。这一步是把"draft 内部相关"和"verifier 逐位条件接受"在数学上切开，正是化解粒度混淆的核心。

### E. 块产量曲线 E[accepted | n]（工程友好）
把 DSpark 的接受期望建模成**块大小 n 的 1-D 曲线**（而非 α_i 向量）：

\[
y(n) = E[\text{accepted} \mid \text{block size } n]
\]

可用饱和曲线（如 `y(n) = n·p·r^{\,n-1}` 或更平滑的 saturating form）拟合。它直接刻画"块越大、块内相关累积导致整块出错概率上升"的非线性，且天然区别于 MTP 的位置衰减链。

### F. 粒度感知统一框架（推荐落地方向）
最干净的统一解法：让每个投机方法声明

- (a) **接受粒度**：token-level / block-level；
- (b) **接受数分布 A**（或其期望来源）：MTP 用条件 α_i 的乘积链；DSpark 用块联合存活或 P(A=k)；
- (c) **校准源**：MTP 的 α_i 只服务于 token 方法，DSpark 的 `p_block`/P(A=k) 只来自 DSpark 自身的 Confidence/Markov/实测。

折叠函数统一为 `latency / (1 + E[A])`，但 `E[A]` 由声明的方法自身计算。这样从框架层面杜绝"跨粒度借用 α_i"。

---

## 6. 内部是否已沉淀相关算法框架？

**结论：接口已分离、原材料已就位，但 DSpark 专属的"接受期望算法框架"尚未接线。**

已具备的资产：
1. **接口分离**：`dspark_acceptance_length` 与 `mtp_acceptance_rate` 是不同字段；折叠统一收口在 `base_throughput_optimizer._fold_decode_latency_ms`（`612:652`）。
2. **MTP 的链式 α_i 公式**已存在（legacy `sum(rates)+1` / new `accept+1`），可作为 token-level 方法的参考模板。
3. **DSpark 自身的信号源**已存在但未用于接受建模：
   - `MarkovHead`（vanilla/gated/rnn）—— 块内串行相关的显式建模；
   - `ConfidenceHead` —— 每个位置置信度，但代码明确写死 `counted for compute only (does not drive acceptance)`（`dspark.py:126`）。
4. **原生 DSpark**（`deepseek_v41_dspark.py`）已实现 `anchor → markov_head → argmax` 的串行提案与 FP8 sliding-window 草稿 KV，具备在仿真中"对拍统计接受数"的计算基础。

缺失的环节：
- `dspark_acceptance_length` 目前是**用户提供的标量**（默认 5.0，clamp 到 `block-1`），没有内置任何从 DSpark 自身信号推导 `p_block` / P(A=k) 的算法。
- 没有"粒度感知接受模型"框架把 MTP 与 DSpark 的 `E[A]` 计算解耦到各自方法里。

**建议**：按第 5 节 F 落地——在 `_fold_decode_latency_ms` 之外增加一个 `acceptance_expectation(optimizer_data)` 分发点，MTP 走乘积链 / Σα_i，DSpark 走块联合存活（B/C/D/E 任一），并让 `ConfidenceHead` 从"compute only"升级为"可选驱动块存活标定的信号源"。

---

## 7. 关键文件索引

| 文件 | 作用 |
|---|---|
| `serving_cast/service/base_throughput_optimizer.py` (`612:652`) | 投机 decode 折叠统一入口 `latency/(accept+1)` |
| `serving_cast/service/utils.py` (`637:681`) | 标签生成与默认 accept=5.0 |
| `tensor_cast/layers/mtp.py` (`352:409`) | MTP 逐层串行前向计算建模 |
| `tensor_cast/layers/dspark.py` (`136:263`) | DSpark 块内串行 Markov/Confidence 采样 |
| `tensor_cast/layers/deepseek_v41_dspark.py` (`236:325`) | 原生 DSpark 串行提案（V4.1） |
| `tensor_cast/model_config.py` (`511:560`) | `DsparkConfig.dspark_acceptance_length` 默认值与 clamp |
| `tools/perf_data_collection/grid_generator/optimizer_spec.py` (`78`) | `DEFAULT_MTP_ACCEPTANCE_RATES=(0.9,0.6,0.4,0.2)` |
| `cli/utils.py` (`492:535`) | `resolve_draft_block_and_acceptance` 解析与 clamp |
| `cli/inference/throughput_optimizer.py` / `text_generate.py` | 命令行入口与 `--acceptance-length` |
