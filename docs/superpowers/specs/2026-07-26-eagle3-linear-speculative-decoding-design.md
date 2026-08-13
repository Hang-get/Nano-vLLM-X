# Qwen3 EAGLE3 线性推测解码设计

> **Implementation status (2026-08-13): Superseded as a complete runtime description.**
>
> This document records the original linear EAGLE3 design. The linear path is
> implemented, but current EAGLE3 also supports tree decoding. Treat the code
> and `docs/Speculative-Decoding.md` as the operational contract.
>
> - `tree_top_k=1` selects this document's linear rejection-sampling path.
> - `tree_top_k>=2` with `tree_max_depth>=1` selects tree proposal, target
>   tree attention, rank verification, and accepted-path KV commits.
> - `num_speculative_tokens` is used by linear EAGLE3 only; tree depth includes
>   the pending root and can be reduced by available KV capacity.
> - EAGLE3 remains single-GPU and eager-only, disables prefix caching, and now
>   validates the Qwen3-14B Thoughtworks pair at startup. The supported pair is
>   `Qwen/Qwen3-14B` plus `thoughtworks/Qwen3-14B-Eagle3`.
> - The current target context limit is 40960 tokens; 4096 is the initial A800
>   80G benchmark setting. The historical 4B model details below are retained
>   as the original design record, not as the current runtime contract.

日期：2026-07-26

状态：已实现；保留为线性 EAGLE3 的历史设计记录

## 1. 目标

在 nano-vLLM-MS2 中增加固定长度、线性 EAGLE3 speculative decoding，首个且唯一的兼容模型组合为：

- Target：`Qwen/Qwen3-4B-Instruct-2507`
- Draft：`andyjjrt/Qwen3-4B-Instruct-2507-Eagle3`
- Draft checkpoint revision：`408d111ec6cde42f2784f50cd14189d626a6eae4`

首个里程碑优先保证权重兼容、概率分布正确、target/draft KV 状态一致和批处理正确。吞吐提升必须测量和报告，但不是首版正确性通过条件。

## 2. 已验证的 checkpoint 事实

Draft checkpoint 不是 4B target 权重，而是独立的 EAGLE3 draft model。公开元数据和 `config.json` 声明：

- `architectures = ["LlamaForCausalLMEagle3"]`
- `num_hidden_layers = 1`
- `hidden_size = 2560`
- `intermediate_size = 12288`
- `num_attention_heads = 32`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `draft_vocab_size = 32000`
- `vocab_size = 151936`
- `max_position_embeddings = 2048`
- `torch_dtype = bfloat16`

模型卡说明该模型使用 SpecForge 训练，在 vLLM 上做过运行测试，但尚未给出 speculative decoding 效果评估。因此设计不得引用未经本项目复现的接受率或加速比。

vLLM 的公开 Qwen3 EAGLE3 适配实现确认以下权重与计算语义：

- checkpoint 的 `midlayer.*` 映射到 draft model 的 `layers.0.*`。
- Q/K/V 和 gate/up 权重分别加载到项目已有的 packed linear 参数。
- `d2t` 权重提供 draft token ID 到 target token ID 的偏移映射；`t2d` 不参与推理加载。
- 默认使用三路 target auxiliary hidden states，拼接后通过 `fc` 投影回 2560 hidden size。
- Draft 第一层对 token embedding 和融合后的 target hidden state 分别归一化，再拼接后进入 attention。
- Draft logits 只覆盖 32000 个候选；通过 `d2t` scatter 到 151936 target vocabulary，未覆盖 token 的 draft logit 为负无穷。

Target checkpoint 有 36 个 decoder layers。vLLM 的 EAGLE3 默认规则是 `(2, num_layers // 2, num_layers - 3)`，因此本模型组合固定捕获 `(2, 18, 33)`。该常量由 checkpoint adapter 提供而不是暴露成首版用户参数；集成前必须使用参考实现输出做逐层特征和 draft logits 对齐测试。

## 3. 范围

### 3.1 首版包含

- 单 GPU。
- `enforce_eager=True`。
- 固定最大 draft 长度 `K`，batch 内请求允许不同有效长度。
- Qwen3 target 的指定层 hidden-state capture。
- EAGLE3 draft prefill、逐 token 线性 proposal 和独立 draft KV cache。
- 完整 draft probability 传递和 rejection sampling。
- Target/draft KV 的预留、提交、拒绝回滚和请求结束清理。
- EOS、`max_tokens`、上下文上限、KV block 边界和 KV 不足 fallback。
- 正确性指标、接受率和分阶段耗时。

### 3.2 首版不包含

- EAGLE2/EAGLE3 动态候选树。
- Tree attention。
- Tensor parallel。
- CUDA graph。
- Prefix cache 与 EAGLE3 的组合。
- 训练、微调或 checkpoint 转换。
- 其他 target/draft checkpoint 的兼容承诺。

## 4. 方案选择

采用“固定长度线性 EAGLE3 + 最小 proposer 抽象”。

不把 EAGLE3 直接写进现有 N-gram proposer，因为后者是无模型、无 KV 的 CPU token lookup，生命周期完全不同。也不在首版实现通用 tree proposer，因为当前 attention、scheduler 和 block manager 都没有树形位置或分支 KV 语义。

首版只提取能真实降低耦合的公共边界：proposal 数据结构、验收结果数据结构和 target/draft 生命周期接口。N-gram 行为必须保持不变。

## 5. 组件边界

```text
LLMEngine
  `- ModelRunner：编排持久 anchor state、draft 和 verify
       |- Target Qwen3：普通推理 + 按需捕获 EAGLE3 auxiliary features
       |- Eagle3Drafter：draft model、d2t、draft KV、线性 proposal
       `- RejectionSampler：使用 target_probs 和 draft_probs 验收
```

### 5.1 `ModelRunner`

负责：

- 加载和 warmup target/draft 两个模型。
- 建立联合显存预算。
- 编排 per-request anchor state、draft proposal、target verification。
- 将 draft probabilities 交给 rejection sampler。

不负责：

- EAGLE3 网络内部权重映射。
- `d2t` scatter 细节。
- 通过 token 相等性反推 accepted count。

### 5.2 `Eagle3Drafter`

负责：

- Draft checkpoint 校验和加载。
- 三路 target features 的融合。
- Token embedding 与融合特征的 shift/alignment。
- Draft forward 同时返回 post-norm hidden 用于 logits 和 pre-norm auxiliary hidden 用于下一 recurrent step。
- 独立 draft KV cache。
- 固定长度线性 proposal。
- 32000 draft vocabulary 到 151936 target vocabulary 的映射。
- 根据 target verification 结果提交或回滚 draft KV。

### 5.3 Target Qwen3

普通 `forward()` 的返回值和性能路径保持不变。只有 EAGLE3 路径显式请求时，模型才收集指定 decoder layers 的输出。捕获逻辑不得使用长期 module hooks；由 forward 参数控制，并返回明确的结构化结果。

### 5.4 Scheduler 和 BlockManager

Scheduler 继续拥有请求状态和逻辑 block table。Target/draft cache 使用相同逻辑 block ID、不同物理 tensor，使预留、提交、抢占和释放可以按同一生命周期执行。

`Scheduler.schedule()` 必须把本轮抢占的 `seq_id` 暴露给 `LLMEngine.step()`；后者在执行新 batch 前调用 `ModelRunner.release_eagle_states(preempted_seq_ids)`。Speculative postprocess 同样返回 finished IDs，并在本轮结束时释放对应 request anchor。物理 KV tensor 不清零，但 block table 和 request state 同时失效。

## 6. 数据契约

```python
@dataclass
class DraftProposal:
    token_ids: list[list[int]]
    probabilities: torch.Tensor
    lengths: list[int]


@dataclass
class SpecDecodeResult:
    output_token_ids: list[list[int]]
    accepted_draft_counts: list[int]


@dataclass
class Eagle3RequestState:
    anchor_hidden_states: torch.Tensor
    draft_num_computed_tokens: int
    valid: bool
```

`DraftProposal.probabilities` 的 shape 为 `[sum(lengths), target_vocab_size]`，行顺序与 ragged `token_ids` 展平顺序一致。首版选择完整 target-vocab probability，是为了直接满足精确 rejection sampling 的 recovered distribution `max(p - q, 0)`。稀疏或融合表示只作为后续性能优化。

每个 draft 行先在 32000 draft vocabulary 上应用该请求的 temperature 并计算 softmax，再依据 `d2t` scatter 到 target vocabulary；未覆盖 target token 的概率严格为零。Target verification 使用相同 temperature 计算 `p`，避免 proposal 与验收处于不同分布。

`SpecDecodeResult.accepted_draft_counts` 是验收算法的显式输出。Scheduler 不再通过比较 draft/output token 来猜测接受长度。

`Eagle3RequestState.anchor_hidden_states` 保存“当前 pending token 的前驱位置”对应的三路 target auxiliary hidden states，单请求 shape 为 `[3 * target_hidden_size]`。当前 pending token 本身保存在 `Sequence.last_token`，尚未写入 target KV。状态由 `ModelRunner` 按 `seq_id` 管理；请求结束和抢占时必须显式释放，重新 prefill 时重新建立。

位置对齐固定为：target 位置 `t` 的 token embedding 与 target 位置 `t-1` 的 auxiliary hidden states 配对，并写入 draft 位置 `t-1`。因此所有已提交边界必须满足 `draft_num_computed_tokens = max(target_num_computed_tokens - 1, 0)`。Prompt prefill 使用 `input_ids[1:]` 配对 target auxiliary rows `[:-1]`；decode 的 pending root 使用其前驱 anchor，并写入下一空闲 draft slot。

## 7. 端到端数据流

### 7.1 Prefill

```text
prompt tokens
  -> target Qwen3 prefill
     -> 写 target KV
     -> 捕获三路 auxiliary hidden states
     -> 产生最后 prompt 位置的 target logits
  -> EAGLE3 draft prefill
     -> 按 checkpoint 的 token/feature shift 构建 draft KV
  -> 保存最后 prompt 位置的 auxiliary hidden states 作为 request anchor
  -> 普通 sampler 从 target logits 产生第一个 completion token
```

生成出的 completion token 是下一轮的 pending root，尚未写入 target KV。它的前驱位置就是最后一个 prompt token，因此已保存的 request anchor 可以直接驱动 draft model，不需要额外执行一次 target forward。

### 7.2 Decode round

```text
1. Load request anchor
   读取 pending root token
   读取其前驱位置的三路 target auxiliary hidden states

2. Linear draft
   将 pending root embedding 与融合后的 anchor hidden state 输入 Eagle3
   用 Eagle3 draft model 依次产生 K 个候选
   再执行一次不采样的 cache-fill forward，将第 K 个候选写入 draft KV
   写临时 draft KV
   保存每个候选的 q 分布

3. Target verify
   一次 target varlen prefill 处理 pending root + K 个候选
   写临时 target KV
   pending root 行的 logits 对应第一个 draft 的 p 分布
   后续 draft 行的 logits 对应后续 draft 及 bonus 分布
   同时捕获 root 和各 draft 位置的三路 auxiliary hidden states

4. Accept/reject
   返回 accepted_draft_counts
   返回 accepted prefix + recovered token
   或返回全部 drafts + bonus token

5. Commit
   target KV 提交 pending root + accepted draft prefix
   draft KV 提交与该前缀对齐的状态
   释放未使用的临时 blocks
   recovered/bonus 追加到 Sequence，成为下一轮 pending root
   选择 accepted_draft_counts 对应的 target auxiliary row 作为新 anchor
```

若接受 `a` 个 draft，则新 pending root 的前驱位置是 target verification 输入中的第 `a` 行：`a=0` 时是旧 pending root，`a>0` 时是最后一个已接受 draft。该行的三路 auxiliary hidden states 构成下一轮 request anchor。全接受时同样选择最后一个 draft 行，bonus token 成为 pending root。

Rejection sampling 只作用于 K 个 Eagle3 draft tokens；pending root 已由上一轮 target 分布产生，不计入 proposed/accepted draft 指标。

Draft forward 次数是 `K + 1` 而不是 `K`：第 0 次处理 pending root 并产生第一个候选，第 `K - 1` 次产生第 K 个候选，第 K 次只处理第 K 个候选并填充 KV，不保留其 logits。这样接受 `a` 个候选后，可以提交 pending root 加接受前缀对应的 `a + 1` 个 draft slots；全接受时最后一个候选的 KV 也已存在。有效 draft 长度为零的 fallback 仍执行一次 pending-root cache-fill forward，但 proposed/accepted draft 指标保持为零。

实现 token/feature shift 前，必须建立长度为 4 到 8 的人工序列，用 vLLM 参考适配器和本项目适配器分别输出融合输入、draft hidden state 和 logits，逐位置比较。该测试定义位置语义，不能以“生成文本看起来正常”代替。

### 7.3 Ragged batch

每个请求的有效 draft 长度为以下约束的最小值：

- 配置的 `num_speculative_tokens`。
- 请求剩余 `max_tokens`。
- Target 与 draft 的上下文剩余长度。
- 当前可预留 KV 容量。

所有概率 tensor 使用 `lengths` 定义展平布局，不依赖 padding token 参与概率运算。

## 8. KV cache 和显存

Target 与 draft 共用逻辑 block ID：

```text
block_id = n
  target_kv_cache[:, :, n, ...]
  draft_kv_cache[:, :, n, ...]
```

物理 tensor 独立，避免 head 数、层数和 dtype 被错误共享。

联合 block 成本为：

```text
per_block_bytes = target_kv_block_bytes + draft_kv_block_bytes
```

初始化顺序固定为：

1. 加载 target/draft 权重。
2. Warmup target feature capture、draft step 和 target verification 的最大预期临时张量。
3. 读取稳定后的显存峰值。
4. 根据联合每 block 字节数分配相同数量的 target/draft blocks。

拒绝后不清零 KV 内容，只收缩 committed length 并释放多余 block。不可达位置的旧值随后被覆盖。任何 kernel 都必须依据 committed length 和 slot mapping 访问，不能把“物理 tensor 中存在值”当作有效状态。

## 9. 配置

用户接口：

```python
LLM(
    target_model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
    speculative_config={
        "method": "eagle3",
        "draft_model": draft_model_path,
        "num_speculative_tokens": 5,
    },
)
```

Target 和 draft 在当前项目中都必须是本地目录。Hugging Face model ID 用于记录来源，不改变现有本地加载约束。

EAGLE3 初始化时强制：

- `tensor_parallel_size == 1`。
- `enforce_eager is True`。
- Prefix cache 被禁用。
- `num_speculative_tokens >= 1`。
- `max_model_len <= min(target_limit, draft_limit)`；当前 draft limit 为 2048。

## 10. 权重加载规则

新增专用 EAGLE3 loader，不扩大普通 target loader 的模糊匹配行为。

加载规则至少包括：

- `midlayer.` 到 `layers.0.`。
- `q_proj/k_proj/v_proj` 到 packed `qkv_proj`。
- `gate_proj/up_proj` 到 packed `gate_up_proj`。
- `d2t` 到不可训练的 `draft_id_to_target_id`。
- `t2d` 明确跳过并记录原因。
- 不属于 `lm_head` 的 draft 权重按实现结构添加 `model.` 前缀。

该 checkpoint 不携带完整 target embedding；draft 的 `embed_tokens.weight` 必须与已加载 target `model.embed_tokens.weight` 共享参数，并验证 vocabulary、hidden size、dtype 和设备一致。加载结束必须报告并验证 consumed、skipped、injected、missing 和 unexpected 权重集合。除显式跳过的 `t2d` 和由 target 明确注入的 `embed_tokens.weight` 外，任何 missing/unexpected tensor 都使初始化失败。

## 11. 错误和 fallback

初始化阶段验证：

- 模型目录和 `config.json` 存在。
- Target 与 draft architecture 匹配本设计。
- Hidden size、attention heads、KV heads、dtype 和 tensor shape 一致。
- Auxiliary layer IDs 合法且数量为 3。
- `d2t` 长度为 32000，映射结果全部落在 target vocabulary 范围内，且目标 ID 不重复。
- BOS、EOS 和 tokenizer vocabulary 与 target 一致。
- Target/draft 最大位置长度满足配置。

错误消息包含字段名、checkpoint 路径、期望值和实际值。

运行时只允许一种 fallback：当某请求本轮无法预留任何 draft token 时，将其有效 draft 长度设为零，仍通过 EAGLE3 verification 路径处理 pending root、捕获新 anchor、采样 bonus，并增加 `fallback_decode_count`。这样不会因调用现有普通 decode 路径而丢失下一轮所需的 auxiliary hidden states。Checkpoint、shape、映射、概率或 KV 状态错误不得静默回退到 N-gram 或普通 decode。

## 12. 测试设计

### 12.1 配置和 loader

- 正确配置成功构造。
- 不支持的 target/draft architecture 失败。
- 缺失、冗余和 shape 不匹配权重失败。
- `d2t` 越界、重复或长度错误失败。
- 所有必要 checkpoint tensors 被消费。

### 12.2 Target feature capture

- 开启 capture 前后最终 logits 数值一致。
- 三路 layer 输出的 shape、dtype、token 顺序正确。
- Prefill、verification 和 batch 分别覆盖。

### 12.3 Draft model

- 单步 KV forward 与无 cache 全序列 forward 一致。
- Batch proposal 与逐请求 proposal 一致。
- 人工短序列上的 token/feature shift 与 vLLM 参考输出一致。
- `midlayer`、packed projections、`fc` 和 `d2t` 加载结果与 checkpoint tensor 一致。
- Scatter 后未覆盖 target tokens 的 logits 为负无穷。

### 12.4 Rejection sampler

增加纯 PyTorch/CPU 参考实现，并允许注入固定 uniform random numbers。

覆盖：

- 全接受。
- 第一位置拒绝。
- 中间位置拒绝。
- 零 draft token。
- Ragged batch。
- 不同 temperature。
- 小词表上最终输出分布与 target 分布一致。
- CPU 参考与 Triton kernel 在相同输入和随机数下逐 token 一致。

### 12.5 KV 状态机

- Accepted count 为 `0..K` 的全部情况。
- 跨 KV block 边界。
- EOS 和 `max_tokens` 在 draft 中间触发。
- KV 不足导致有效 K 缩短或零 draft verification fallback。
- 请求结束、抢占和重新 prefill 后无旧状态污染。
- 连续多轮 verification logits 与普通 target 自回归 logits 在容差内一致。

## 13. 指标

Scheduler/engine 暴露：

```text
draft_tokens_proposed
draft_tokens_accepted
mean_effective_draft_length
acceptance_rate
fallback_decode_count
draft_time_ms
verify_time_ms
sampling_time_ms
```

接受率仍定义为 accepted draft tokens / proposed draft tokens，不包含 bonus token。

## 14. 首个里程碑验收门槛

- Draft checkpoint 在零 unresolved missing/unexpected tensor 下完成加载，并确认 target embedding 注入成功。
- Target feature capture 不改变普通推理 logits。
- Draft 单步、batch 和 KV 路径通过参考对齐。
- CPU 与 Triton rejection sampler 在固定随机输入下一致。
- EAGLE3 verification logits 与逐 token target forward 在数值容差内一致。
- Batch、EOS、上下文上限和 KV block 边界测试通过。
- 连续请求前后 GPU allocated memory 不持续增长。
- Target-only 与 EAGLE3 使用相同 prompt、sampling 参数和随机种子运行，并记录输出、接受率、各阶段耗时和总吞吐。
- 性能没有提升时仍可判定正确性里程碑通过，但必须保留数据并定位 draft、verify 或采样开销。

## 15. 后续阶段

正确性里程碑完成后，按以下顺序扩展：

1. 优化完整 target-vocab draft probability 的内存和 scatter 开销。
2. 支持 prefix cache，并证明 target/draft cache hit 同步。
3. 增加 CUDA graph。
4. 增加 tensor parallel。
5. 设计动态候选树和 tree attention。

每一步都必须保留线性 eager 路径作为正确性参考。

## 16. 关键反思记录

- EAGLE3 不是 N-gram proposer 的简单替换；它要求 target auxiliary features、独立模型和独立 KV 生命周期。
- EAGLE3 proposal 使用“pending root token + 其前驱位置的 target auxiliary hidden states”；不能在 proposal 前额外运行 4B target，也不能丢弃上一轮 verification 产生的 anchor features。
- 当前 target verification 已经处理“最后一个 pending token + drafts”，这个布局应被保留并扩展为同时捕获下一轮 anchor，而不是重写成额外 target decode。
- Draft vocabulary 是 target vocabulary 的子集，若忽略 `d2t` 会产生合法范围内但语义错误的 token。
- 文本看起来正常不能证明位置和概率正确；核心验收证据是 reference logits、固定随机数验收结果和 KV 状态一致性。
- 模型卡没有提供效果评估，不能预设接受率或吞吐收益。
- 首版限制不是临时隐式行为，而是初始化时强制执行的能力边界。

## 17. 参考资料

- Draft checkpoint：<https://huggingface.co/andyjjrt/Qwen3-4B-Instruct-2507-Eagle3>
- Target checkpoint：<https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>
- vLLM Qwen3 EAGLE3 adapter：<https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_eagle3.py>
