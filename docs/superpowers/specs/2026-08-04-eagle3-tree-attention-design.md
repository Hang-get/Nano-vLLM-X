# EAGLE3 TreeAttention 设计

日期：2026-08-04

状态：已确认，待实施计划

## 1. 目标

在现有线性 EAGLE3 推测解码基础上，实现**树形候选生成 + 统一验证**的 TreeAttention 功能。

核心变更：Draft model 每步采样 **top-k** 个候选 token（而非 greedy 1 个），形成 token 树；Target model 在**一次 forward** 中使用自定义 attention mask 验证所有树节点；采用 **排名接受**（rank-based acceptance）在树上递归游走选择最优路径——大模型从自身分布采样，若采样 token 落在草稿模型的 top-L 候选集合中即接受。

通过 `top_k=1` 自然回退到当前线性模式，确保向后兼容。

## 2. 方案选择

采用方案 A：**自定义 Tree Attention Mask + 单次 Target Forward**（vLLM/TRT-LLM 同款思路）。

被排除的方案：
- 方案 B（逐路径验证）：多次 target forward，效率低，前缀重复计算
- 方案 C（虚拟序列隔离）：position 偏移影响 RoPE 质量，KV cache 空洞浪费

## 3. 配置参数

`SpeculativeConfig` 新增字段：

```python
tree_top_k: int = 1       # 每个节点的分支数。1 = 线性模式（回退）
tree_max_depth: int = 0   # 树深度。0 = 使用 num_speculative_tokens 作为线性深度
```

| tree_top_k | tree_max_depth | 行为 |
|------------|----------------|------|
| 1 | 任意 | 线性模式，等价于当前行为 |
| >=2 | >=1 | 树形模式，节点数 = sum_{d=0}^{max_depth-1} top_k^d |

验证约束：
- `tree_top_k >= 1`
- 树形模式下 `tree_max_depth >= 1`
- 树节点总数不超过 `max_model_len - seq_len`

## 4. 树形拓扑数据结构

### 4.1 TreeTopology

```python
@dataclass
class TreeTopology:
    total_nodes: int              # 逻辑节点总数（含 root: 1 + sum(top_k^d) for d=1..max_depth-1）
    draft_nodes: int              # 实际 draft 节点数（不含 root）
    parent: list[int]             # parent[i] = 父节点索引，root 为 -1
    children: list[list[int]]     # children[i] = [子节点索引列表]
    depth: list[int]              # depth[i]，root=0
    rope_positions: list[int]     # RoPE 位置: [prompt_len + depth[i]] (draft节点only)
    bfs_to_node: list[int]        # BFS 顺序 → 树节点 ID (draft 节点 only)

    def build_tree_internal_mask(self) -> torch.Tensor:
        """构建 draft 节点间的 N×N boolean mask，True=ancestor"""
        ...
```

### 4.2 线性化策略

BFS（广度优先）平整化。示例 `top_k=2, depth=3`（7 个节点）：

```
树结构:                    BFS 线性化:
   0(root)                 0: root
  /      \                 1: depth-1 node 0
 1        2                2: depth-1 node 1
/ \      / \               3: depth-2 node 0
3  4    5   6              4: depth-2 node 1
                           5: depth-2 node 2
                           6: depth-2 node 3
```

线性回退 (`top_k=1`)：`parent = [-1, 0, 1, ..., K-1]`，等效于当前行为。

### 4.3 Attention Mask 构建

Tree-internal mask（仅 tree 节点之间）：

```
mask[i][j] = True iff j is ancestor of i (including i itself)

示例 (7节点树):
             0  1  2  3  4  5  6
        0    1  0  0  0  0  0  0
        1    1  1  0  0  0  0  0
        2    1  0  1  0  0  0  0
        3    1  1  0  1  0  0  0
        4    1  1  0  0  1  0  0
        5    1  0  1  0  0  1  0
        6    1  0  1  0  0  0  1
```

完整 mask（tree → prompt + tree）见 Section 7.3。Tree 节点可以 attend 到所有 prompt token（包括 root）和所有 tree ancestor，但不能 attend 到兄弟节点。

## 5. Draft 生成：Top-k 树形 Proposal

### 5.1 算法

BFS 层级生成，每层节点统一 batch forward：

```
current_nodes = [root]  # root = pending_root，已在 cache
for depth in range(1, max_depth):  # depth 1..max_depth-1
    收集当前层所有活跃节点的 (input_ids, fused_hidden, positions)
    一次 batch forward → logits, next_hidden
    每个节点采样 top_k candidates → 创建子节点
    子节点入队列（下一层）
    更新 TreeTopology
```

### 5.2 Batch 策略

同深度节点批量处理：

| Depth | Batch size |
|-------|-----------|
| 1 | top_k |
| 2 | top_k^2 |
| 3 | top_k^3 |

> 注意：root (depth=0) 是 pending_root，已在 target KV cache 中，draft 不需要重新 forward。

### 5.3 Eagle3Proposer.propose() 签名变更

```python
def propose(seqs, reservations, temperatures) -> DraftProposal:
    # 返回值 DraftProposal 新增 tree_topologies 字段
```

### 5.4 DraftProposal 扩展

```python
@dataclass
class DraftProposal:
    token_ids: list[list[int]]          # 树形: BFS 平整化的 draft tokens
    probabilities: torch.Tensor         # draft token 的 prob (BFS 顺序)
                                         # 用途: 仅用于 draft 阶段的 top-k 选择
                                         # 排名验证不需要此字段
    lengths: list[int]                  # draft token 总数
    tree_topologies: list[TreeTopology | None]  # None = 线性模式

@dataclass
class SpecDecodeResult:
    output_token_ids: list[list[int]]
    accepted_draft_counts: list[int]
    accepted_paths: list[list[int] | None]  # 新增: 树上接受路径的节点索引, None=线性模式
```

## 6. Draft KV Cache：Copy-on-Write

### 6.1 核心机制

- 子节点**共享引用**父节点的 KV blocks（引用计数）
- 首次写入共享 block 时触发 **copy-on-write**：分配新 block，复制数据，写入
- 验证完成后**只保留接受路径**的 blocks，释放其他分支

### 6.2 实现范围

由 `Eagle3Proposer` 内部管理，不耦合外层 `BlockManager`。KV slot 分配（每个 tree node 写入 `draft_kv_cache` 的哪个位置）由 `TreeDraftKVManager` 内部决定，不在 `TreeTopology` 中暴露。

```python
class TreeDraftKVManager:
    def fork(self, parent_node: int, child_node: int) -> None:
        """子节点继承父节点的 draft KV block 引用，ref_count+=1"""
    
    def allocate_slot(self, node: int) -> int:
        """为树节点分配新的 draft KV slot，若 block 被共享则 copy-on-write"""
    
    def write(self, node: int, position: int, k: Tensor, v: Tensor) -> None:
        """将 KV 写入 slot，trigger copy-on-write if needed"""
    
    def commit(self, accepted_leaf: int) -> list[int]:
        """返回 root→accepted_leaf 路径上所有独占的 block IDs"""
    
    def release_node(self, node: int) -> None:
        """释放节点及其独占 blocks，ref_count-=1"""
```

### 6.3 Block 预留

`SpecReservation.new_block_ids` 预留量从 `K`（线性）变为 `num_tree_nodes`：

```
树节点总数 = sum_{d=0}^{max_depth-1} top_k^d
```

例：`top_k=2, max_depth=4` → 15 节点 → 预留 15 token 位置（约 1 block）。

## 7. Target 验证：Tree Attention Mask

### 7.1 关键问题

`flash_attn_varlen_func` 不支持自定义 attention mask。`flash_attn_func` 支持 mask 但有 block-alignment 限制（通常要求 seqlen 是 128 的倍数）。

**解法**：Tree attention 部分使用 `torch.nn.functional.scaled_dot_product_attention`（PyTorch 原生 SDPA），它对小尺寸（N≤40）足够高效，会自动 dispatch 到最优后端并支持任意形状的 boolean mask。Prompt KV 部分仍使用 FlashAttention 的 paged attention。

或者简化为：整个 tree verification 的 attention 计算统一使用 `F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)`，对 N≤40 的场景性能完全足够。

### 7.2 每层 Attention 计算流程

```
输入: x_tree = [N, dim]  (tree draft tokens 的 hidden states, 不含 root)

1. Gather prompt KV from paged cache:
   K_prompt = gather(k_cache[layer], block_table)  # [P, kv_heads, dim]
   V_prompt = gather(v_cache[layer], block_table)

2. Compute tree KV from current input (不写入 cache):
   K_tree = W_K @ x_tree  # [N, kv_heads, dim]
   V_tree = W_V @ x_tree  # [N, kv_heads, dim]
   # 注意: 此处 store_kvcache 被跳过，tree tokens 的 KV 不持久化

3. Concat + build mask:
   K_full = cat[K_prompt, K_tree]  # [P+N, kv_heads, dim]
   V_full = cat[V_prompt, V_tree]
   
   full_mask[0:P, 0:P] = causal (lower triangular)    # prompt 内部
   full_mask[P:P+N, 0:P] = True (tree→prompt: all)    # tree 可以看到所有 prompt
   full_mask[P:P+N, P:P+N] = tree_mask (ancestor-only) # tree 内部只有祖先行

4. PyTorch SDPA (自动 dispatch 到最优后端):
   o = F.scaled_dot_product_attention(
       Q_tree,           # [N, q_heads, dim]
       K_full,           # [P+N, kv_heads, dim]
       V_full,           # [P+N, kv_heads, dim]
       attn_mask=full_mask[P:P+N, :],  # boolean mask
       scale=self.scale,
   )
```

> **为什么用 `F.scaled_dot_product_attention` 而不是 `flash_attn_func`**：SDPA 对小尺寸（N≤40）无 block-alignment 限制，自动 dispatch 到最优 kernel（可能是 FlashAttention、Memory-efficient attention 或 math fallback），同时支持任意 boolean mask。

> **关键设计点**: Tree tokens 的 KV 仅在 `K_full`/`V_full` 中存在，不调用 `store_kvcache` 写入主 cache。Commit 后接受路径上的 token 通过正常 decode 流程进入 cache。

### 7.3 Mask 示意

```
prompt tokens (P=5):    [t0, t1, t2, t3, t4]      (cached, 含 pending_root)
tree draft tokens (N=6): [c1, c2, g1, g2, g3, g4]  (新建，pending_root不在这)
                          ^   ^   ^   ^   ^   ^
                         depth=1  depth=2

Tree → full context mask (6 rows × 11 cols):
              t0 t1 t2 t3 t4 | c1 c2 g1 g2 g3 g4
         c1    1  1  1  1  1 |  1  0  0  0  0  0     c1 → prompt + 自己
         c2    1  1  1  1  1 |  0  1  0  0  0  0     c2 → prompt + 自己 (兄弟不可见)
         g1    1  1  1  1  1 |  1  0  1  0  0  0     g1 → prompt + c1 + 自己
         g2    1  1  1  1  1 |  1  0  0  1  0  0     g2 → prompt + c1 + 自己
         g3    1  1  1  1  1 |  0  1  0  0  1  0     g3 → prompt + c2 + 自己
         g4    1  1  1  1  1 |  0  1  0  0  0  1     g4 → prompt + c2 + 自己
```

> pending_root (t4) 在 prompt cache 中，tree draft 节点通过 `full_mask[P:, 0:P] = True` 自然可以 attend 到它。

### 7.4 位置编码

RoPE 位置使用 `prompt_len + logical_depth`（同层不同分支共享相同 position，tree mask 已隔离互不可见，逻辑正确）。

KV cache slot 分配使用独立的 BFS index（每个树节点唯一 slot），与 RoPE position **解耦**：

| 用途 | 值 | 说明 |
|------|-----|------|
| RoPE 位置 | `prompt_len + depth` | 同层分支共享，保证相对位置正确 |
| KV cache slot | `prompt_len + bfs_index` | 每个节点唯一，避免分支间冲突 |

Tree token KV **不写入主 target KV cache**（因为多数会被拒绝）。验证时 tree token 的 K/V 仅在 `K_full`/`V_full` 拼接张量中存在，不持久化。Commit 时只将接受路径上的 token KV 通过正常 decode 流程写入 cache。

### 7.5 Attention 模块修改

`Attention.forward()` 新增 `tree_verify` 分支：

```python
def forward(self, q, k, v):
    context = get_context()
    if context.is_tree_verify:
        # 跳过 store_kvcache (tree tokens 的 KV 不持久化)
        return self._tree_attention(q, k, v)
    elif context.is_prefill:
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
        # ... 不变
    else:
        # decode: store_kvcache 已有，不变
        ...

def _tree_attention(self, q, k, v):
    # 1. gather prompt KV from cache (每层)
    # 2. concat [K_prompt, K_tree], [V_prompt, V_tree]
    # 3. F.scaled_dot_product_attention(Q_tree, K_full, V_full, attn_mask=tree_mask)
    # 注意: K_tree = k, V_tree = v (来自当前层的投影，不存 cache)
```

> 关键区别：`tree_verify` 分支**不调用** `store_kvcache`，tree tokens 的 K/V 仅在 attention 计算中使用。

### 7.6 Prompt KV Gather 说明

prompt KV 的 gather 操作在**每层 attention 调用时执行**（不同层的 KV cache 内容不同）。Gather 是 tensor indexing 操作，不涉及显存分配和拷贝。对于 36 层 × 2048 prompt × 8 kv_heads × 128 dim × bfloat16 ≈ 150MB 的读取量，在 GPU 带宽下（~1TB/s）耗时 < 0.2ms，可以接受。

伪代码：
```python
def gather_prompt_kv(k_cache_layer, block_table, prompt_len, block_size):
    # k_cache_layer: [num_blocks, block_size, kv_heads, dim]
    # Output: [prompt_len, kv_heads, dim]
    blocks = k_cache_layer[block_table]       # gather blocks
    flat = blocks.reshape(-1, kv_heads, dim)   # flatten
    return flat[:prompt_len]                   # trim padding
```

### 7.7 多 Sequence 批处理

树形模式不使用 `cu_seqlens`（需要每 sequence 独立 mask），采用 **padding + block-diagonal mask**：

```
批处理流程:
1. 找到 batch 内最大的 N_max = max(N_i) 和 K_max = max(P_i + N_i)
2. 填充短序列到最大长度 (0 填充，mask 中填 False)
   Q: [B, N_max, q_heads, dim]
   K_full: [B, K_max, kv_heads, dim]
3. 构建 block-diagonal 掩码:
   mask[b, i, j] = 1  iff (i < N_b and j < P_b+N_b and tree_mask_b[i][j])
   否则 0
4. F.scaled_dot_product_attention(Q, K_full, V_full, attn_mask=mask)
```

### 7.8 Context 扩展

```python
# context 新增字段（仅 tree_verify 时有效）
is_tree_verify: bool
tree_attn_mask: torch.Tensor          # [N, P+N] boolean mask
prompt_k_cache: torch.Tensor          # prompt KV cache 引用 (整个 cache tensor)
prompt_v_cache: torch.Tensor
prompt_block_table: torch.Tensor      # 用于 gather: [num_prompt_blocks]
prompt_seq_len: int                   # prompt token 数量
```

## 8. 树形排名验证（Rank-based Acceptance）

采用 EAGLE-2 风格的排名接受机制，**不依赖草稿模型的概率值**进行验证。

### 8.1 算法

从根节点开始，在树上递归游走：

```
accepted = []
node = root

while True:
    # 从大模型在当前节点的真实分布中采样一个 token
    sampled_token = sample(target_logits[node])  # 带 temperature

    # 检查采样出的 token 是否落在草稿树当前节点的子节点集合中
    children_tokens = {child.token for child in node.children}
    if sampled_token in children_tokens:
        # 接受：找到对应的子节点，沿该分支继续
        accepted_child = node.children[find(child.token == sampled_token)]
        accepted.append(accepted_child.token)
        node = accepted_child
    else:
        # 拒绝：采样出的 token 不在任何子节点中
        # 该 token 本身就是 recovery token，验证在此层终止
        accepted.append(sampled_token)
        break

    # 如果到达叶子节点且仍在接受 → bonus token
    if not node.children:
        bonus = sample(target_logits[node])
        accepted.append(bonus)
        break
```

### 8.2 与线性版/原拒接采样版的差异

| 方面 | 线性拒绝采样 | 树形排名验证 |
|------|------------|------------|
| 遍历方式 | 顺序 for 循环 | 根→叶递归 |
| 接受条件 | `target_prob / draft_prob >= rand()` | 大模型采样 token ∈ draft 的 top-L 集合 |
| 是否用 draft 概率 | **是** | **否** |
| 拒绝恢复 | `sample(target - draft)` | 采样出的 token 即是输出 |
| Bonus token | 所有 K 个全接受时 | 到达叶节点且仍被接受时 |

### 8.3 关键性质

- **分布等价性**：只要草稿模型提供的 top-L 集合包含了大模型会采样的 token，接受路径就能继续。最终输出分布与大模型真实分布一致（数学上可证）。
- **对概率值不敏感**：草稿模型只需要输出正确的**相对排名**（top-L 集合），不需要精确概率。训练和使用更加友好。
- **线性回退**：`top_k=1` → 每个节点只有 1 个子节点 → 退化为"大模型每步采样是否恰好等于草稿模型预测"的单链验证。

## 9. Model Runner 集成

### 9.1 prepare_spec_decode 变更

```python
def prepare_spec_decode(seqs, draft_token_ids, reservations, 
                        tree_topologies=None, include_query_lengths=False):
    if tree_topologies is not None:
        # 树形路径: 使用 padding-batched flash_attn_func
        #   1. BFS 平整化 draft tokens
        #   2. 构建每个 request 的 tree attention mask
        #   3. Pad 所有 sequence 到相同长度 (mask 用 False 填充)
        #   4. Q: [B, N_max, dim], K: [B, P_max+N_max, kv_heads, dim]
        #   5. 计算 verify_row_indices（每个 tree node 1 行 logit）
        return (input_ids, positions, verify_rows, tree_masks, query_lengths)
    else:
        # 线性路径：使用 cu_seqlens + flash_attn_varlen_func (现有行为不变)
        # 输入: flat [total_tokens] + cu_seqlens
        ...
```

> 树形和线性模式下 `input_ids` 和 `positions` 的形状不同：
> - 线性: `[total_tokens]`（所有 sequence 拼接）
> - 树形: `[B, N_max]`（padding batched）

### 9.3 混合 Batch 处理

如果 batch 内部分 sequence 使用树形、部分使用线性：
- 线性 sequence 构造**退化的 TreeTopology**（`top_k=1` 的链，causal mask）
- 统一走树形路径，用 padding-based batching
- 避免维护两个并行代码路径

但首次实现建议**不分叉**：要么全部线性、要么全部树形（通过 `tree_topologies is not None` 判断）。

### 9.4 run_eagle3_spec_decode 变更

```python
def run_eagle3_spec_decode(seqs, reservations):
    # 1. Draft proposal
    proposal = proposer.propose(...)

    # 2. Target verification (forward + get logits)
    if proposal.tree_topologies:
        # 树形路径: 使用 padding-batched SDPA (不用 cu_seqlens)
        prepared = prepare_spec_decode(seqs, proposal, reservations,
                                       proposal.tree_topologies, include_query_lengths=True)
        input_ids, positions, verify_rows, tree_masks, query_lens = prepared
        set_context(tree_verify=True, tree_attn_masks=tree_masks, ...)
        target_output = model(input_ids, positions, aux_layer_ids)
        # 注意: tree tokens 的 KV 未写入主 cache
    else:
        # 线性路径：现有行为不变（cu_seqlens + flash_attn_varlen_func）
        ...

    # 3. Rank-based 验证 (替代 rejection sampling)
    # 大模型从 logits 采样，检查是否在 draft 的 top-L 集合中
    # 不需要 draft_probs 参数
    result = rank_verifier(proposal=proposal,
                            target_logits=verification_logits,
                            temperatures=temperatures)

    # 4. Commit
    # 树形: 接受路径上的 tokens 通过后续正常 decode 写入 KV cache
    #       draft KV manager 释放非接受分支的 copy-on-write blocks
    proposer.commit(seqs, verification_aux, result)
```

## 10. 修改文件清单

| 文件 | 改动 | 复杂度 |
|------|------|--------|
| `config.py` | 新增 `tree_top_k`, `tree_max_depth` | 低 |
| `layers/attention.py` | 新增 `_tree_attention()` 分支 | 中 |
| `utils/context.py` | 新增 tree_verify 相关字段 | 低 |
| `v1/spec_decode/types.py` | 新增 `TreeTopology`，扩展 `DraftProposal` | 低 |
| `v1/spec_decode/eagle3_proposer.py` | `propose()` top-k 生成 + copy-on-write KV 管理 | **高** |
| `engine/model_runner.py` | `prepare_spec_decode()`, `run_eagle3_spec_decode()` 树形分支 | 高 |
| `v1/sample/rejection_sampler.py` | 新增 `RankVerifier` 类（排名验证）; 原有 `RejectionSampler` 保留不变 | 中 |
| `engine/scheduler.py` | block 预留量适配树节点数 | 低 |
| `tests/` | 新增 tree attention 相关测试 | - |

## 11. 不变的部分

以下模块**零改动**：
- Target 模型 (`qwen3.py`)
- Draft 模型 (`qwen3_eagle3.py`)
- Embed head, layernorm, sampler, linear 等 layers
- N-gram proposer
- LLM Engine 上层逻辑
- Prefix cache

## 12. 回退兼容性

`top_k=1` → 树退化为链 → TreeTopology 退化为线性结构 → 所有新代码路径等价于旧行为。不配置新参数时行为完全不变。

## 13. 测试计划

1. **树形拓扑正确性**：验证 BFS 线性化、mask 构建、position 分配的数学正确性
2. **Copy-on-Write 正确性**：验证 fork/write/commit 后 KV block 引用计数和内容正确
3. **排名验证正确性**：树形排名验证的分布等价性（top_k=1 时与线性版输出一致，top_k>1 时接受率提升）
4. **集成测试**：Qwen3-4B + EAGLE3 checkpoint，树形 vs 线性接受率对比
5. **边界条件**：空树、单节点树、max_depth 截断、batch 混合（树形+线性）

## 14. 边界条件处理

| 场景 | 处理方式 |
|------|----------|
| `max_depth` 截断（seq 剩余长度不足） | 动态减小 `max_depth`，树在允许的深度提前终止 |
| batch 内混合树形/线性 | 通过 `tree_topologies` 的 per-sequence None/非None 区分。线性序列走原代码路径，树形序列走新路径 |
| 相同 position 不同分支的 RoPE | 已验证可行：tree mask 隔离互不可见，RoPE 的同 position 编码不影响注意力计算结果 |
| 树节点数 > 预留 block 容量 | scheduler 预留阶段检查，不足时 fallback 到线性模式 |
| 树的 leaf 节点在非最大深度被截断 | 正常流程：leaf 的 `children=[]`，排名验证在此处自然终止 |
| Draft KV cache 全满 | 与当前线性版行为一致：拒绝生成更多 draft，`max_depth` 截断 |
