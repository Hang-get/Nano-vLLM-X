# EAGLE3 TreeAttention 设计

> **Implementation status (2026-08-13): Implemented.**
>
> Tree EAGLE3 is active when `tree_top_k>=2` and `tree_max_depth>=1`. The
> implementation builds a BFS `TreeTopology`, reserves independent target and
> draft-COW pools, releases pruned draft blocks before verification, verifies
> root plus draft rows in one target forward, and commits only the accepted
> path. `tree_top_k=1` remains the separate linear `RejectionSampler` path.
> The current checkpoint contract is the Qwen3-14B Thoughtworks pair documented in
> `docs/Speculative-Decoding.md`; any Qwen3-4B references below are historical.
>
> Tree depth includes the pending root, available KV capacity can lower the
> effective depth per request, and target draft rows are staged until rank
> acceptance. Current source of truth: `nanovllm/v1/spec_decode/`,
> `nanovllm/layers/attention.py`, and `nanovllm/engine/model_runner.py`.

日期：2026-08-04

状态：已实现；保留为 TreeAttention 的历史设计记录

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
tree_top_k: int = 1         # 每个节点的最大分支数。1 = 线性模式（回退）
tree_max_depth: int = 0     # 最大树深度。0 = 使用 num_speculative_tokens 作为线性深度
tree_prune_ratio: float = 0.0  # 动态修剪阈值。0 = 不修剪；>0 = 只保留 prob >= ratio*max_prob 的子节点
```

| tree_top_k | tree_max_depth | tree_prune_ratio | 行为 |
|------------|----------------|-------------------|------|
| 1 | 任意 | 任意 | 线性模式，等价于当前行为 |
| >=2 | >=1 | 0 | 固定 k-ary 树，节点数 = sum(top_k^d) |
| >=2 | >=1 | >0 | 动态修剪，每节点保留 prob 高于阈值的子节点 |

`tree_top_k=1` 是严格的 legacy mode：不构造 `TreeTopology`，不创建 staging buffer，并继续使用当前线性 `RejectionSampler`。只有 `tree_top_k>=2` 启用 TreeAttention、`RankVerifier` 与两类 tree block pool。

验证约束：
- `tree_top_k >= 1`
- 树形模式下 `tree_max_depth >= 1`
- `tree_prune_ratio` in `[0, 1]`
- 最大接受路径长度（不含 pending root）不超过 `max_model_len - seq_len`；树节点总数只约束 Draft COW pool 与 Target staging buffer 的容量

## 4. 树形拓扑数据结构

### 4.1 TreeTopology

```python
@dataclass
class TreeTopology:
    total_nodes: int              # 逻辑节点总数（含 root，实际值由动态修剪决定）
    draft_nodes: int              # 实际 draft 节点数（不含 root）
    parent: list[int]             # parent[i] = 父节点索引，root 为 -1
    children: list[list[int]]     # children[i] = [子节点索引列表]
    depth: list[int]              # depth[i]，root=0
    rope_positions: list[int]     # RoPE: root=P-1，draft=P-1+depth[i]
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

链状拓扑可用于单元测试：`parent = [-1, 0, 1, ..., K-1]`。但生产运行时 `tree_top_k=1` 直接走 legacy linear path，不构造 `TreeTopology`。

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

BFS 层级生成，每层节点统一 batch forward，每节点动态修剪低概率分支：

```
current_nodes = [root]  # root = pending_root；Target anchor 已就绪
for depth in range(1, max_depth):  # depth 1..max_depth-1
    if not current_nodes:
        break  # 所有分支被修剪，提前终止
    收集当前层所有活跃节点的 (input_ids, fused_hidden, positions)
    一次 batch forward → logits, next_hidden
    next_nodes = []  # 下一层节点
    for each node in current_nodes:
        probs = softmax(logits[node])  # 该节点的概率分布
        topk_values, topk_indices = topk(probs, tree_top_k)  # 取 top-k
        # 动态修剪: 只保留概率高于阈值的子节点
        threshold = tree_prune_ratio * topk_values[0]  # ratio * max_prob
        keep = topk_values >= threshold
        for i in where(keep):
            create child_node(token=topk_indices[i], prob=topk_values[i])
            next_nodes.append(child_node)
    更新 TreeTopology
    current_nodes = next_nodes
```

### 5.2 Batch 策略

同深度节点批量处理，由于动态修剪，batch size 可能小于理论最大值：

| Depth | Batch size (最大) |
|-------|-----------|
| 1 | ≤ top_k |
| 2 | ≤ top_k^2 |
| 3 | ≤ top_k^3 |

> 注意：root (depth=0) 是 pending_root。Draft 的首次展开会写入 root 的 Draft KV；Target root KV 则在 tree verification 的 root query row 中写入主 cache。

> 动态修剪的效果：每个节点的子节点数在 [1, top_k] 之间动态变化。当概率分布高度集中时（大多数情况），修剪使树偏向链状，节省计算；当分布更平坦时，保留更多分支以提升命中率。

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

由于动态修剪，树的实际节点数在生成完成后才知道，但 scheduler 必须在生成前分配 blocks。

**策略：按最大节点数预留 blocks，生成后释放未使用的**

```
最大 draft 节点数（不含 root）= sum_{d=1}^{max_depth-1} top_k^d
```

root 不是分支 Draft node；其常规 continuation slot 由 scheduler 单独保证，不计入 TreeDraftKVManager 的新增 COW pool。

**COW 对 block 数量的影响**：

Copy-on-Write 下，不能简单按 `ceil(node_count / block_size)` 估算 block 数量。多个分支共享同一个父节点 block 时，每个子分支的首次写入都会触发 COW，产生新 block。

最坏情况：每个 draft 节点的首次写入都触发 COW → 需要的 block 数 = draft 节点数。

```
示例: top_k=3, depth=4（4层含root）
  root:                常规 continuation slot（不计入 COW pool）
  depth=1: top_k=3,    最坏 3 个 COW blocks
  depth=2: top_k^2=9,  最坏 9 个 COW blocks
  depth=3: top_k^3=27, 最坏 27 个 COW blocks
  -----------------
  总计: 最多 39 个 draft blocks 需要预留
```

Tree mode 需要两个彼此独立的 block pool：

- **Draft COW pool**：按最大 draft 节点数预留，供 `TreeDraftKVManager` fork/copy-on-write 使用。
- **Target accepted-path pool**：按最大可接受路径长度预留，供 rank verifier 选中的 draft 节点在提交时写入主 Target KV cache。它不按树节点总数预留。

不能继续让单一 `new_block_ids` 同时承担两种含义；否则 scheduler 会把 Draft block 错误提交到 Target block table。

```python
# 树形模式下的 SpecReservation
SpecReservation(
    max_path_draft_len=...,              # 最大可接受 draft 路径长度
    effective_tree_max_depth=...,        # 发生容量降级后的实际深度
    draft_block_ids=[...],               # Draft COW pool
    target_block_ids=[...],              # Target accepted-path pool
)

# draft_block_ids 不按 position // block_size 解释，而是 COW 存储池。
# target_block_ids 按接受路径的连续逻辑位置解释，供提交后的主 Target KV cache 使用。
```

**降级必须发生在 Draft 生成前**。如果预留 block 不足，降级需**同步更新**以下所有组件：

```
depth 4 → 3 降级需要同步：
  ✓ reservation.max_path_draft_len / effective_tree_max_depth → 更新为新深度
  ✓ reservation.draft_block_ids / target_block_ids             → 重新按两个 pool 预留
  ✓ proposer 的 max_depth  → 传入更新后的值
  ✓ TreeTopology            → 构建时使用新深度
  ✓ proposal.lengths        → 生成后与新深度一致
  ✓ scheduler commit/release → 按更新后的 reservation 操作
```

不一致示例（必须避免）：reservation 只有 13 个 block，但 topology 仍返回 39 个节点的状态。

## 7. Target 验证：Tree Attention Mask

### 7.1 关键问题

`flash_attn_varlen_func` 不支持自定义 attention mask。`flash_attn_func` 支持 mask 但有 block-alignment 限制（通常要求 seqlen 是 128 的倍数）。

**解法**：Tree attention 部分使用 `torch.nn.functional.scaled_dot_product_attention`（PyTorch 原生 SDPA），它对小尺寸（N≤40）足够高效，会自动 dispatch 到最优后端并支持任意形状的 boolean mask。Prompt KV 部分仍使用 FlashAttention 的 paged attention。

或者简化为：整个 tree verification 的 attention 计算统一使用 `F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)`，对 N≤40 的场景性能完全足够。

**张量契约与 GQA**：为保持 `qwen3.py` 和 RoPE 接口不变，ModelRunner 将逻辑上的 padding batch `[B, N_max+1]` 展平为 `[B * (N_max+1)]` 后再调用模型。`Attention._tree_attention()` 根据 context 还原并转置为 SDPA 所需布局：

```text
Q:      [B, q_heads, N_max+1, head_dim]
K/V:    [B, kv_heads, K_max, head_dim]
mask:   [B, 1, N_max+1, K_max]  # broadcast 到所有 query heads
output: transpose + flatten -> [B * (N_max+1), q_heads, head_dim]
```

当 `q_heads != kv_heads` 时，调用 `F.scaled_dot_product_attention(..., enable_gqa=True)`；若部署环境的 PyTorch backend 不支持该参数，则在进入 SDPA 前沿 head 维显式 repeat K/V。该兼容性是 TreeAttention 的启动检查项。

### 7.2 每层 Attention 计算流程

```
输入: x_tree = [N+1, dim]  (pending_root + N个 draft tokens 的 hidden states)

1. Gather prompt KV (排除 root 的最后位置避免重复):
   K_prompt = gather(k_cache[layer], block_table, cached_prompt_len=P-1)
              # [P-1, kv_heads, dim]
   V_prompt = gather(v_cache[layer], block_table, cached_prompt_len=P-1)

2. Compute root + draft KV:
   K_tree = W_K @ x_tree  # [N+1, kv_heads, dim]  (含 root)
   V_tree = W_V @ x_tree  # [N+1, kv_heads, dim]
   # root 的 K/V 写入 target cache；draft rows 使用 slot=-1，不持久化

3. Concat + build mask:
   K_full = cat[K_prompt, K_tree]  # [(P-1)+(N+1), kv_heads, dim]
   V_full = cat[V_prompt, V_tree]

   full_mask[0:P-1, 0:P-1] = causal        # prompt 内部
   full_mask[P-1:P+N, 0:P-1] = True        # tree→prompt: all
   full_mask[P-1, P-1] = True              # root→root: self only
   full_mask[P-1, P:] = False               # root→其他tree: none
   draft_to_root_tree = cat[ones(N, 1), tree_internal_mask]  # [N, N+1]
   full_mask[P:P+N, P-1:P+N] = draft_to_root_tree

4. PyTorch SDPA:
   o = F.scaled_dot_product_attention(
       Q_tree,           # [N+1, q_heads, dim]  (含 root query)
       K_full,           # [P-1+N+1, kv_heads, dim]
       V_full,           # [P-1+N+1, kv_heads, dim]
       attn_mask=full_mask[P-1:P+N, :],
       scale=self.scale,
   )
```

> **为什么用 `F.scaled_dot_product_attention` 而不是 `flash_attn_func`**：SDPA 对小尺寸（N≤40）无 block-alignment 限制，自动 dispatch 到最优 kernel（可能是 FlashAttention、Memory-efficient attention 或 math fallback），同时支持任意 boolean mask。

> **关键设计点**: Draft tree tokens 的 KV 仅在 `K_full`/`V_full` 中存在，不调用 `store_kvcache` 写入主 cache。Root 是本轮必然保留的 pending token，使用有效 slot 写入 target cache；Commit 后接受路径上的 draft token 通过正常 decode 流程进入 cache。

### 7.3 Mask 示意

```
prompt cache (P-1=4):   [t0, t1, t2, t3]           (cached, 不含 pending_root)
tree tokens (N+1=7):    [root, c1, c2, g1, g2, g3, g4]  (pending_root + 6 drafts)

Tree → full context mask (7 rows × 11 cols):
              t0 t1 t2 t3 | r  c1 c2 g1 g2 g3 g4
         r    1  1  1  1  | 1  0  0  0  0  0  0     root → prompt + 自己
         c1   1  1  1  1  | 1  1  0  0  0  0  0     c1 → prompt + root + 自己
         c2   1  1  1  1  | 1  0  1  0  0  0  0     c2 → prompt + root + 自己
         g1   1  1  1  1  | 1  1  0  1  0  0  0     g1 → prompt + root + c1 + 自己
         g2   1  1  1  1  | 1  1  0  0  1  0  0
         g3   1  1  1  1  | 1  0  1  0  0  1  0
         g4   1  1  1  1  | 1  0  1  0  0  0  1
```

> root = pending_root，作为 tree 的第一个 query token 参与验证。其 logits 直接用于排名验证的 root 步采样。prompt cache gather 排除 root 的旧 KV 位置，避免重复。

### 7.4 位置编码

令 `P` 表示**包含 pending root 的完整上下文长度**。Root 的逻辑位置固定为 `P-1`，draft 节点的 RoPE 位置为：

```text
rope_position(root)     = P - 1
rope_position(draft[d]) = P - 1 + depth[d]
```

同层不同分支共享相同 position；tree mask 已隔离兄弟分支，因此不会互相影响。不能把排除 root 后的缓存长度直接代入 `prompt_len + depth`，否则会产生一位偏移。

Target root slot、Draft KV slot 与 RoPE position **解耦**：

| 用途 | 值 | 说明 |
|------|-----|------|
| RoPE 位置 | `P - 1 + depth` | root 为 `P-1`，同层分支共享位置 |
| Target root cache slot | sequence block table 中的 `P-1` slot | root 本轮写入主 cache |
| Draft KV slot | `TreeDraftKVManager` 分配的 pool slot | 每个 draft node 唯一，按 COW 管理 |

Draft tree token KV **不写入主 target KV cache**（因为多数会被拒绝）。验证时 draft token 的 K/V 仅在 `K_full`/`V_full` 拼接张量中存在，不持久化。Root KV 在当前轮写入主 cache；Commit 时接受路径上的 draft token 通过正常 decode 流程写入 cache。

### 7.5 Attention 模块修改

`Attention.forward()` 新增 `tree_verify` 分支，**按 token 类型区分 store 策略**：

```python
def forward(self, q, k, v):
    context = get_context()
    if context.is_tree_verify:
        # root rows: store 到 target cache (有效 slot)
        # draft rows: slot=-1，跳过 store_kvcache
        store_root_kv_only(k, v, self.k_cache, self.v_cache,
                           context.tree_root_row_indices,
                           context.slot_mapping)
        return self._tree_attention(q, k, v)
    elif context.is_prefill:
        # ... 不变
    else:
        # decode: 不变
        ...

def _tree_attention(self, q, k, v):
    # 1. gather prompt KV from cache (每层)
    # 2. concat [K_prompt, K_tree], [V_prompt, V_tree]
    # 3. F.scaled_dot_product_attention(Q_tree, K_full, V_full, attn_mask=tree_mask)
```

> 关键区别：`tree_verify` 分支对 root row **写入** target cache（避免下一轮重复计算）；对 draft rows **跳过** store（多数会被拒绝，不污染 cache）。单 sequence 时 root 是第 0 行；多 sequence padding batch 通过 `tree_root_row_indices` 定位每条序列的 root 行，其他 draft 行均为 `-1`。

### 7.6 Prompt KV Gather 说明

prompt KV 的 gather 操作在**每层 attention 调用时执行**（不同层的 KV cache 内容不同）。Gather 是 tensor indexing 操作，不涉及显存分配和拷贝。对于 36 层 × 2048 prompt × 8 kv_heads × 128 dim × bfloat16 ≈ 150MB 的读取量，在 GPU 带宽下（~1TB/s）耗时 < 0.2ms，可以接受。

伪代码：
```python
def gather_prompt_kv(k_cache_layer, block_table, cached_prompt_len, block_size):
    # k_cache_layer: [num_blocks, block_size, kv_heads, dim]
    # cached_prompt_len = P - 1，排除 pending root
    # Output: [cached_prompt_len, kv_heads, dim]
    blocks = k_cache_layer[block_table]       # gather blocks
    flat = blocks.reshape(-1, kv_heads, dim)   # flatten
    return flat[:cached_prompt_len]            # trim padding
```

### 7.7 多 Sequence 批处理

树形模式不使用 `cu_seqlens`（需要每 sequence 独立 mask），采用 **padding + block-diagonal mask**：

```
批处理流程:
1. 找到 batch 内最大的 N_max = max(N_i) 和 K_max = max(P_i + N_i)
   （每条序列 query 数为 N_i+1，key 数为 (P_i-1)+(N_i+1)=P_i+N_i）
2. 填充短序列到最大长度 (0 填充，mask 中填 False)，逻辑形状为
   Q: [B, N_max+1, q_heads, dim]，K_full: [B, K_max, kv_heads, dim]
   调用现有 Qwen3 前，将 input_ids/positions 展平为 [B * (N_max+1)]。
3. 构建 block-diagonal 掩码:
   mask[b, i, j] = 1  iff (i < N_b+1 and j < P_b+N_b and tree_mask_b[i][j])
   否则 0
4. Attention 将 Q/K/V 转置为 [B, heads, sequence, dim]，并调用
   F.scaled_dot_product_attention(Q, K_full, V_full,
                                  attn_mask=mask[:, None], enable_gqa=True)
5. 输出转置并展平回 [B * (N_max+1), q_heads, dim]，供未修改的 Qwen3 后续层使用。
```

### 7.8 Context 扩展

```python
# context 新增字段（仅 tree_verify 时有效）
is_tree_verify: bool
tree_attn_mask: torch.Tensor          # [N+1, P+N]，第 0 行是 root
prompt_k_cache: torch.Tensor          # prompt KV cache 引用 (整个 cache tensor)
prompt_v_cache: torch.Tensor
prompt_block_table: torch.Tensor      # 用于 gather: [num_prompt_blocks]
cached_prompt_len: int                # P-1，不含 pending root
root_position: int                    # P-1
tree_root_row_indices: torch.Tensor   # batch 中 root query 的行索引
tree_batch_size: int                  # B
tree_query_width: int                 # N_max + 1；用于 Q/K/V reshape
```

### 7.9 Target Tree KV Staging 与提交

Tree verification 结束前无法知道哪条 draft 路径会被接受，因此 draft tree K/V 既不能直接污染主 Target cache，也不能在每层 attention 返回后丢弃。每层 attention 必须将本层的 draft rows K/V 写入仅覆盖本轮的 `TreeTargetKVStager`：

```python
class TreeTargetKVStager:
    def stage(self, layer_id: int, draft_k: Tensor, draft_v: Tensor) -> None:
        """暂存本层所有 BFS draft node 的 K/V；root 已直接写入主 cache。"""

    def commit_path(
        self,
        accepted_paths: list[list[int]],
        target_block_ids: list[list[int]],
        first_target_positions: list[int],
    ) -> None:
        """将每条接受路径的节点 K/V scatter 到主 Target cache 的连续逻辑位置。"""

    def release(self) -> None:
        """释放本轮所有未提交的暂存 K/V。"""
```

执行顺序：

1. root row 在每层 projection 后立即写入其主 Target cache slot；draft rows 只写入 stager。
2. Target forward 完成后，rank verifier 产生每条 request 的 `accepted_paths`。
3. `commit_path()` 按路径顺序将 node K/V 写入 `target_block_ids` 对应的连续 Target slots；非接受节点从未写入主 cache。
4. scheduler 仅用 `target_block_ids` 调用 Target cache 的 commit；随后释放 stager 和 Draft COW pool 中未保留的 blocks。

这样仍只有一次 Target forward。Staging buffer 的容量按实际 tree node 数 × 层数分配或复用；它是本轮临时显存，不是 paged Target KV cache 的永久容量。

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

    # 如果到达叶子节点且仍在接受 → 该 leaf 的 logits 就是 bonus 分布
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
- **线性回退**：`tree_top_k=1` 不进入本节的 RankVerifier，而是保留当前单链 DraftProposal、`RejectionSampler` 和线性 FlashAttention 路径；TreeAttention 仅处理 `tree_top_k>=2`。

### 8.4 Root Logits 获取

排名验证的第一步需要大模型在 **root 节点**（pending_root）处的 target 分布来进行采样。

**解法：将 pending_root 作为 tree 验证输入的第一个 query row，在同一轮 Target forward 中显式计算其 logits。**

**输入结构**：

```
tree 验证输入 = [pending_root_id, draft_depth1_nodes..., draft_depth2_nodes...]
                 ^                                   ^
              root query row                   draft query rows (BFS 顺序)
```

**Attention mask 硬约束**：

```
K_full = prompt[0..P-2] + [K_root, K_draft...]

mask 必须满足:
  - root row:   prompt[0..P-2] = True,  root K = True,  其他 tree K = False
  - draft rows: prompt[0..P-2] = True,  root K = True,  祖先 K = True, 兄弟 K = False
  - root 的 K 不出现在 prompt[0..P-2] 中（已排除 P-1 位置）
```

**位置编号**：
```
prompt tokens:   [0, 1, ..., P-2]        (共 P-1 个)
root (tree):     逻辑位置 P-1            (query row index 0 in tree input)
draft depth=1:   逻辑位置 P, P, ...      (同层共享 position；按 BFS 顺序存储)
draft depth=2:   逻辑位置 P+1, P+1, ...  (同层共享 position；按 BFS 顺序存储)
```

**不可以直接用 `prompt_len + depth` 套在"排除 root 后的 prompt 长度"上**，否则会产生一位偏移。必须明确 root 的逻辑位置就是 P-1。

**Root K/V 持久化策略**：

| token | cache slot | 说明 |
|-------|-----------|------|
| root (pending_root) | **有效 slot** (P-1) | root 是本轮一定会保留的 token，应在本轮写入 target KV cache |
| draft tree nodes | **slot = -1** | 多数会被拒绝，不写入 target cache |

`slot_mapping` 必须区分对待：
```
root row:    块表[ (P-1) // block_size ] * block_size + (P-1) % block_size
tree rows:   -1 (跳过 store_kvcache)
```

**verify_rows 结构**：

树形 Target forward 的 query 只有 root 和所有 draft nodes，因此单 request 的 logits 顺序必须是：
```
verify_logits = [root_logits | draft_node0_logits | ... | draft_nodeN_logits]
                   ↑              ↑                           ↑
              第一层验证       非叶节点验证子节点       leaf 节点采样 bonus
```

树形模式**不存在额外的全局 bonus row**。每个 leaf 都有自己的条件分布 `p(next | leaf)`，rank verifier 到达该 leaf 时直接采样其 node logits。batch 实现必须显式记录每条 request 的 root 与 draft-node 行索引；rank verifier、auxiliary hidden split、Target KV stager 和 padding mask 必须使用同一套行映射，不能依赖隐含的 flatten 顺序。

### 8.5 为什么不能保存上一轮 bonus logits 作为新 root

bonus row 表示的是 `p(next_token | 最后一个 draft token)`：

```
root → d1 → d2 → d3
bonus row: p(x | d3)
```

但下一轮需要的是 `p(x | b)`，其中 b 是本轮的 bonus/recovery **采样结果**：

```
下一轮 root = b (从 p(x | d3) 采样得到的 token)
下一轮需要: p(x | b)  ≠ p(x | d3)
```

发生 rejection 时同样：recovery token 是从父节点的 logits 采样出来的，这行 logits 的条件分布与 recovery token 的下一步分布不同。因此不能通过保存上一轮 bonus/parent logits 来充当下一步 root logits。这里不影响当前树内的 bonus：当前树到达 leaf 时，直接使用该 leaf 的 node logits。

## 9. Model Runner 集成

### 9.1 prepare_spec_decode 变更

```python
def prepare_spec_decode(seqs, draft_token_ids, reservations, 
                        tree_topologies=None, include_query_lengths=False):
    if tree_topologies is not None:
        # 树形路径: 使用 padding-batched SDPA
        #   1. BFS 平整化 draft tokens
        #   2. 构建每个 request 的 tree attention mask
        #   3. Pad 所有 sequence 到相同长度 (mask 用 False 填充)
        #   4. 逻辑 Q: [B, N_max+1, dim]（第 0 行是 root）
        #      调用 model 前展平 input_ids/positions: [B * (N_max+1)]
        #      Attention 内重建 batch，并构造 K: [B, P_max+N_max, kv_heads, dim]
        #   5. 计算 root/draft-node 的 verify_row_indices，并创建临时 Target KV stager
        return (input_ids, positions, verify_rows, tree_masks, query_lengths, stager)
    else:
        # 线性路径：使用 cu_seqlens + flash_attn_varlen_func (现有行为不变)
        # 输入: flat [total_tokens] + cu_seqlens
        ...
```

> 树形和线性模式下 `input_ids` 和 `positions` 的形状不同：
> - 线性: `[total_tokens]`（所有 sequence 拼接）
> - 树形逻辑布局: `[B, N_max+1]`（第 0 列为 pending root，padding batched）；传入未修改 Qwen3 的实际布局为展平的 `[B * (N_max+1)]`。

### 9.3 混合 Batch 处理

首次实现不支持混合 Tree/linear batch：同一轮 Target forward 要么全部线性，要么全部树形。由于 `tree_top_k` 属于全局 `SpeculativeConfig`，正常运行时一个 ModelRunner 天然只处于其中一种模式。

未来若引入 per-request tree 配置，scheduler 必须按模式将请求划分为独立的 homogeneous batches；不能把 `tree_top_k=1` 请求伪装为退化 TreeTopology，否则会改变 legacy `RejectionSampler` 行为。

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
        input_ids, positions, verify_rows, tree_masks, query_lens, stager = prepared
        set_context(tree_verify=True, tree_attn_masks=tree_masks,
                    tree_kv_stager=stager, ...)
        target_output = model(input_ids, positions, aux_layer_ids)
        # root KV 写入主 cache；draft tree KV 写入本轮临时 stager
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
    # 树形: 无需第二次 Target forward；将接受路径的暂存 K/V scatter 到主 cache
    stager.commit_path(result.accepted_paths,
                       reservations.target_block_ids,
                       first_target_positions=[len(seq) for seq in seqs])
    # scheduler 仅提交 target_block_ids；Draft KV manager 释放未保留的 COW blocks
    proposer.commit(seqs, verification_aux, result)
```

## 10. 修改文件清单

| 文件 | 改动 | 复杂度 |
|------|------|--------|
| `config.py` | 新增 `tree_top_k`, `tree_max_depth`, `tree_prune_ratio` | 低 |
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

`tree_top_k=1` → 完全绕过 TreeTopology、TreeAttention、Target KV stager 和 RankVerifier，继续使用旧的线性 proposal、FlashAttention 与 `RejectionSampler`。因此固定随机种子下的输出、接受计数和 KV/block 行为均与当前版本一致。只有 `tree_top_k>=2` 进入新树形路径。

## 13. 测试计划

1. **树形拓扑正确性**：验证 BFS 线性化、mask 构建、position 分配的数学正确性
2. **Draft COW 正确性**：验证 fork/write/commit 后 Draft block 引用计数、内容和两个 reservation pool 的释放隔离
3. **Target KV staging 正确性**：验证只将接受路径 scatter 到主 Target cache；非接受分支不写入主 cache；无需第二次 Target forward
4. **SDPA 张量契约**：验证 flatten/reshape/transpose 后的输出与单请求参考实现一致，并覆盖 GQA backend 与 K/V repeat fallback
5. **排名验证正确性**：验证 root/node 行映射、leaf logits 作为 bonus 分布和 tree 分布等价性
6. **线性回退回归**：固定随机种子下 `tree_top_k=1` 与当前 `RejectionSampler` 路径的输出、接受计数和 block 行为逐项一致
7. **集成测试**：Qwen3-4B + EAGLE3 checkpoint，树形 vs 线性接受率对比
8. **边界条件**：空树、单节点树、max_depth 截断、Target/Draft block pool 容量不足；per-request 模式出现前拒绝混合 Tree/linear batch

## 14. 边界条件处理

| 场景 | 处理方式 |
|------|----------|
| `max_depth` 截断（seq 剩余长度不足） | 动态减小 `max_depth`，树在允许的深度提前终止 |
| batch 内混合树形/线性 | 首次实现拒绝混合 batch；未来 per-request 配置时由 scheduler 按模式拆分为 homogeneous batches |
| 相同 position 不同分支的 RoPE | 已验证可行：tree mask 隔离互不可见，RoPE 的同 position 编码不影响注意力计算结果 |
| 树节点数 > 预留 block 容量 | scheduler 按最大节点数预留，不足时减小 `max_depth` 或 fallback 到线性模式 |
| 树的 leaf 节点在非最大深度被截断 | 正常流程：leaf 的 `children=[]`，排名验证在此处自然终止 |
| Draft KV cache 全满 | 与当前线性版行为一致：拒绝生成更多 draft，`max_depth` 截断 |
| 动态修剪导致某层无子节点 | 树在该深度提前终止（正常行为），排名验证在终止处取 bonus token |
| `tree_prune_ratio=1`（修剪所有子节点） | 实际效果：每个节点仅保留 top-1（退化接近线性，但 root 仍可取 top-1 作为唯一子节点） |
