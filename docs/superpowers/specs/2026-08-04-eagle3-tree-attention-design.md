# EAGLE3 TreeAttention 设计

日期：2026-08-04

状态：已确认，待实施计划

## 1. 目标

在现有线性 EAGLE3 推测解码基础上，实现**树形候选生成 + 统一验证**的 TreeAttention 功能。

核心变更：Draft model 每步采样 **top-k** 个候选 token（而非 greedy 1 个），形成 token 树；Target model 在**一次 forward** 中使用自定义 attention mask 验证所有树节点；rejection sampling 在树上递归游走选择最优路径。

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
    nodes: int                    # 总节点数（含 pending_root）
    parent: list[int]             # parent[i] = 父节点索引，root 为 -1
    children: list[list[int]]     # children[i] = [子节点索引列表]
    depth: list[int]              # depth[i]，root=0
    position_ids: list[int]       # 在原始序列中的 position (prompt_len + depth)
    bfs_indices: list[int]        # BFS 顺序 → 树节点 ID 的映射
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

## 5. Draft 生成：Top-k 树形 Proposal

### 5.1 算法

BFS 层级生成，每层节点统一 batch forward：

```
for depth in range(max_depth):
    收集当前层所有活跃节点的 (input_ids, fused_hidden, positions)
    一次 batch forward → logits, next_hidden
    每个节点采样 top_k candidates → 创建子节点
    子节点入队列（下一层）
    更新 TreeTopology
```

### 5.2 Batch 策略

同深度节点批量处理，减少 CUDA kernel launch：

| Depth | Batch size |
|-------|-----------|
| 0 | 1 (root only) |
| 1 | top_k |
| 2 | top_k^2 |
| 3 | top_k^3 |

### 5.3 Eagle3Proposer.propose() 签名变更

```python
def propose(seqs, reservations, temperatures) -> DraftProposal:
    # 返回值 DraftProposal 新增 tree_topologies 字段
```

### 5.4 DraftProposal 扩展

```python
@dataclass
class DraftProposal:
    token_ids: list[list[int]]          # 不变：平整化的 draft tokens
    probabilities: torch.Tensor         # 不变
    lengths: list[int]                  # 不变
    tree_topologies: list[TreeTopology | None]  # 新增：None = 线性模式
```

## 6. Draft KV Cache：Copy-on-Write

### 6.1 核心机制

- 子节点**共享引用**父节点的 KV blocks（引用计数）
- 首次写入共享 block 时触发 **copy-on-write**：分配新 block，复制数据，写入
- 验证完成后**只保留接受路径**的 blocks，释放其他分支

### 6.2 实现范围

由 `Eagle3Proposer` 内部管理，不耦合外层 `BlockManager`。

```python
class TreeDraftKVManager:
    def fork(self, parent_node: int, child_node: int) -> None:
        """子节点继承父节点的 block 引用，ref_count+=1"""
    
    def write(self, node: int, position: int, slot_mapping: int) -> None:
        """若 block 被共享，则 copy-on-write 分配新 block"""
    
    def commit(self, accepted_leaf: int) -> list[int]:
        """返回 root→leaf 路径上所有独占的 block IDs"""
    
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

`flash_attn_varlen_func` 不支持自定义 attention mask，只有 `flash_attn_func` 支持。

**解法**：验证时使用 `flash_attn_func`，手动拼接 prompt KV cache + tree token KV，传入自定义 mask。

### 7.2 每层 Attention 计算流程

```
输入: x_tree = [N, dim]  (tree tokens 的 hidden states)

1. Gather prompt KV from paged cache:
   K_prompt = gather(k_cache, block_table)  # [P, heads, dim]
   V_prompt = gather(v_cache, block_table)

2. Compute tree KV:
   K_tree = W_K @ x_tree  # [N, heads, dim]
   V_tree = W_V @ x_tree

3. Concat + build mask:
   K_full = [K_prompt; K_tree]  # [P+N, heads, dim]
   V_full = [V_prompt; V_tree]
   
   full_mask[0:P, 0:P] = causal (lower triangular)
   full_mask[P:P+N, 0:P] = True (tree can attend to all prompt)
   full_mask[P:P+N, P:P+N] = tree_mask (ancestor-only)

4. FlashAttention:
   # Note: flash_attn_func attn_mask: True = attend, False = mask
   # full_mask is already True=attend, pass directly
   o = flash_attn_func(
       Q_tree,           # [N, heads, dim] — only tree tokens query
       K_full,           # [P+N, heads, dim]
       V_full,           # [P+N, heads, dim]
       causal=False,
       attn_mask=full_mask[P:P+N, :]  # boolean mask, True=allow
   )
```

### 7.3 Mask 示意

```
prompt tokens (P=5):   [t0, t1, t2, t3, t4]     (cached)
tree tokens (N=7):     [r0, c1, c2, g1, g2, g3, g4]  (new)

Tree → full context mask (7 rows × 12 cols):
              t0 t1 t2 t3 t4 | r0 c1 c2 g1 g2 g3 g4
         r0    1  1  1  1  1 |  1  0  0  0  0  0  0
         c1    1  1  1  1  1 |  1  1  0  0  0  0  0
         c2    1  1  1  1  1 |  1  0  1  0  0  0  0
         g1    1  1  1  1  1 |  1  1  0  1  0  0  0
         g2    1  1  1  1  1 |  1  1  0  0  1  0  0
         g3    1  1  1  1  1 |  1  0  1  0  0  1  0
         g4    1  1  1  1  1 |  1  0  1  0  0  0  1
```

### 7.4 位置编码

每个树节点使用 `prompt_len + depth` 作为 position。同层但不同分支的节点 share 相同 position，tree mask 已隔离互不可见。

### 7.5 Attention 模块修改

`Attention.forward()` 新增 `tree_verify` 分支：

```python
def forward(self, q, k, v):
    context = get_context()
    if context.is_tree_verify:
        return self._tree_attention(q, k, v)
    elif context.is_prefill:
        ...  # 不变
    else:
        ...  # 不变

def _tree_attention(self, q, k, v):
    # 1. gather prompt KV from cache
    # 2. concat with tree KV
    # 3. flash_attn_func with tree mask
```

### 7.6 Prompt KV Gather 说明

prompt KV 的 gather 操作在**每层 attention 调用时执行**（因为不同层的 KV cache 内容不同）。Gather 本身是 tensor indexing 操作（轻量），不涉及显存拷贝。

伪代码：
```python
def gather_prompt_kv(k_cache_layer, block_table, prompt_len, block_size):
    # Gather prompt tokens' KV from paged cache to contiguous tensor
    # Input:  k_cache_layer [num_blocks, block_size, heads, dim]
    # Output: [prompt_len, heads, dim]
    blocks = k_cache_layer[block_table]       # [num_blocks, block_size, heads, dim]
    flat = blocks.reshape(-1, heads, dim)      # [num_blocks*block_size, heads, dim]
    return flat[:prompt_len]                   # trim padding
```

### 7.7 Context 扩展

```python
# context 新增字段（仅 tree_verify 时有效）
is_tree_verify: bool
tree_attn_mask: torch.Tensor          # [N, P+N] boolean mask
prompt_k_cache: torch.Tensor          # prompt KV cache 引用 (整个 cache tensor)
prompt_v_cache: torch.Tensor
prompt_block_table: torch.Tensor      # 用于 gather: [num_prompt_blocks]
prompt_seq_len: int                   # prompt token 数量
```

## 8. 树形 Rejection Sampling

### 8.1 算法

从根节点开始，在树上递归游走：

```
accepted = []
node = root
while node is not rejected:
    # 检查当前节点的所有子节点
    accepted_children = []
    for child in node.children:
        if target_prob[child] / draft_prob[child] >= rand():
            accepted_children.append(child)
    
    if not accepted_children:
        # 全部拒绝 → 从 target 采样 recovery token
        recovery = sample(target_logits[node] - draft_probs[children])
        accepted.append(recovery)
        break
    
    # 在接受的子节点中随机选择一个（按 target_prob 加权）
    child = categorical_sample(target_probs[accepted_children])
    accepted.append(child.token)
    node = child  # 继续向下

# 如果到达叶子节点 → bonus token
if node.is_leaf and node.was_accepted:
    accepted.append(target_bonus_token)
```

### 8.2 与线性版的差异

| 方面 | 线性版 | 树形版 |
|------|--------|--------|
| 遍历方式 | 顺序 for 循环 | 根→叶递归 |
| 接受条件 | 单 token vs 随机数 | 兄弟并行检查，按概率选一个 |
| 拒绝恢复 | sample(target - draft) | 同上 |
| Bonus token | 所有 K 个全接受时 | 到达叶节点继续接受时 |

线性回退：`top_k=1` → 每个节点只有 1 个子节点 → 退化为顺序扫描。

## 9. Model Runner 集成

### 9.1 prepare_spec_decode 变更

```python
def prepare_spec_decode(seqs, draft_token_ids, reservations, 
                        tree_topologies=None, include_query_lengths=False):
    if tree_topologies is not None:
        # 树形路径：
        #   1. BFS 平整化 draft tokens
        #   2. 构建每个 request 的 tree attention mask
        #   3. 计算 verify_row_indices（每个树节点 1 行 logit）
        #   4. 准备 prompt KV gather 参数
        return (input_ids, positions, verify_rows, tree_masks, query_lengths)
    else:
        # 线性路径：现有行为不变
        ...
```

### 9.2 run_eagle3_spec_decode 变更

```python
def run_eagle3_spec_decode(seqs, reservations):
    # 1. Draft proposal
    proposal = proposer.propose(...)
    
    # 2. Target verification
    if proposal.tree_topologies:
        prepared = prepare_spec_decode(seqs, proposal, reservations,
                                       proposal.tree_topologies, include_query_lengths=True)
        input_ids, positions, verify_rows, tree_masks, query_lens = prepared
        set_context(tree_verify=True, tree_attn_masks=tree_masks, ...)
        target_output = model(input_ids, positions, aux_layer_ids)
    else:
        # 线性路径：现有行为不变
        ...
    
    # 3. Rejection sampling
    result = rejection_sampler(proposal, verify_logits, tree_topologies)
    
    # 4. Commit: 只保留接受路径的 KV blocks
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
| `v1/sample/rejection_sampler.py` | 线性扫描 → 树形递归游走 | 中 |
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
3. **概率分布验证**：树形 rejection sampling 与线性版的概率一致性（top_k=1 时输出相同）
4. **集成测试**：Qwen3-4B + EAGLE3 checkpoint，树形 vs 线性接受率对比
5. **边界条件**：空树、单节点树、max_depth 截断、batch 混合（树形+线性）

## 14. 边界条件处理

| 场景 | 处理方式 |
|------|----------|
| `max_depth` 截断（seq 剩余长度不足） | 动态减小 `max_depth`，树在允许的深度提前终止 |
| batch 内混合树形/线性 | 通过 `tree_topologies` 的 per-sequence None/非None 区分。线性序列走原代码路径，树形序列走新路径 |
| 相同 position 不同分支的 RoPE | 已验证可行：tree mask 隔离互不可见，RoPE 的同 position 编码不影响注意力计算结果 |
| 树节点数 > 预留 block 容量 | scheduler 预留阶段检查，不足时 fallback 到线性模式 |
| 树的 leaf 节点在非最大深度被截断 | 正常流程：leaf 的 `children=[]`，rejection sampling 在此处自然终止 |
| Draft KV cache 全满 | 与当前线性版行为一致：拒绝生成更多 draft，`max_depth` 截断 |
