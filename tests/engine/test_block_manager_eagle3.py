from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


def test_disabled_prefix_cache_ignores_matching_cached_block(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    manager = BlockManager(3, 4, enable_prefix_cache=False)
    token_ids = [1, 2, 3, 4]
    block_hash = manager.compute_hash(token_ids)
    manager.blocks[1].update(block_hash, token_ids)
    manager.hash_to_block_id[block_hash] = 1

    seq = Sequence(token_ids)
    manager.allocate(seq)

    assert seq.block_table == [0]
    assert seq.num_cached_tokens == 0
    assert manager.blocks[0].hash == -1


def test_enabled_prefix_cache_preserves_hash_reuse(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    manager = BlockManager(3, 4)
    token_ids = [1, 2, 3, 4]
    block_hash = manager.compute_hash(token_ids)
    manager.blocks[1].update(block_hash, token_ids)
    manager.hash_to_block_id[block_hash] = 1

    seq = Sequence(token_ids)
    manager.allocate(seq)

    assert seq.block_table == [1]
    assert seq.num_cached_tokens == 4


def test_disabled_prefix_cache_reserves_and_commits_across_boundary(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    manager = BlockManager(4, 4, enable_prefix_cache=False)
    seq = Sequence([1, 2, 3])
    manager.allocate(seq)

    new_block_ids = manager.reserve_spec_append(seq, 2)
    seq.append_tokens([4, 5, 6])
    manager.commit_spec_append(seq, new_block_ids, num_computed_tokens=5)

    assert new_block_ids == [1]
    assert seq.block_table == [0, 1]
    assert manager.hash_to_block_id == {}
    assert all(manager.blocks[block_id].hash == -1 for block_id in seq.block_table)


def test_disabled_prefix_cache_may_append_without_hashing(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    manager = BlockManager(3, 4, enable_prefix_cache=False)
    seq = Sequence([1, 2, 3, 4])
    manager.allocate(seq)

    manager.may_append(seq)
    assert manager.blocks[seq.block_table[-1]].hash == -1

    seq.append_token(5)
    manager.may_append(seq)

    assert seq.block_table == [0, 1]
    assert manager.hash_to_block_id == {}
