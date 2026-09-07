import torch

from nanovllm.v1.sample.rejection_sampler import RankVerifier
from nanovllm.v1.spec_decode.types import DraftProposal, TreeTopology


class ArgmaxSampler:
    def __call__(self, logits, temperatures):
        return logits.argmax(dim=-1)


def make_depth_two_proposal(token_id=5):
    topology = TreeTopology(
        total_nodes=2,
        draft_nodes=1,
        parent=[-1, 0],
        children=[[1], []],
        depth=[0, 1],
        rope_positions=[4],
        bfs_to_node=[1],
    )
    return DraftProposal(
        token_ids=[[token_id]],
        probabilities=torch.ones((1, 8)),
        lengths=[1],
        tree_topologies=[topology],
    )


def test_rank_verifier_accepts_child_then_samples_leaf_bonus():
    proposal = make_depth_two_proposal(token_id=5)
    logits = torch.zeros((2, 8))
    logits[0, 5] = 10
    logits[1, 7] = 10

    result = RankVerifier(ArgmaxSampler())(
        proposal,
        logits,
        torch.zeros(1),
        [[0, 1]],
    )

    assert result.output_token_ids == [[5, 7]]
    assert result.accepted_draft_counts == [1]
    assert result.accepted_paths == [[1]]


def test_rank_verifier_stops_at_root_when_no_child_matches():
    proposal = make_depth_two_proposal(token_id=5)
    logits = torch.zeros((2, 8))
    logits[0, 4] = 10

    result = RankVerifier(ArgmaxSampler())(
        proposal,
        logits,
        torch.zeros(1),
        [[0, 1]],
    )

    assert result.output_token_ids == [[4]]
    assert result.accepted_draft_counts == [0]
    assert result.accepted_paths == [[]]
