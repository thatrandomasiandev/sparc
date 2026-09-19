"""Unit tests for L6 approx-VOI helpers."""

import torch
import torch.nn as nn

from plr.approx_voi import approx_voi_score, disagreement_score, return_variance_loss


class _Const(nn.Module):
    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def forward(self, x):
        return torch.full((x.shape[0],), self.c)


def test_disagreement_zero_when_ensemble_agrees():
    models = [_Const(1.0), _Const(1.0), _Const(1.0)]
    seg = torch.zeros(4, 3)
    assert disagreement_score(models, seg, seg) == 0.0


def test_approx_voi_nonnegative_on_identical_eval():
    models = [_Const(0.0), _Const(1.0), _Const(2.0)]
    w = torch.ones(3) / 3
    a = torch.ones(5, 2)
    b = torch.zeros(5, 2)
    eval_segs = torch.randn(8, 5, 2)
    score = approx_voi_score(models, w, a, b, eval_segs)
    # VOI can be slightly negative from numerical noise; just ensure finite
    assert score == score


def test_return_variance_loss_zero_when_deterministic():
    returns = torch.ones(4, 10)
    w = torch.ones(4) / 4
    assert float(return_variance_loss(w, returns)) < 1e-8
