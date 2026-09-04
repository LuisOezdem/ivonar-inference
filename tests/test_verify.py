from __future__ import annotations

import pytest
import torch

from ivonar_inference.decoder import StaticDecoder
from ivonar_inference.verify import DecoderCheck, check_decoder
from conftest import tiny_model


@torch.no_grad()
def test_check_decoder_agrees_with_the_one_pass_forward() -> None:
    torch.manual_seed(12)
    model = tiny_model(seed=12)
    ids = torch.randint(2, 300, (20,)).tolist()
    decoder = StaticDecoder(model, max_len=32, device="cpu", fuse_projections=True)
    check = check_decoder(model, decoder, ids, split=8)
    assert check.positions == 12
    assert check.top1_agreement == 1.0
    assert check.max_abs_diff < 1e-2
    assert abs(check.decoder_log_prob - check.reference_log_prob) < 1e-3
    assert check.passed
    assert check.lines()[-1] == "result: PASS"
    with pytest.raises(ValueError, match="split"):
        check_decoder(model, decoder, ids, split=0)


def test_decoder_check_fails_on_disagreement() -> None:
    check = DecoderCheck(
        positions=10,
        max_abs_diff=3.0,
        mean_abs_diff=0.5,
        top1_agreement=0.6,
        reference_log_prob=-2.0,
        decoder_log_prob=-2.5,
    )
    assert not check.passed
    assert check.lines()[-1] == "result: FAIL"
