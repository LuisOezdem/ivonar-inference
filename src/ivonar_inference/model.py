from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import LatentAttention
from .config import ModelConfig
from .layers import GatedFeedForward, TokenHead, RMSNorm
from .recurrent import RecurrentMixer


class IvonarBlock(nn.Module):
    """One residual block: a normalized sequence mixer followed by a normalized feed-forward."""

    def __init__(
        self,
        mixer_type: str,
        mixer_norm: RMSNorm,
        mixer: RecurrentMixer | LatentAttention,
        ffn_norm: RMSNorm,
        ffn: GatedFeedForward,
    ) -> None:
        super().__init__()
        if mixer_type not in {"mamba", "mla"}:
            raise ValueError("layer type must be mamba or mla")
        self.mixer_type = mixer_type
        self.mixer_norm = mixer_norm
        self.mixer = mixer
        self.ffn_norm = ffn_norm
        self.ffn = ffn

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        mixer_out, state = self.mixer(self.mixer_norm(x))
        x = x + mixer_out
        return x + self.ffn(self.ffn_norm(x)), state


class IvonarModel(nn.Module):
    """An Ivonar language model assembled from packed ternary weights.

    ``forward`` scores a whole sequence in one pass; ``prefill`` does the same
    for a prompt and hands the per-layer states to the decoder.
    """

    def __init__(
        self,
        config: ModelConfig,
        token_io: TokenHead,
        blocks: list[IvonarBlock],
        final_norm: RMSNorm,
    ) -> None:
        super().__init__()
        if len(blocks) != config.num_layers:
            raise ValueError(f"expected {config.num_layers} blocks, got {len(blocks)}")
        self.config = config
        self.token_io = token_io
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = final_norm

    @property
    def device(self) -> torch.device:
        return self.token_io.projection.packed_weight.device

    def _run_blocks(self, input_ids: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, seq_len]")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have non-empty batch and sequence dimensions")
        if input_ids.shape[1] > self.config.seq_len:
            raise ValueError("input sequence length exceeds the model context")
        x = self.token_io.embed(input_ids)
        states: list[dict[str, Tensor]] = []
        for block in self.blocks:
            x, state = block(x)
            states.append(state)
        return self.final_norm(x), states

    @torch.no_grad()
    def forward(self, input_ids: Tensor) -> Tensor:
        """Logits for every position of ``input_ids``, shape [batch_size, seq_len, vocab_size]."""

        x, _ = self._run_blocks(input_ids)
        return self.token_io.project(x)

    @torch.no_grad()
    def prefill(self, input_ids: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]:
        """Run a prompt in one pass; returns the last position's logits and the per-layer states."""

        x, states = self._run_blocks(input_ids)
        return self.token_io.project(x[:, -1]), states
