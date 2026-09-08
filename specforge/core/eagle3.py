# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from specforge.core.loss import LogSoftmaxLoss
from specforge.modeling.draft import Eagle3DraftModel
from specforge.utils import padding


class OnlineEagle3Model(nn.Module):
    """EAGLE-3 training-time test (TTT) loop, run on-policy against the target.

    Per batch:

    1. The frozen target has already produced hidden states tapped from three
       of its layers, concatenated to (batch, seq_len, 3 * target_hidden).
    2. `fc` projects that down to the draft's hidden size.
    3. The draft unrolls for `length` steps. At step 0 it sees the target
       projection; at step k > 0 it sees its own previous output, so it learns
       to stay target-compatible after conditioning on its own proposals.
    4. Each step is scored against the target's next-token distribution at the
       matching lookahead position.

    The per-step losses are returned unweighted; the caller applies the
    per-step discount (paper Eq. 2, lambda = 0.8).
    """

    def __init__(
        self,
        draft_model: Eagle3DraftModel,
        length: int = 7,
        attention_backend: str = "sdpa",
    ):
        """
        Args:
            draft_model: the draft model to be trained.
            length: TTT length, i.e. how many steps to unroll during TTT.
            attention_backend: sdpa, fa or flex_attention.
        """
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend

    def _acc_and_loss(
        self,
        *,
        logits: torch.Tensor,
        target_p: torch.Tensor,
        position_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            correct = (
                (logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)
            ).sum()
            acc = correct / loss_mask.sum().clamp_min(1e-6)

        loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
        return acc, loss

    def _prepare_position_ids(
        self,
        position_ids: Optional[torch.Tensor],
        *,
        seq_length: int,
        past_key_values_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if position_ids is None:
            return (
                torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                .unsqueeze(0)
                .view(-1, seq_length)
            )

        position_ids = position_ids.long()
        return position_ids.view(-1, seq_length)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Online eagle model trainer, modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L711

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            past_key_values: We dont use this past_key_values in eagle3, but keep it for compatibility. We control kvcache by cache_hidden.
            position_ids: (batch, seq_len)
        """
        # Step 1: handle vocab size
        target_p_padded, position_mask = _compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=self.length,
        )
        del target
        torch.cuda.empty_cache()

        # basic info
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 2: project the concatenated hidden states to the target hidden size
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 3: process kv cache, position ids and position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        position_ids = self._prepare_position_ids(
            position_ids=position_ids,
            seq_length=seq_length,
            past_key_values_length=past_key_values_length,
            device=hidden_states.device,
        )

        # Step 4: handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # Step 5: run TTT
        plosses = []
        acces = []
        # See the `swap_h_and_emb` note below; fixed for the whole unroll.
        swap_h_and_emb = getattr(self.draft_model.config, "swap_h_and_emb", False)
        # `input_ids`, `position_mask` and `loss_mask` shift left by one per TTT
        # step, so keep a running copy rather than mutating the arguments.
        global_input_ids = input_ids
        if self.attention_backend in ["sdpa", "fa"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        for idx in range(self.length):
            is_last = idx == self.length - 1
            # Teacher distribution for this step's lookahead position.
            target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()

            # Step 5.1: embed the input ids
            inputs_embeds = self.draft_model.embed_input_ids(global_input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # The TTT context: fc(target features) at step 0, the previous
            # step's last-layer output afterwards.
            ttt_context = hidden_states

            # Step 5.2: run the draft model backbone.
            #
            # Which tensor is the *running residual stream* differs between a
            # from-scratch drafter and a warm-started one:
            #
            #   vanilla EAGLE-3  -- the target features are the running stream,
            #     and the token embedding is the constant per-step side.
            #   swap_h_and_emb   -- the reverse. The pretrained backbone was
            #     trained to consume token embeddings, so keeping the embedding
            #     on the residual path means layer 0 computes
            #     `W_pretrained @ LN(embedding)` at every TTT step: with the
            #     converter's [0, W_pretrained] QKV layout, the drafter starts
            #     out reproducing the source model's own forward pass. Feeding
            #     it target features there instead would put it immediately out
            #     of distribution and throw the warm start away.
            #
            # SGLang's draft head applies the same swap at inference; see
            # sglang_patches/.
            if swap_h_and_emb:
                hidden_states_out = self.draft_model.backbone(
                    input_embeds=ttt_context,
                    hidden_states=inputs_embeds,
                    cache_hidden=cache_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                hidden_states_out = self.draft_model.backbone(
                    input_embeds=inputs_embeds,
                    hidden_states=ttt_context,
                    cache_hidden=cache_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)

            # Step 5.5 + 5.6: metric and loss
            acc, loss = self._acc_and_loss(
                logits=logits,
                target_p=target_p,
                position_mask=position_mask,
                loss_mask=loss_mask,
            )
            acces.append(acc)
            plosses.append(loss)

            if not is_last:
                # Step 5.7: we need to update the loss mask
                global_input_ids = padding(global_input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Flex attention mask shrinking is handled inside the
                # attention module.
        return plosses, acces


def _compute_target_p_padded(target, t2d, loss_mask, length):
    with torch.no_grad():
        target_p, position_mask = _compute_target_p(
            target=target,
            t2d=t2d,
            loss_mask=loss_mask,
        )

        assert len(target_p.shape) == 3
        target_p_padded = F.pad(
            target_p,
            pad=(0, 0, 0, length),
            mode="constant",
            # For bitwise equality with previous code
            value=1 / target_p.shape[-1],
        )

        return target_p_padded, position_mask


@torch.compile(dynamic=None)
def _compute_target_p(target, t2d, loss_mask):
    target_head = target
    target_max_token = target_head.argmax(-1)
    target_mask = t2d[target_max_token]
    target_mask = target_mask[..., None].int()
    position_mask = target_mask * loss_mask
    target_head = target_head[..., t2d]
    target_head = target_head.float()
    target_p = nn.Softmax(dim=2)(target_head)
    target_p = target_p.detach()
    return target_p, position_mask
