# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.models import FairseqEncoder
from fairseq.modules import (FairseqDropout, LayerDropModuleList,
                             LayerNorm, PositionalEmbedding,
                             SinusoidalPositionalEmbedding)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


def get_sinusoidal_positional_embedding(length, embed_dim):
    half_dim = embed_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(length, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(length, -1)
    if embed_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(length, 1)], dim=1)
    return emb

@with_incremental_state
class LunarMultiheadAttention(nn.Module):
    """Lunar Multi-headed attention.
    See "Linformer: Self-Attention with Linear Complexity" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_pheads,
        dropout=0.0,
        bias=True,
        self_attention=False,
        encoder_decoder_attention=False,
        tie_kv=True,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_pheads = num_pheads
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)

        self.head_dim = embed_dim // num_heads
        self.phead_dim = embed_dim // num_pheads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        assert (self.phead_dim * num_pheads == self.embed_dim), "projected embed_dim must be divisible by num_pheads"
        self.scaling = self.head_dim ** -0.5
        self.pscaling = self.phead_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.pq_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        if tie_kv:
            self.pc_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
            self.c_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
            self.pk_proj = self.k_proj = None
            self.pv_proj = self.v_proj = None
        else:
            self.pk_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
            self.pv_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
            self.k_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
            self.v_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
            self.pc_proj = self.c_proj = None

        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
        raise NotImplementedError('onnx for linear attention not implemented')

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True
        raise NotImplementedError('TPU for linear attention not implemented')

    def reset_parameters(self):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.pq_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)

        if self.pc_proj is not None:
            nn.init.xavier_uniform_(self.pc_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.c_proj.weight, gain=gain)
        else:
            nn.init.xavier_uniform_(self.pk_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.pv_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def _compute_pcontext(self, pquery, context, context_padding_mask):
        # N x B x D
        len, bsz, dim = context.size()
        if self.pc_proj is not None:
            c = self.pc_proj(context)
            # N x B x D -> N x B*H x K
            k = v = c.view(len, bsz * self.num_pheads, self.phead_dim)
        else:
            # N x B x D -> N x B*H x K
            k = self.pk_proj(context).view(len, bsz * self.num_pheads, self.phead_dim)
            v = self.pv_proj(context).view(len, bsz * self.num_pheads, self.phead_dim)

        # N x B*H x K -> B*H x K x N
        k = k.permute(1, 2, 0)
        # N x B*H x K -> B*H x N x K
        v = v.transpose(0, 1)

        plen = pquery.size(0)
        # L x B x D -> L x B*H x K
        pq = self.pq_proj(pquery).view(plen, bsz * self.num_pheads, self.phead_dim)
        # L x B*H x K -> B*H x L x K
        pq = pq.transpose(0, 1) * self.pscaling
        # B*H x L x N
        pqk = torch.bmm(pq, k)
        if context_padding_mask is not None:
            pqk = pqk.view(bsz, self.num_pheads, plen, len)
            pqk = pqk.masked_fill(context_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            pqk = pqk.view(bsz * self.num_pheads, plen, len)

        pqk = F.softmax(pqk, dim=-1)
        pqk = self.dropout_module(pqk)
        # B*H x L x K
        pc = torch.bmm(pqk, v)
        # B*H x L x K -> L x B*H x K -> L x B x D
        pc = pc.transpose(0, 1).contiguous().view(plen, bsz, dim)
        return pc

    def compute_pcontext(self,
        query,
        pquery,
        context: Optional[Tensor],
        context_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        static_context: bool = False,
    ) -> Union[Tensor, None]:

        if context is None:
            return context
        else:
            return self._compute_pcontext(pquery, context, context_padding_mask)

    def forward(
        self,
        query,
        pquery,
        context: Optional[Tensor],
        context_padding_mask: Optional[Tensor] = None,
        pcontext_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        static_context: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            context_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert not self.self_attention or incremental_state is None, \
            'For incremental self attention (causal attention), please use LunarCausalAttention'

        if self.self_attention:
            context = query

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_context:
                    assert self.encoder_decoder_attention and not self.self_attention
                    context = None
        else:
            saved_state = None

        # L x B x D
        pcontext = self.compute_pcontext(query, pquery, context, context_padding_mask,
                                         incremental_state, static_context)

        key_padding_mask = pcontext_padding_mask

        q = self.q_proj(query)
        if pcontext is None:
            assert context is None
            k = v = None
        elif self.c_proj is not None:
            k = v = self.c_proj(pcontext).view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            k = self.k_proj(pcontext).view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = self.v_proj(pcontext).view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        q = q * self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_context:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_context:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            # pcontext are stored with shape (bsz, proj_len, model_dim)
            if "prev_pcontext" in saved_state:
                # TODO save prev_pcontext for causal attention
                _prev_pcontext = saved_state["prev_pcontext"]
                assert _prev_pcontext is not None
                prev_pcontext = _prev_pcontext.transpose(0, 1)
                if static_context:
                    pcontext = prev_pcontext
                else:
                    raise RuntimeError('pcontext error')

            assert k is not None and v is not None and pcontext is not None
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_pcontext"] = pcontext.transpose(0, 1)
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None and v is not None and pcontext is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = LunarMultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, pcontext, attn_weights

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

class LunaEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, index):
        super().__init__()
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.index = index
        self.embed_dim = args.encoder_embed_dim
        self.normalize_before = args.encoder_normalize_before

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)

        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.self_atten_proj_layer_norm = LayerNorm(self.embed_dim)

        self.activation_fn = utils.get_activation_fn(activation=getattr(args, "activation_fn", "relu"))
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(float(activation_dropout_p), module_name=self.__class__.__name__)

        self.fc1 = self.build_fc1(self.embed_dim, args.encoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.fc2 = self.build_fc2(args.encoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return LunarMultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            args.encoder_projected_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            tie_kv=not args.untie_luna_kv,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, px, encoder_padding_mask, encoder_projected_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            px (Tensor): projected input to the layer of shape `(proj_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            encoder_projected_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, proj_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
            projected output of shape `(proj_len, batch, embed_dim)`
        """

        residual = x
        presidual = px
        # apply prev layer norm
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
            px = self.self_atten_proj_layer_norm(px)

        x, px, _ = self.self_attn(query=x, pquery=px, context=x,
                                  context_padding_mask=encoder_padding_mask,
                                  pcontext_padding_mask=encoder_projected_padding_mask)
        # apply dropout
        x = self.dropout_module(x)
        px = self.dropout_module(px)
        # residual
        x = residual + x
        px = presidual + px

        # apply post layer norm
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
            px = self.self_atten_proj_layer_norm(px)

        #######################################################################
        # Feed-Forward Network
        residual = x
        # apply prev layer norm
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        # apply dropout
        x = self.dropout_module(x)
        # residual
        x = residual + x

        # apply post layer norm
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, px


class LunaEncoder(FairseqEncoder):
    """
    Luna encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`LunaEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        assert embed_dim == args.encoder_embed_dim, 'encoder embedding dim mismatch.'
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        assert not args.layernorm_embedding or not args.encoder_normalize_before

        if args.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
            self.layernorm_porjected_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None
            self.layernorm_porjected_embedding = None

        self.proj_len = args.projection_length
        self.dynamic_projection = not args.fix_projection_length
        self.projected_embeddings = Parameter(torch.Tensor(self.proj_len, embed_dim))
        nn.init.normal_(self.projected_embeddings, mean=0., std=embed_dim ** -0.5)
        if not args.no_token_positional_embeddings and not args.encoder_learned_pos:
            projected_positions = get_sinusoidal_positional_embedding(self.proj_len, embed_dim)
        else:
            projected_positions = None
        self.register_buffer('projected_positions', projected_positions)

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([self.build_encoder_layer(i, args) for i in range(args.encoder_layers)])
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
            self.proj_layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
            self.proj_layer_norm = None

    def build_encoder_layer(self, layer_id, args):
        return LunaEncoderLayer(args, layer_id)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        return x, embed

    def forward_projected_embedding(self, src_lengths):
        max_len = src_lengths.max() if self.dynamic_projection else self.proj_len
        px = proj_embed = self.embed_scale * self.projected_embeddings[:max_len]
        if self.projected_positions is not None:
            px = px + self.projected_positions[:max_len]
        if self.layernorm_porjected_embedding is not None:
            px = self.layernorm_porjected_embedding(px)

        return px, proj_embed

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False, token_embeddings: Optional[torch.Tensor] = None):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens)
        px, projected_embedding = self.forward_projected_embedding(src_lengths)

        bsz = x.size(0)
        len, dim = px.size()

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # L x C -> L x B x C
        px = px.unsqueeze(1).expand(len, bsz, dim)

        x = self.dropout_module(x)
        px = self.dropout_module(px)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if self.dynamic_projection:
            pidx= torch.arange(len).unsqueeze(0).to(x.device)
            encoder_projected_padding_mask = pidx.ge(src_lengths.unsqueeze(1))
        else:
            encoder_projected_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x, px = layer(x, px, encoder_padding_mask, encoder_projected_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append((x, px))

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            px = self.proj_layer_norm(px)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }
    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict
