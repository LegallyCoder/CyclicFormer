
"""PyTorch CyclicFormer model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from configuration_cyclicformer import CyclicFormerConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class FeedForward(nn.Module):
    def __init__(self, hidden_size, drop_prob=None):
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(hidden_size,hidden_size)
        self.lin2 = nn.Linear(hidden_size,hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x

class CyclicAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, cyclic_size):
        super(CyclicAttention, self).__init__()
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.d_k = hidden_size // num_attention_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.c_proj = nn.Linear(hidden_size, cyclic_size)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, hidden_size = x.size()
        return x.view(batch_size, seq_length, self.num_attention_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.q_proj(q))
        k = self.split_heads(self.k_proj(k))
        v = self.split_heads(self.v_proj(v))
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        h = self.o_proj(self.combine_heads(attn_output))
        c = self.c_proj(self.combine_heads(attn_output))
        return c,h

class DecoderLayer(nn.Module):
    def __init__(self, config: CyclicFormerConfig):
        super(DecoderLayer, self).__init__()
        self.attention = CyclicAttention(hidden_size = config.hidden_size ,num_attention_heads = config.num_attention_heads, cyclic_size = config.cyclic_size)
        self.norm1 = LayerNorm(hidden_size=config.hidden_size)
        self.dropout1 = nn.Dropout(p=config.drop_prob)

        self.ffn = FeedForward(hidden_size=config.hidden_size, drop_prob=config.drop_prob)
        self.norm2 = LayerNorm(hidden_size=config.hidden_size)
        self.dropout2 = nn.Dropout(p=config.drop_prob)

    def forward(self, x, mask):
        _x = x
        c, x = self.attention(q=x, k=x, v=x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return c,x

class CyclicFormerPreTrainedModel(PreTrainedModel):
    config_class = CyclicFormerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class CyclicFormerModel(CyclicFormerPreTrainedModel):
    def __init__(self, config: CyclicFormerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.n_loop = config.n_loop
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None)-> Union[Tuple, BaseModelOutputWithPast]:
        x = self.embed_tokens(input_ids)
        c = 0
        all_hidden_states = list()
        cyclic_attentions = torch.zeros(self.n_loop, self.config.num_hidden_layers, input_ids.shape[0], input_ids.shape[1], self.config.cyclic_size)
        for i in range(self.n_loop):
          z = 0
          for layer in self.layers:
              c,x = layer(x, attention_mask)
              cyclic_attentions[i][z] = c
              x = self.norm(x)
              all_hidden_states.append(x)
              z += 1
          x = all_hidden_states[-1]
        all_hidden_states = list()
        for i in range(self.n_loop):
          all_hidden_states.append(torch.cat(cyclic_attentions[i].unbind(dim=0), dim=-1))
        return BaseModelOutputWithPast(
                last_hidden_state = torch.cat(sum(cyclic_attentions).unbind(dim=0), dim=-1),
                hidden_states=all_hidden_states,
            )

class CyclicFormerForCausalLM(CyclicFormerPreTrainedModel):
    def __init__(self, config: CyclicFormerConfig):
        super().__init__(config)
        self.model = CyclicFormerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear((config.cyclic_size * config.num_hidden_layers), config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self,
                input_ids: torch.LongTensor = None,
                labels: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                )-> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self, input_ids, **kwargs
    ):
        model_inputs = {"input_ids": input_ids}
        return model_inputs



class CyclicFormerForSequenceClassification(CyclicFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = CyclicFormerModel(config)
        self.score = nn.Linear((config.cyclic_size * config.num_hidden_layers), self.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
        )


