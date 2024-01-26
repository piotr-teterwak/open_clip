""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re

import torch
import torch.nn as nn
from torch import TensorType

import types

import peft
from peft import LoraConfig, get_peft_model

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PretrainedConfig, LlamaConfig,GenerationConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None


    class BaseModelOutput:
        pass


    class PretrainedConfig:
        pass

from .hf_configs import arch_dict

class ResMLP(torch.nn.Module):
    def __init__(self,
                 bottleneck_size=512,
                 module_type='MLP1',
                 emb_dimension=5120,
                 nonlinearity='relu', # activation function
                 layer_norm=True,
                 dropout=0.0,
                 residual=True,
                 ):
        """MLP class for soft prompt re-parameterization. MLP can have a Residual connection.
        Args:
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used.
                Currently supports 1-layer and 2-layer MLPs, and simple transformer layer ('MLP1'/'MLP2'/'transformer').
                Defaults to 'MLP1'.
            emb_dimension (int, optional): Dimension of T5 model embeddings. Defaults to 512 (T5-small embedding dimension).
            residual (bool, optional): Whether to use residual connection in MLP. Defaults to True.
        """
        super().__init__()
        assert module_type in ['MLP1', 'MLP2', 'transformer', 'LSTM', 'LSTM1', 'LSTM2']
        assert nonlinearity in ['relu', 'tanh', 'sigm']

        self.module_type = module_type

        if module_type not in ['LSTM', 'LSTM1', 'LSTM2', 'transformer']:
            layers = [nn.Linear(emb_dimension, bottleneck_size)]

            if nonlinearity=='relu':
                layers.append(nn.ReLU())
            elif nonlinearity=='tanh':
                layers.append(nn.Tanh())
            elif nonlinearity=='sigm':
                layers.append(nn.Sigmoid())

            layers.append(nn.Linear(bottleneck_size, emb_dimension))

            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            if layer_norm:
                layers.append(nn.LayerNorm(emb_dimension))

            if module_type=='MLP2':
                layers = layers + layers # repeat twice
            self.module = torch.nn.Sequential(*layers)

        elif module_type in ['LSTM1', 'LSTM2', 'LSTM']:
            self.lstm_head = torch.nn.LSTM(input_size=emb_dimension,
                                           hidden_size=emb_dimension // 2,
                                           num_layers=1 if module_type=='LSTM1' else 2,
                                           dropout=0.05,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(emb_dimension, emb_dimension),
                                          nn.ReLU(),
                                          nn.Linear(emb_dimension, emb_dimension))


        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        if self.module_type=='LSTM':
            output_embeds = self.mlp_head(self.lstm_head(inputs)[0]).squeeze()
        elif self.module_type in ['LSTM1', 'LSTM2']:
            output_embeds = self.lstm_head(inputs)[0].squeeze()
            if self.residual:
                output_embeds += inputs
            return output_embeds

        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)



# utils

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _create_mask_func(prompt_tuning_tokens):
    def _prepare_decoder_attention_mask(self,attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

            combined_attention_mask[:,:,:-prompt_tuning_tokens,-prompt_tuning_tokens:] = torch.finfo(inputs_embeds.dtype).min
            combined_attention_mask[:,:,-prompt_tuning_tokens:,:] = 0.0
        return combined_attention_mask
    return _prepare_decoder_attention_mask


def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


# TODO: ?last - for gpt-like models
_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.hidden_states[-1] * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class AttentionPooler(nn.Module):
    """Mean pooling"""

    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.in_proj = nn.Linear(in_dim,hidden_dim,bias=False)
        self.attn_pool = nn.MultiheadAttention(hidden_dim,8,batch_first=True)


    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        x = self.in_proj(x.hidden_states[-1])
        x = self.attn_pool(x,x,x,key_padding_mask=~attention_mask.to(torch.bool))[0]
        x = x[:,0,:]
        return x

@register_pooler
class FancyAttentionPooler(nn.Module):
    def __init__(self,
        in_dim,
        hidden_dim,
        norm =nn.LayerNorm
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim,hidden_dim,bias=False)
        self.attn_pool = nn.MultiheadAttention(hidden_dim,8,batch_first=True)
        self.cls_q = nn.Parameter(torch.zeros(1,1,hidden_dim))
        self.norm = norm(hidden_dim)
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        batch_size = x.hidden_states[-1].shape[0]
        x = self.in_proj(x.hidden_states[-1])
        query = torch.tile(self.cls_q,(batch_size,1,1))
        x = self.attn_pool(query,x,x,key_padding_mask=~attention_mask.to(torch.bool))[0]
        x = x[:,0,:]
        return self.norm(x)

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



@register_pooler
class EosAttentionPooler(nn.Module):
    def __init__(self,
        in_dim,
        num_tokens,
        norm =nn.LayerNorm
    ):
        super().__init__()
        self.attn_pool = nn.MultiheadAttention(in_dim,8,batch_first=True)
        self.cls_q = nn.Parameter(torch.zeros(1,1,in_dim))
        self.norm = norm(in_dim)
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)
        self.num_tokens = num_tokens

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        batch_size = x.hidden_states[-1].shape[0]
        x = x.hidden_states[-1][:, -self.num_tokens:, :]
        query = torch.tile(self.cls_q,(batch_size,1,1))
        x = self.attn_pool(query,x,x)[0]
        x = x[:,0,:]
        return self.norm(x)

@register_pooler
class AllAttentionPooler(nn.Module):
    def __init__(self,
        in_dim,
        num_tokens,
        norm =nn.LayerNorm
    ):
        super().__init__()
        self.attn_pool = nn.MultiheadAttention(in_dim,8,batch_first=True)
        self.cls_q = nn.Parameter(torch.zeros(1,1,in_dim))
        self.norm = norm(in_dim)
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)
        self.num_tokens = num_tokens

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        batch_size = x.hidden_states[-1].shape[0]
        x = x.hidden_states[-1][:, :-self.num_tokens, :]
        query = torch.tile(self.cls_q,(batch_size,1,1))
        x = self.attn_pool(query,x,x)[0]
        x = x[:,0,:]
        return self.norm(x)



    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

@register_pooler
class EfficientEosAttentionPooler(nn.Module):
    def __init__(self,
        in_dim,
        h_dim,
        num_tokens,
        norm =nn.LayerNorm
    ):
        super().__init__()
        self.attn_pool = nn.MultiheadAttention(h_dim,8,kdim=in_dim,vdim=in_dim,batch_first=True)
        self.cls_q = nn.Parameter(torch.zeros(1,1,h_dim))
        self.norm = norm(h_dim)
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)
        self.num_tokens = num_tokens

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        batch_size = x.hidden_states[-1].shape[0]
        x = x.hidden_states[-1][:, -self.num_tokens:, :]
        query = torch.tile(self.cls_q,(batch_size,1,1))
        x = self.attn_pool(query,x,x)[0]
        x = x[:,0,:]
        return self.norm(x)



    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.hidden_states[-1].masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        if (self.use_pooler_output and
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.hidden_states[-1][:, self.cls_token_position, :]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return x.hidden_states[-1][:, self.cls_token_position, :]


@register_pooler
class EosPooler(nn.Module):
    """EOS token pooling
    """

    def __init__(self, num_tokens=24):
        super().__init__()
        self.eos_token_position = -1 * num_tokens

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return torch.mean(x.hidden_states[-1][:, self.eos_token_position:, :],dim=1)

@register_pooler
class LearnedEosPooler(nn.Module):
    """EOS token pooling
    """

    def __init__(self, num_tokens=24):
        super().__init__()
        self.eos_token_position = -1 * num_tokens
        self.linear = torch.nn.Linear(num_tokens, 1, bias=False)

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        x = x.hidden_states[-1][:,self.eos_token_position:,:]
        x = x.permute(0,2,1)
        x  = self.linear(x)
        x = x.permute(0,2,1)[:,0,:]
        return x



class PartialOverrideEmbedding(nn.Module):
    def __init__(self,
                wte: nn.Embedding,
                start_override: int = 110, # [unused100] for my transformer
                length_override: int = 800, # [unused900] for my transformer
                initialize_from_vocab: bool = True):
        """appends learned embedding to
        Args:
            wte (nn.Embedding): original transformer word embedding
            start_override (int, optional): first token id which will be trained separately. Defaults to 110 ([unused100] for BERT).
            length_override (int, optional): how many tokens are to be trained separately after the first. Defaults to 800 ([unused900] for BERT).
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(PartialOverrideEmbedding, self).__init__()
        self.start_override = start_override
        self.length_override = length_override
        self.wte = wte
        self.wte.weight.requires_grad=False
        #self.reparam = ResMLP()
        self.wte_override = nn.Embedding(
            length_override, wte.weight.shape[1]
        )
        if initialize_from_vocab:
            with torch.no_grad():
                #self.wte_override.weight[:] = torch.mean(wte.weight,dim=0,keepdims=True)
                self.wte_override.weight[:] = self.wte.weight[self.start_override:self.start_override+self.length_override] + torch.mean(wte.weight[0:self.start_override],dim=0,keepdims=True)
        self.initial_start_override = start_override
        self.initial_start_override = start_override
        self.initial_length_override = length_override

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        # Detect which tokens are not in range for the override, and prepare masks for them
        mask_below = (tokens < self.start_override)
        mask_above = (tokens >= self.start_override + self.length_override)
        mask_out = torch.logical_or(mask_below, mask_above)



        embedded_tokens = self.wte(tokens)

        # Every token without representation has to be brought into appropriate range
        modified_tokens = tokens - self.start_override
        # Zero out the ones which already have pretrained embedding
        modified_tokens[mask_out] = 0
        # Get the
        #embedded_tokens_after_override = self.reparam(self.wte_override(modified_tokens))
        embedded_tokens_after_override = self.wte_override(modified_tokens)
        #embedded_tokens_after_override += torch.normal(torch.zeros_like(embedded_tokens_after_override), torch.ones_like(embedded_tokens_after_override)*0.02)

        # And finally change appropriate tokens from placeholder embedding created by
        # pretrained into trainable embeddings.
        #return embedded_tokens * torch.logical_not(mask_out) + embedded_tokens_after_override * mask_out
        embedded_tokens_after_override[mask_out] = embedded_tokens[mask_out]

        return embedded_tokens_after_override

    def commit_changes(self):
        with torch.no_grad():
            self.wte.weight[self.initial_start_override:self.initial_start_override+self.initial_length_override] = self.wte_override.weight[:].detach().clone()



class HFTextEncoder(nn.Module):
    """HuggingFace model adapter"""
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            model_name_or_path: str,
            output_dim: int,
            hidden_dim: int,
            config: PretrainedConfig = None,
            pooler_type: str = None,
            proj: str = None,
            pretrained: bool = True,
            output_tokens: bool = False,
            load_pretrained_checkpoint: bool = True,
            generate: bool = False,
            prompt_tuning: bool = False,
            num_prompt_tokens: int = 10,
            num_prefix_tokens: int = 0,
            lora: bool = False
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.load_pretrained_checkpoint = load_pretrained_checkpoint
        self.generate = generate
        self.prompt_tuning = prompt_tuning
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.num_prompt_tokens = num_prompt_tokens
        self.num_prefix_tokens = num_prefix_tokens
        self.lora = lora


        # TODO: find better way to get this information
        uses_transformer_pooler = (pooler_type == "cls_pooler")

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            create_func, model_args = (AutoModelForCausalLM.from_pretrained, model_name_or_path) if pretrained else (
                AutoModelForCausalLM.from_config, self.config)
            # TODO: do all model configs have this attribute? PretrainedConfig does so yes??
            if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
                self.transformer = create_func(model_args,output_hidden_states=True)
                self.transformer = self.transformer.encoder
            elif isinstance(self.config, LlamaConfig):
                self.transformer = create_func(model_args, output_hidden_states=True)
            else:
                self.transformer = create_func(model_args, add_pooling_layer=uses_transformer_pooler,output_hidden_states=True)
        else:
            self.config = config
            self.transformer = AutoModelForCausalLM.from_config(config)

        if self.lora:
            config = LoraConfig(
                 r=16,
                 lora_alpha=16,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                 lora_dropout=0.1,
                 bias="none",
            )
            self.transformer = get_peft_model(self.transformer, config)
            self.transformer.lm_head.weight.requires_grad = False
        if pooler_type is None:  # get default arch pooler
            pooler_type = (arch_dict[self.config.model_type]["pooler"])
        pooler_kwargs = {}
        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])

        if pooler_type == "attention_pooler" or pooler_type =="fancy_attention_pooler":
            pooler_kwargs["in_dim"] = d_model
            pooler_kwargs["hidden_dim"] = hidden_dim
            d_model = hidden_dim
        elif pooler_type == "eos_attention_pooler":
            pooler_kwargs["in_dim"] = d_model
            pooler_kwargs["num_tokens"] = self.num_prompt_tokens
        elif pooler_type == "efficient_eos_attention_pooler":
            pooler_kwargs["in_dim"] = d_model
            pooler_kwargs["num_tokens"] = self.num_prompt_tokens
            pooler_kwargs["h_dim"] = output_dim
            d_model = output_dim
        elif pooler_type == "eos_pooler":
            pooler_kwargs["num_tokens"] = self.num_prompt_tokens
        elif pooler_type == "learned_eos_pooler":
            pooler_kwargs["num_tokens"] = self.num_prompt_tokens


        # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)


        self.pooler = _POOLERS[pooler_type](**pooler_kwargs)



        if self.prompt_tuning == True:
            #This was moved, remember for eval
            self.transformer.resize_token_embeddings(self.vocab_size + self.num_prompt_tokens + self.num_prefix_tokens)
            self.register_buffer('learned_token', torch.arange(self.vocab_size ,self.vocab_size + self.num_prompt_tokens, dtype=torch.long))
            if self.num_prefix_tokens > 0:
                self.register_buffer('learned_prefix', torch.arange(self.vocab_size + self.num_prompt_tokens, self.vocab_size + self.num_prompt_tokens + self.num_prefix_tokens,dtype=torch.long))

            if isinstance(self.transformer, peft.peft_model.PeftModel):
                self.transformer.model.model.embed_tokens = PartialOverrideEmbedding(self.transformer.model.model.embed_tokens, start_override= self.vocab_size , length_override=self.num_prompt_tokens + self.num_prefix_tokens)
                self.transformer.model.model_prepare_decoder_attention_mask = types.MethodType( _create_mask_func(self.num_prompt_tokens), self.transformer.model.model)
            else:
                self.transformer.model.embed_tokens = PartialOverrideEmbedding(self.transformer.model.embed_tokens, start_override= self.vocab_size , length_override=self.num_prompt_tokens + self.num_prefix_tokens)
                self.transformer.model._prepare_decoder_attention_mask = types.MethodType( _create_mask_func(self.num_prompt_tokens), self.transformer.model)

        if (d_model == output_dim) and (proj is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )

    def forward(self, x: TensorType):
        batch_size = x.shape[0]
        if self.prompt_tuning == True:
            learned_token = torch.tile(self.learned_token,(batch_size,1))
            x = torch.cat((x,learned_token),dim=1)
            if self.num_prefix_tokens > 0:
                prefix_token = torch.tile(self.learned_prefix,(batch_size,1))
                x = torch.cat((prefix_token,x),dim=1)
        attn_mask = (x != self.config.pad_token_id).long()

        if self.generate:
            generate_config = GenerationConfig(max_new_tokens=100,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True)
            out = self.transformer.generate(input_ids=x,
                    attention_mask=attn_mask,
                    generation_config=generate_config)
            gen_states = [states[-1] for states in out.hidden_states]
            out.hidden_states = torch.concat(gen_states,dim=1).unsqueeze(0)
            attn_mask = (out.sequences[:,:-1] != self.config.pad_token_id).long()
            print(self.tokenizer.decode(out.sequences[0]))
        else:
            out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        seq_len = out.hidden_states[-1].shape[1]
        tokens = (
            out.hidden_states[-1][:, torch.arange(seq_len) != self.pooler.cls_token_position, :]
            if type(self.pooler) == ClsPooler
            else out.hidden_states[-1]
        )

        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if ("LayerNorm" in n.split(".") or "layernorm" in n) else False

            if isinstance(self.transformer.model.embed_tokens,PartialOverrideEmbedding):
                self.transformer.model.embed_tokens.wte_override.weight.requires_grad = True
                #self.transformer.model.embed_tokens.reparam.requires_grad_(True)
            return

        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder.model, arch_dict[self.config.model_type]["config_names"]["layer_attr"])
        print(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model")
        embeddings = getattr(
            self.transformer.model, arch_dict[self.config.model_type]["config_names"]["token_embeddings_attr"])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if ("LayerNorm" in n.split(".") or "layernorm" in n) else False

        if isinstance(self.transformer.model.embed_tokens,PartialOverrideEmbedding):
            self.transformer.model.embed_tokens.wte_override.weight.requires_grad = True
            #self.transformer.model.embed_tokens.reparam.requires_grad_(True)

        self.transformer.lm_head.weight.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
