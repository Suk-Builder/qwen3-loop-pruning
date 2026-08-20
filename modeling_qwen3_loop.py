"""
Qwen3 Loop Attention: 在最后一层Attention加循环检测门
只修改最后一层(layer_idx=27)，其余26层完全保持原样
核心修复：forward签名用past_key_value（单数），匹配父类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, Qwen3ForCausalLM, apply_rotary_pos_emb, repeat_kv
)

class Qwen3LoopAttention(Qwen3Attention):
    """
    继承标准Qwen3Attention，只在softmax后加循环检测调制。
    关键：forward签名和父类完全一致（past_key_value单数）。
    """
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        # 只在最后一层加loop_gate
        if layer_idx == config.num_hidden_layers - 1:
            # 标量参数：控制对循环的敏感度
            # 初始为0 -> sigmoid(0)=0.5 -> 训练初期不影响已有行为
            self.loop_gate = nn.Parameter(torch.tensor(0.0))
            print(f"[LoopAttention] Layer {layer_idx}: loop_gate initialized (0.0)")
        else:
            self.loop_gate = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,  # <-- 单数！和父类完全一致
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        bsz, q_len, _ = hidden_states.size()

        # === 标准投影（和原版完全一致）===
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 0.6B不走headwise/elementwise gate分支
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # === Attention计算（和原版完全一致）===
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.inv_sqrt_head_dim

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # === 循环检测门（唯一新增代码）===
        suppress = None
        if self.loop_gate is not None and q_len > 1 and self.training:
            # 取相邻query的attention分布
            curr = attn_weights[:, :, 1:, :].reshape(bsz, self.num_heads, q_len - 1, -1)
            prev = attn_weights[:, :, :-1, :].reshape(bsz, self.num_heads, q_len - 1, -1)
            # cosine similarity: 1=完全相似(循环), -1=完全不同
            similarity = F.cosine_similarity(curr, prev, dim=-1)  # [bsz, heads, q_len-1]
            # 均值作为循环风险指标
            loop_risk = similarity.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [bsz, heads, 1, 1]
            # loop_gate学为正数时：相似度高->suppress小->输出衰减->打破循环
            suppress = torch.sigmoid(-self.loop_gate * loop_risk)  # [bsz, heads, 1, 1]

        attn_output = torch.matmul(attn_weights, value_states)

        # 应用抑制（只在最后一层、训练时、q_len>1）
        if suppress is not None:
            attn_output = attn_output * suppress

        # === 后处理（和原版完全一致）===
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def load_model_with_loop(model_path, **kwargs):
    """
    加载Qwen3模型，并将最后一层Attention替换为LoopAttention。
    必须指定attn_implementation="eager"，否则拿不到显式attn_weights。
    """
    if "attn_implementation" not in kwargs:
        kwargs["attn_implementation"] = "eager"

    model = Qwen3ForCausalLM.from_pretrained(model_path, **kwargs)
    config = model.config
    last_idx = config.num_hidden_layers - 1

    # 替换最后一层
    old_attn = model.model.layers[last_idx].self_attn
    new_attn = Qwen3LoopAttention(config, layer_idx=last_idx)

    # 复制原权重（loop_gate是新参数，不会被复制，保持初始化值0）
    new_attn.load_state_dict(old_attn.state_dict(), strict=False)
    model.model.layers[last_idx].self_attn = new_attn

    print(f"[load_model_with_loop] Layer {last_idx} replaced. "
          f"New param: loop_gate = {new_attn.loop_gate.item():.4f}")

    return model
