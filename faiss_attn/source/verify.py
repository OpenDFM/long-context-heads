from typing import Optional
import torch
import torch.nn as nn

from utils import get_static_attention_kernel_Mm

# ==========================================
# Ground-truth code that simulates original Llama behavior
# ==========================================
@torch.no_grad()
def forward_simulation_llama(model, layer, hidden_states, dim: Optional[int]=None):
    from modeling_llama import apply_rotary_pos_emb, repeat_kv
    model_attn = model.model.layers[layer].self_attn
    bsz, q_len, _ = hidden_states.size()

    query_states = model_attn.q_proj(hidden_states)
    key_states = model_attn.k_proj(hidden_states)
    value_states = model_attn.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, model_attn.num_heads, model_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, model_attn.num_key_value_heads, model_attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, model_attn.num_key_value_heads, model_attn.head_dim).transpose(1, 2)

    position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    cos, sin = model_attn.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    key_states = repeat_kv(key_states, model_attn.num_key_value_groups)
    value_states = repeat_kv(value_states, model_attn.num_key_value_groups)

    if dim is not None:
        query_states = torch.stack(
            (query_states[..., :query_states.shape[-1]//2], query_states[..., query_states.shape[-1]//2:]), dim=-1
        ).transpose(2, 3)[:, :, dim]   # [bsz, num_heads, q_len, 2]
        key_states = torch.stack(
            (key_states[..., :key_states.shape[-1]//2], key_states[..., key_states.shape[-1]//2:]), dim=-1
        ).transpose(2, 3)[:, :, dim]   # [bsz, num_heads, q_len, 2]
    qk_results = torch.matmul(query_states, key_states.transpose(2, 3))   # [bsz, num_heads, q_len, q_len]
    return qk_results


def forward_simulation_qwen2(model, layer, hidden_states, dim: Optional[int]=None):
    from modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    model_attn = model.model.layers[layer].self_attn
    bsz, q_len, _ = hidden_states.size()

    query_states = model_attn.q_proj(hidden_states) - model_attn.q_proj.bias
    key_states = model_attn.k_proj(hidden_states) - model_attn.k_proj.bias
    value_states = model_attn.v_proj(hidden_states) - model_attn.v_proj.bias

    query_states = query_states.view(bsz, q_len, model_attn.num_heads, model_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, model_attn.num_key_value_heads, model_attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, model_attn.num_key_value_heads, model_attn.head_dim).transpose(1, 2)

    position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    cos, sin = model_attn.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    key_states = repeat_kv(key_states, model_attn.num_key_value_groups)
    value_states = repeat_kv(value_states, model_attn.num_key_value_groups)

    if dim is not None:
        query_states = torch.stack(
            (query_states[..., :query_states.shape[-1]//2], query_states[..., query_states.shape[-1]//2:]), dim=-1
        ).transpose(2, 3)[:, :, dim]   # [bsz, num_heads, q_len, 2]
        key_states = torch.stack(
            (key_states[..., :key_states.shape[-1]//2], key_states[..., key_states.shape[-1]//2:]), dim=-1
        ).transpose(2, 3)[:, :, dim]   # [bsz, num_heads, q_len, 2]
    qk_results = torch.matmul(query_states, key_states.transpose(2, 3))   # [bsz, num_heads, q_len, q_len]
    return qk_results

# ==========================================
# Verification main routine
# ==========================================

@torch.no_grad()
def run_verification_llama():
    from modeling_llama import LlamaForCausalLM
    import torch

    MODEL_DIR = "~/PretrainedModels/Llama-3.2-3B-Instruct"
    model = LlamaForCausalLM.from_pretrained(
        MODEL_DIR, 
        local_files_only=True, 
        device_map="cuda:0", 
        torch_dtype=torch.float32,
        attn_implementation="flash_attention_2",
    )
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    model.eval()

    # Test parameters
    TEST_LAYER = 14
    TEST_HEAD = 22
    TEST_Q_POS = 256
    TEST_K_POS = 5
    
    print(f"--- Start verification Head={TEST_HEAD}, Dist={TEST_K_POS - TEST_Q_POS} ---")
    # Random input vector
    hidden_state = torch.randn(1, 10000, model.config.hidden_size, device="cuda:0", dtype=torch.float32)
    x_q = hidden_state[0, TEST_Q_POS, :].unsqueeze(0)  # (1, hidden_size)
    x_k = hidden_state[0, TEST_K_POS, :].unsqueeze(0)  # (1, hidden_size)

    all_pair_matrix = []
    # 1. Component-wise verification
    for test_pair in range(model.config.hidden_size // model.config.num_attention_heads // 2):
        M_matrix = get_static_attention_kernel_Mm(model, TEST_LAYER, TEST_HEAD, TEST_K_POS - TEST_Q_POS, test_pair)
        score_A_attn = (x_q @ M_matrix @ x_k.T).squeeze().item()
        # print(f"Path A (Static Kernel Score): {score_A_attn:.8f}")
        score_B = forward_simulation_llama(model, TEST_LAYER, hidden_state, dim=test_pair)[0]
        score_B_attn = score_B[TEST_HEAD, TEST_Q_POS, TEST_K_POS].item()
        # print(f"Path B (Forward Sim Score)  : {score_B_attn:.8f}")
        diff = abs(score_A_attn - score_B_attn)
        # print(f"Difference: {diff:.8e}")
        if diff < 1e-3:
            pass
        else:
            raise ValueError(
                "❌ Verification failed! Component results are inconsistent."
                f" Layer={TEST_LAYER}, Head={TEST_HEAD}, Pair={test_pair}, "
                f"ScoreA={score_A_attn:.8f}, ScoreB={score_B_attn:.8f}, Diff={diff:.8e}"
            )
        all_pair_matrix.append(M_matrix)

    # 2. Sum verification
    # Path A: static kernel method
    M_matrix = torch.stack(all_pair_matrix, dim=0).sum(dim=0)
    score_A_attn = (x_q @ M_matrix @ x_k.T).squeeze().item()
    # print(f"Path A (Static Kernel Score): {score_A_attn:.8f}")
    # Path B: simulated forward pass
    score_B = forward_simulation_llama(model, TEST_LAYER, hidden_state)[0]
    score_B_attn = score_B[TEST_HEAD, TEST_Q_POS, TEST_K_POS].item()
    # print(f"Path B (Forward Sim Score)  : {score_B_attn:.8f}")
    # Compare
    diff = abs(score_A_attn - score_B_attn)
    # print(f"Difference: {diff:.8e}")
    
    if diff < 1e-3:
        print("✅ Verification passed! Static kernel computation logic is correct.")
    else:
            raise ValueError(
                "❌ Verification failed! Component results are inconsistent."
                f" Layer={TEST_LAYER}, Head={TEST_HEAD}, Pair={test_pair}, "
                f"ScoreA={score_A_attn:.8f}, ScoreB={score_B_attn:.8f}, Diff={diff:.8e}"
            )


@torch.no_grad()
def run_verification_qwen2():
    from modeling_qwen2 import Qwen2ForCausalLM
    from transformers import AutoConfig
    import torch

    MODEL_DIR = "/public/share/model/Qwen2.5-3B-Instruct"
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    config.max_position_embeddings = 131072
    config.rope_scaling = {
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "type": "yarn"
    }
    model = Qwen2ForCausalLM.from_pretrained(
        MODEL_DIR, 
        config=config,
        local_files_only=True, 
        device_map="cuda:0", 
        torch_dtype=torch.float32,
        attn_implementation="flash_attention_2",
    )
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    model.eval()

    # Test parameters
    TEST_LAYER = 14
    TEST_HEAD = 12
    TEST_Q_POS = 9986
    TEST_K_POS = 5
    
    print(f"--- Start verification Head={TEST_HEAD}, Dist={TEST_K_POS - TEST_Q_POS} ---")
    # Random input vector
    hidden_state = torch.randn(1, 10000, model.config.hidden_size, device="cuda:0", dtype=torch.float32)
    x_q = hidden_state[0, TEST_Q_POS, :].unsqueeze(0)  # (1, hidden_size)
    x_k = hidden_state[0, TEST_K_POS, :].unsqueeze(0)  # (1, hidden_size)

    all_pair_matrix = []
    # 1. Component-wise verification
    for test_pair in range(model.config.hidden_size // model.config.num_attention_heads // 2):
        M_matrix = get_static_attention_kernel_Mm(model, TEST_LAYER, TEST_HEAD, TEST_K_POS - TEST_Q_POS, test_pair)
        score_A_attn = (x_q @ M_matrix @ x_k.T).squeeze().item()
        # print(f"Path A (Static Kernel Score): {score_A_attn:.8f}")
        score_B = forward_simulation_qwen2(model, TEST_LAYER, hidden_state, dim=test_pair)[0]
        score_B_attn = score_B[TEST_HEAD, TEST_Q_POS, TEST_K_POS].item()
        # print(f"Path B (Forward Sim Score)  : {score_B_attn:.8f}")
        diff = abs(score_A_attn - score_B_attn)
        # print(f"Difference: {diff:.8e}")
        if diff < 1e-3:
            pass
        else:
            raise ValueError(
                "❌ Verification failed! Component results are inconsistent."
                f" Layer={TEST_LAYER}, Head={TEST_HEAD}, Pair={test_pair}, "
                f"ScoreA={score_A_attn:.8f}, ScoreB={score_B_attn:.8f}, Diff={diff:.8e}"
            )
        all_pair_matrix.append(M_matrix)

    # 2. Sum verification
    # Path A: static kernel method
    M_matrix = torch.stack(all_pair_matrix, dim=0).sum(dim=0)
    score_A_attn = (x_q @ M_matrix @ x_k.T).squeeze().item()
    # print(f"Path A (Static Kernel Score): {score_A_attn:.8f}")
    # Path B: simulated forward pass
    score_B = forward_simulation_qwen2(model, TEST_LAYER, hidden_state)[0]
    score_B_attn = score_B[TEST_HEAD, TEST_Q_POS, TEST_K_POS].item()
    # print(f"Path B (Forward Sim Score)  : {score_B_attn:.8f}")
    # Compare
    diff = abs(score_A_attn - score_B_attn)
    # print(f"Difference: {diff:.8e}")
    
    if diff < 1e-3:
        print("✅ Verification passed! Static kernel computation logic is correct.")
    else:
            raise ValueError(
                "❌ Verification failed! Component results are inconsistent."
                f" Layer={TEST_LAYER}, Head={TEST_HEAD}, Pair={test_pair}, "
                f"ScoreA={score_A_attn:.8f}, ScoreB={score_B_attn:.8f}, Diff={diff:.8e}"
            )


if __name__ == "__main__":
    run_verification_qwen2()
    pass