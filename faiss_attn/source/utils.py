import glob

import numpy as np

def load_context(fpath="PaulGrahamEssays/*.txt", ctx_len=100000):
    context = ""
    for file in glob.glob(fpath):
        with open(file, 'r') as f: 
            context += f.read()
    LLAMA_CHAR_TO_TOKEN_RATIO = 3.66
    context = context[: int(ctx_len * LLAMA_CHAR_TO_TOKEN_RATIO)]
    return context

def insert_needle(context, needle, depth):
    context = context.split(".")
    c_len = len(context)
    needle_place = int(depth * c_len)
    context = ".".join(context[:needle_place]) + "." + needle + ".".join(context[needle_place:])
    return context


import glob
import os
from typing import List, Union
import transformers
import torch
import math

class NIAHBuilder:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, haystack_dir: str):
        """
        Initialize NIAHBuilder

        Args:
            tokenizer: Tokenizer instance used for encoding and decoding
            haystack_dir: Directory path containing background text files (.txt)
        """
        self.tokenizer = tokenizer
        self.haystack_dir = haystack_dir
        
        # Preload all background text for better efficiency
        self.full_haystack_text = self._read_context_files()

        # Try to automatically get the token id for '.', used for finding insertion points
        # Note: different tokenizers may encode '.' differently; this is a simple heuristic
        # Usually '.' maps to one id, but it may include BOS or other tokens
        dummy_enc = self.tokenizer.encode(".", add_special_tokens=False)
        if len(dummy_enc) > 0:
            # Use the last token as the period token (ignoring possible BOS tokens)
            self.period_token_id = dummy_enc[-1]
        else:
            # Fallback: if not found, set to None; later logic won't strictly depend on periods
            self.period_token_id = None
            print("Warning: Could not determine period token ID. Needle insertion will simply be at the exact position.")

    def _read_context_files(self) -> str:
        """Read and concatenate all txt file contents from haystack_dir."""
        context = ""
        files = glob.glob(os.path.join(self.haystack_dir, "*.txt"))
        if not files:
            raise ValueError(f"No .txt files found in {self.haystack_dir}")
        
        # Simple sorting for deterministic behavior
        files.sort()
        
        for file in files:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                context += f.read()
        return context

    def _ensure_context_length(self, target_len: int) -> str:
        """Ensure the background text is long enough; repeat if needed."""
        context = self.full_haystack_text
        current_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        
        while len(current_tokens) < target_len:
            context += " " + self.full_haystack_text # Add a space as separator
            current_tokens = self.tokenizer.encode(context, add_special_tokens=False)
            
        return context

    def generate_context(self, context_len: int, depth_ratio: float, needle_str: str, question_str: str, **chat_template_kwargs) -> str:
        """
        Generate a complete prompt containing the needle.

        Args:
            context_len (int): Target context length (number of tokens)
            depth_ratio (float): Insertion position ratio (0.0 - 1.0)
            needle_str (str): Needle text to insert
            question_str (str): Question text to ask

        Returns:
            str: Final constructed prompt string before tokenization
        """
        # 1. Prepare sufficiently long background text
        raw_context = self._ensure_context_length(context_len)
        
        # 2. Encode needle and context
        needle_tokens = self.tokenizer.encode(needle_str, add_special_tokens=False)
        context_tokens = self.tokenizer.encode(raw_context, add_special_tokens=False)

        # 3. Compute the actual available space for context
        # Reserve some space for the system prompt, chat template structure, and question
        # For simplicity, we assume final output length is mainly controlled by context_len
        # Original logic: context_len -= final_context_length_buffer (e.g., 200)
        # Here we reserve 200 tokens for the template wrapper and question
        buffer_len = 200 
        available_context_len = max(0, context_len - buffer_len)

        # Truncate context to make room for the needle
        if len(context_tokens) + len(needle_tokens) > available_context_len:
            context_tokens = context_tokens[:available_context_len - len(needle_tokens)]

        # 4. Insert the needle
        if depth_ratio >= 1.0:
            # Insert at the end
            new_context_tokens = context_tokens + needle_tokens
        elif depth_ratio <= 0.0:
            # Insert at the beginning
            new_context_tokens = needle_tokens + context_tokens
        else:
            # Compute insertion point
            insertion_point = int(len(context_tokens) * depth_ratio)

            # Insert near the nearest period to preserve sentence integrity
            if self.period_token_id is not None:
                # Search backward until a period is found
                # Use a search limit to avoid scanning all the way back to the start
                search_limit = min(insertion_point, 500) 
                found_period = False
                original_point = insertion_point
                
                for i in range(search_limit):
                    if context_tokens[insertion_point - 1] == self.period_token_id:
                        found_period = True
                        break
                    insertion_point -= 1
                
                # If no period is found, fall back to the original insertion point
                if not found_period:
                    insertion_point = original_point

            # Concatenate
            new_context_tokens = (
                context_tokens[:insertion_point] + 
                needle_tokens + 
                context_tokens[insertion_point:]
            )

        # 5. Decode back to string
        final_context_text = self.tokenizer.decode(new_context_tokens)

        # 6. Build the final prompt with chat template
        # Build a prompt structure similar to the original code: <book>{context}</book> ...
        user_content = f"<book>{final_context_text}</book>\nBased on the content of the book, Question: {question_str}\nAnswer:"
        
        messages = [
            {"role": "user", "content": user_content}
        ]

        # Generate the final string using apply_chat_template
        try:
            final_prompt = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs
            )
        except ValueError:            # If the model does not support apply_chat_template, directly concatenate
            final_prompt = final_context_text

        return final_prompt


@torch.no_grad()
def get_static_attention_kernel_Mm(
    model, 
    layer_idx: int, 
    head_idx: int, 
    distance_n: int, 
    m_pair_idx: int
) -> torch.Tensor:
    """
    Fully based on modeling_llama.py logic, compute the kernel matrix M_m(n)
    for a specific layer, head, and frequency pair from static parameters.
    
    Args:
        model: LlamaForCausalLM or LlamaModel instance
        layer_idx: Target layer index
        head_idx: Target query head index
        distance_n: Relative distance (j - i)
        m_pair_idx: Frequency pair index, in [0, head_dim/2 - 1].
                    Corresponding physical dimensions are [m, m + head_dim/2]
    
    Returns:
        M_m: Kernel matrix with shape (hidden_size, hidden_size)
    """
    # 1. Get base configuration
    config = model.config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)
    
    # Validate m_pair_idx range
    if m_pair_idx >= head_dim // 2:
        raise ValueError(f"m_pair_idx {m_pair_idx} out of bounds for head_dim {head_dim} (max {head_dim//2 - 1})")

    # 2. Get the Attention module for this layer
    # Note: model may be LlamaForCausalLM (wrapped by model.model) or LlamaModel
    if hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers
        
    layer = layers[layer_idx]
    attn = layer.self_attn
    
    # 3. Extract specific rows of W_Q and W_K
    # Llama Linear weights are stored with shape (out_features, in_features)
    # W_Q_all shape: (num_heads * head_dim, hidden_size)
    W_Q_all = attn.q_proj.weight
    W_K_all = attn.k_proj.weight
    
    # 3.1 Extract W_Q block for the current head
    q_start = head_idx * head_dim
    q_end = (head_idx + 1) * head_dim
    W_Q_head = W_Q_all[q_start:q_end, :] # (head_dim, hidden_size)
    
    # 3.2 Handle GQA (Grouped Query Attention)
    # Determine the corresponding key head index for the current query head
    num_k_heads = config.num_key_value_heads
    num_q_per_k = num_heads // num_k_heads # Number of query heads per KV head
    k_head_idx = head_idx // num_q_per_k
    
    k_start = k_head_idx * head_dim
    k_end = (k_head_idx + 1) * head_dim
    W_K_head = W_K_all[k_start:k_end, :] # (head_dim, hidden_size)

    # 4. Extract projection vectors for the two dimensions in the pair
    # According to rotate_half logic in modeling_llama.py:
    # x1 = x[..., :half], x2 = x[..., half:]
    # Rotation occurs between x[i] and x[i + half]
    half_dim = head_dim // 2
    idx1 = m_pair_idx
    idx2 = m_pair_idx + half_dim
    
    # Extract projection weight row vectors for these two dimensions
    # w_q_pair shape: (2, hidden_size)
    w_q_pair = torch.stack([W_Q_head[idx1], W_Q_head[idx2]], dim=0)
    w_k_pair = torch.stack([W_K_head[idx1], W_K_head[idx2]], dim=0)

    # 5. Compute rotation matrix R
    # Reuse inv_freq computed by the model to ensure frequency consistency
    # inv_freq shape: (head_dim / 2,)
    device = w_q_pair.device
    dtype = w_q_pair.dtype

    rotary_emb = None
    try:
        rotary_emb = attn.rotary_emb
    except AttributeError:
        try:
            # Compatibility with newer versions
            rotary_emb = model.model.rotary_emb
        except AttributeError:
            rotary_emb = model.rotary_emb
    inv_freq = rotary_emb.inv_freq.to(device)
    
    # Get theta frequency for the current pair
    freq = inv_freq[m_pair_idx]
    
    # Compute rotation angle
    angle = distance_n * freq
    
    # Get scaling factor (supports variants like YaRN)
    try:
        scale = rotary_emb.attention_scaling
    except AttributeError:
        scale = 1.0
    
    # Compute cos and sin
    # Note: LlamaRoPE implementation uses cos * scale and sin * scale
    c = torch.cos(angle) * scale
    s = torch.sin(angle) * scale
    
    # 6. Build rotated kernel R_n
    # Recall rotate_half(x): [-x2, x1]
    # Q_rot = Q * cos + rotate_half(Q) * sin
    # For our pair [q1, q2]:
    # q1_rot = q1*c - q2*s
    # q2_rot = q2*c + q1*s
    # In matrix form: (q1_rot, q2_rot)^T = R * (q1, q2)^T
    # R = [[c, -s], 
    #      [s,  c]]
    
    R = torch.tensor([
        [c, -s],
        [s,  c]
    ], device=device, dtype=dtype)
    
    # 7. Compute M_m(n) = W_Q_pair^T @ R @ W_K_pair
    # Shapes: (hidden, 2) @ (2, 2) @ (2, hidden) -> (hidden, hidden)
    M_m = w_q_pair.T @ R @ w_k_pair
    
    return M_m
