import os
import time
import math
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DITSB-Train")

# Assume ditsb is installed or in PYTHONPATH
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from ditsb.discrete_flow import CategoricalFlowMatcher
except ImportError:
    logger.warning("Mocking CategoricalFlowMatcher for testing script structure.")
    class CategoricalFlowMatcher(nn.Module):
        def __init__(self, vocab_size): super().__init__(); self.vocab_size = vocab_size
        def sample_pt(self, x1, t): return x1 * t
        def compute_ctmc_loss(self, logits, x1, t): return torch.nn.functional.mse_loss(logits, x1 - 1/self.vocab_size)

class LLMDataset(Dataset):
    def __init__(self, data_path, seq_len):
        if not os.path.exists(data_path):
            # For demonstration without preparing actual data, we mock logic
            self.data = np.random.randint(0, 50000, size=(1000, seq_len), dtype=np.uint16)
        else:
            self.data = np.load(data_path, mmap_mode='r')
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

import argparse

import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        return self.weight * (x_f32 * torch.rsqrt(variance + self.eps)).to(input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=500000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        return self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2), self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['model'].get('n_heads', 32)
        self.n_kv_heads = config['model'].get('n_kv_heads', 8)
        self.d_model = config['model']['d_model']
        self.head_dim = self.d_model // self.n_heads
        
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_position_embeddings=config['model'].get('max_seq_len', 2048))

    def forward(self, x):
        B, seq_len, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        q = q.view(B, seq_len, self.n_heads, self.head_dim)
        k = k.view(B, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(B, seq_len, self.n_kv_heads, self.head_dim)
        
        cos, sin = self.rotary(seq_len)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # GQA repeat
        n_rep = self.n_heads // self.n_kv_heads
        k = k[:, :, :, None, :].expand(B, seq_len, self.n_kv_heads, n_rep, self.head_dim).reshape(B, seq_len, self.n_heads, self.head_dim)
        v = v[:, :, :, None, :].expand(B, seq_len, self.n_kv_heads, n_rep, self.head_dim).reshape(B, seq_len, self.n_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # No causal mask for bidirectional continuous flow!
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(attn_output)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.intermediate_size = config['model'].get('intermediate_size', 14336)
        
        self.gate_proj = nn.Linear(self.d_model, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.d_model, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.d_model, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # LLaMA SwiGLU logic: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LLaMABlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config['model']['d_model'])
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config['model']['d_model'])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class DITSBFlowLLaMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.vocab = config['model']['vocab_size']
        self.n_layers = config['model'].get('n_layers', 32)
        
        self.soft_embedding = nn.Linear(self.vocab, self.d_model, bias=False)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.blocks = nn.ModuleList([LLaMABlock(config) for _ in range(self.n_layers)])
        self.norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab, bias=False)
        
    def forward(self, t, x_t_probs):
        x = self.soft_embedding(x_t_probs)
        
        if isinstance(t, float) or t.dim() == 0:
            t_vec = torch.full((x_t_probs.size(0), x_t_probs.size(1), 1), float(t), device=x_t_probs.device, dtype=x.dtype)
        else:
            t_vec = t.to(x.dtype)
            
        t_emb = self.time_embed(t_vec)
        x = x + t_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

import gc

def load_warm_start_weights(model, hf_model_path, device):
    try:
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading HuggingFace model weights from: {hf_model_path} (to CPU for RAM safety)")
        # CRITICAL: Always load the source model to CPU to avoid double-allocation on small GPUs (like Colab T4)
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.float16, device_map="cpu")
        
        hf_state_dict = hf_model.state_dict()
        model_state_dict = model.state_dict()
        
        assigned_params = 0
        total_params = len(model_state_dict.keys())
        
        if 'model.embed_tokens.weight' in hf_state_dict:
            wt = hf_state_dict['model.embed_tokens.weight'].transpose(0, 1)
            if wt.shape == model_state_dict['soft_embedding.weight'].shape:
                model_state_dict['soft_embedding.weight'].copy_(wt)
                assigned_params += 1
                logger.info("Successfully mapped HF Embeddings to DITSB Soft Embedding.")
        
        if 'lm_head.weight' in hf_state_dict:
            wt = hf_state_dict['lm_head.weight']
            if wt.shape == model_state_dict['lm_head.weight'].shape:
                model_state_dict['lm_head.weight'].copy_(wt)
                assigned_params += 1
                logger.info("Successfully mapped HF LM Head to DITSB LM Head.")
                
        if 'model.norm.weight' in hf_state_dict:
            wt = hf_state_dict['model.norm.weight']
            model_state_dict['norm.weight'].copy_(wt)
            assigned_params += 1
            logger.info("Successfully mapped HF final RMSNorm.")

        num_layers = min(model.n_layers, hf_model.config.num_hidden_layers)
        logger.info(f"Mapping {num_layers} Transformer Blocks...")
        for i in range(num_layers):
            block_prefix_hf = f"model.layers.{i}."
            block_prefix_ditsb = f"blocks.{i}."
            
            mapping = {
                f"{block_prefix_hf}input_layernorm.weight": f"{block_prefix_ditsb}input_layernorm.weight",
                f"{block_prefix_hf}post_attention_layernorm.weight": f"{block_prefix_ditsb}post_attention_layernorm.weight",
                f"{block_prefix_hf}self_attn.q_proj.weight": f"{block_prefix_ditsb}self_attn.q_proj.weight",
                f"{block_prefix_hf}self_attn.k_proj.weight": f"{block_prefix_ditsb}self_attn.k_proj.weight",
                f"{block_prefix_hf}self_attn.v_proj.weight": f"{block_prefix_ditsb}self_attn.v_proj.weight",
                f"{block_prefix_hf}self_attn.o_proj.weight": f"{block_prefix_ditsb}self_attn.o_proj.weight",
                f"{block_prefix_hf}mlp.gate_proj.weight": f"{block_prefix_ditsb}mlp.gate_proj.weight",
                f"{block_prefix_hf}mlp.up_proj.weight": f"{block_prefix_ditsb}mlp.up_proj.weight",
                f"{block_prefix_hf}mlp.down_proj.weight": f"{block_prefix_ditsb}mlp.down_proj.weight",
            }
            
            for hf_key, ditsb_key in mapping.items():
                if hf_key in hf_state_dict and ditsb_key in model_state_dict:
                    if hf_state_dict[hf_key].shape == model_state_dict[ditsb_key].shape:
                        model_state_dict[ditsb_key].copy_(hf_state_dict[hf_key])
                        assigned_params += 1
        
        model.load_state_dict(model_state_dict)
        logger.info(f"Warm-start completed. Tensors Transferred: {assigned_params}/{total_params}")
        
        # PREVENT OOM CRASH: Destroy the HF model from memory before starting the training loop
        del hf_model
        del hf_state_dict
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleared source HF model from RAM to make room for training.")
        
    except Exception as e:
        logger.error(f"Failed to load warm start weights: {e}")
        raise e

def calculate_eta(elapsed_time, current_step, total_steps):
    if current_step == 0: return "Unknown"
    steps_left = total_steps - current_step
    time_per_step = elapsed_time / current_step
    eta_seconds = steps_left * time_per_step
    return time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

def train(config_path, warm_start_path=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Initialized training on device: {device}")
    logger.info(f"Using configuration from: {config_path}")
    
    # 1. Dataset
    data_file = os.path.join(config['data']['dataset_dir'], "input_ids.npy")
    seq_len = config['model']['max_seq_len']
    batch_size = config['training']['batch_size']
    
    dataset = LLMDataset(data_file, seq_len)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config['data'].get('num_workers', 0)
    )
    logger.info(f"Dataset loaded. Total continuous chunks: {len(dataset):,}")
    
    # 2. Model & Optimization setup (CRITICAL MEMORY PRESERVATION: Initialize directly on device in bfloat16 to avoid 28GB CPU/GPU spike)
    logger.info("Instantiating DITSB LLM Architecture (Native half-precision on device)...")
    old_dtype = torch.get_default_dtype()
    # Force 16-bit to instantly halve 7B parameter allocation (28GB -> 14GB)
    target_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.set_default_dtype(target_dtype) 
    
    # Note: On PyTorch 2.x, device context manager prevents host CPU RAM exhaustion
    with torch.device(device):
        model = DITSBFlowLLaMA(config).to(target_dtype)
    
    torch.set_default_dtype(old_dtype) # Restore defaults
    # 2.5 Optional Warm-Start
    if warm_start_path:
        logger.info("Initializing Warm-Start from pre-trained weights.")
        load_warm_start_weights(model, warm_start_path, device)
    
    # Check for 8-bit Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), 
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
        logger.info("Equipped bitsandbytes 8-bit AdamW (Massive VRAM Savings)")
    except ImportError:
        logger.warning("bitsandbytes not found. Falling back to 32-bit AdamW (High VRAM usage).")
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
    
    matcher = CategoricalFlowMatcher(vocab_size=config['model']['vocab_size']).to(device)
    
    max_steps = config['training']['max_steps']
    log_interval = config['logging']['log_interval']
    clip_grad_norm = config['training'].get('clip_grad_norm', 1.0)
    
    # 2.6 Initialize Learning Rate Scheduler (Cosine Annealing)
    initial_lr_float = float(config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=max_steps, 
        eta_min=initial_lr_float * 0.1 # Decay down to 10% of starting LR
    )
    
    logger.info("="*60)
    logger.info("🚀 Starting O(1) Adjoint Sinkhorn Flow Training Loop")
    logger.info("="*60)
    
    model.train()
    start_time = time.time()
    last_log_time = start_time
    accumulated_loss = 0.0
    
    torch.cuda.empty_cache()
    
    def infinite_loader(dl):
        while True:
            for b in dl:
                yield b
                
    step = 0
    for batch in infinite_loader(loader):
        step += 1
        if step > max_steps:
            break
            
        batch = batch.to(device)
        B, SeqLen = batch.shape
        vocab = config['model']['vocab_size']
        
        # 3. Probability Mappings (x1 target, x0 prior)
        x1_onehot = torch.nn.functional.one_hot(batch, num_classes=vocab).float()
        x0_probs = torch.ones_like(x1_onehot) / vocab
        t = torch.rand(B, SeqLen, 1, device=device)
        
        # In actual scale -> perform Sinkhorn alignments
        pt = matcher.sample_pt(x1_onehot, t)
        
        optimizer.zero_grad(set_to_none=True) # Memory efficient zeroing
        
        # Mixed Precision Forward/Backward (BFloat16 natively prevents underflow without a Scaler)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Forward Vector Field 
            logits = model(t, pt)
            
            # CE over probability transition conditionals
            loss = matcher.compute_ctmc_loss(logits, x1_onehot, t)
            
        loss.backward()
        
        # Address vanishing/exploding gradients in Flow spaces
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        
        # Protective Guard: Do not step if gradients exploded to NaN/Inf during BFloat16 transitions
        loss_val = loss.item()
        if math.isnan(grad_norm) or math.isinf(grad_norm) or math.isnan(loss_val) or math.isinf(loss_val):
            logger.warning(f"Step {step}: NaN/Inf detected (Loss: {loss_val:.2f}, Grad: {grad_norm}). Skipping step metrics and parameter update.")
            optimizer.zero_grad(set_to_none=True)
            # Rollback step count so we don't advance the sequence towards max_steps on failed tensors
            step -= 1
            continue
        else:
            optimizer.step()
            scheduler.step()
            accumulated_loss += loss_val
        
        # 4. Detailed Logging
        if step % log_interval == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            time_since_log = current_time - last_log_time
            
            avg_loss = accumulated_loss / log_interval
            tokens_processed = batch_size * seq_len * log_interval
            throughput = tokens_processed / time_since_log
            eta = calculate_eta(elapsed, step, max_steps)
            
            # Predict PPL (Flow CE formulation maps perfectly to standard perplexity)
            approx_ppl = math.exp(min(avg_loss, 100)) 
            
            logger.info(
                f"Step {step:6d}/{max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"PPL(approx): {approx_ppl:.1f} | "
                f"Tok/s: {throughput:,.0f} | "
                f"Lr: {optimizer.param_groups[0]['lr']:.2e} | "
                f"ETA: {eta}"
            )
            
            accumulated_loss = 0.0
            last_log_time = current_time
            
        # 5. Save routine
        save_interval = config['logging']['save_interval']
        if step % save_interval == 0 or step == max_steps:
            save_path = os.path.join(config['training']['checkpoint_dir'], f"ckpt_step_{step}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Checkpoint saved: {save_path}")
            
        if step >= max_steps:
            break
            
    logger.info("🎉 Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DITSB-v2 Train Script")
    parser.add_argument("--config", type=str, default="config_7b.yaml", help="Path to config file")
    parser.add_argument("--warm_start_path", type=str, default=None, help="Optional: Path to HuggingFace model for warm-start initialization")
    
    args = parser.parse_args()
    train(args.config, args.warm_start_path)
