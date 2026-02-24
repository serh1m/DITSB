import yaml
import math
import argparse

def FLOP_calculation_for_DITSB(config):
    """
    Computes theoretical FLOPs for the DITSB continuous LLM.
    Formula assumes essentially LLaMA scaling:
    C = 6 * N * D (where N = parameter count, D = training tokens)
    Because ODE matching is purely a forward regression on vector fields
    and adjoint backprop takes ~forward cost,
    FLOPs are roughly equivalent to causal modeling!
    """
    cmd = config['model']
    d_model = cmd['d_model']
    n_layers = cmd['n_layers']
    vocab = cmd['vocab_size']
    
    # 1. Parameter Estimation 
    # attention: 4 * d_model^2
    # ffn: 8 * d_model^2 (assuming SwiGLU expansion 8/3)
    # per layer ~ 12 * d_model^2
    layer_params = 12 * (d_model ** 2) * n_layers
    embedding_params = vocab * d_model
    total_params = layer_params + embedding_params
    
    N = total_params
    D = config['data']['tokens_per_epoch']
    
    # Standard compute bounds
    FLOPs = 6 * N * D
    return N, FLOPs

def hardware_time_prediction(config, total_flops):
    """
    Predicts wall-clock time based on hardware specs.
    """
    hw = config['hardware']
    nodes = hw['num_nodes']
    gpus_per_node = hw['gpus_per_node']
    total_gpus = nodes * gpus_per_node
    
    # For H100 bf16, real-world MFU is around ~45-50%
    if "H100" in hw['gpu_type']:
        theoretical_tflops_per_gpu = 989
    elif "A100" in hw['gpu_type']:
         theoretical_tflops_per_gpu = 312
    else:
         theoretical_tflops_per_gpu = 100
         
    practical_tflops = theoretical_tflops_per_gpu * hw['utilization_factor']
    
    total_practical_tflops = practical_tflops * total_gpus
    flops_per_second = total_practical_tflops * (10**12)
    
    seconds = total_flops / flops_per_second
    days = seconds / (24 * 3600)
    
    return days, total_gpus

def scaling_law_loss_prediction(N, D):
    """
    Uses Chinchilla-like scaling laws adapted for Continuous Rate matching.
    L(N, D) = A / N^alpha + B / D^beta + E
    We approximate the equivalent perplexity and continuous MSE loss.
    """
    # Parameters scaled down from Chinchilla for categorical MSE flow
    A = 150.0  
    B = 200.0  
    alpha = 0.34
    beta = 0.28
    E = 0.05 # Irreducible entropy of the flow dataset
    
    mse_loss = (A / (N ** alpha)) + (B / (D ** beta)) + E
    approx_ppl = math.exp(mse_loss * 10) # rough projection multiplier
    
    return mse_loss, approx_ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="llm_training/config_7b.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    print("="*60)
    print(f" DITSB-v2 LLM Prediction Engine ")
    print("="*60)
    
    N, TFLOPs_total = FLOP_calculation_for_DITSB(config)
    print(f"[Model Scaling]")
    print(f"- Estimated Parameters: {N / 1e9:.2f} Billion")
    print(f"- Training Tokens: {config['data']['tokens_per_epoch'] / 1e9:.2f} Billion")
    print(f"- Total Compute: {TFLOPs_total / 1e21:.3f} ZettaFLOPs\n")
    
    days, total_gpus = hardware_time_prediction(config, TFLOPs_total)
    print(f"[Hardware & Time Prediction]")
    print(f"- Setup: {total_gpus}x {config['hardware']['gpu_type']} GPUs (MFU: {config['hardware']['utilization_factor']*100}%)")
    print(f"- Predicted Wall-Clock Time: {days:.2f} Days\n")
    
    mse_loss, ppl = scaling_law_loss_prediction(N, config['data']['tokens_per_epoch'])
    print(f"[Training Effect Prediction]")
    print(f"- Final Predicted CTMC Loss: {mse_loss:.4f}")
    print(f"- Final Predicted PPL: {ppl:.2f}")
    
    print("="*60)
    print("NOTE: Continuous Flow guarantees O(1) memory, allowing sequence length ")
    print(f"      {config['model']['max_seq_len']} without intermediate activation explosion.")
    print("="*60)

if __name__ == "__main__":
    main()
