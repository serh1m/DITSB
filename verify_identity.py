
import torch
import torch.nn as nn
from ditsb.discrete_flow import DiscreteFlowField
from ditsb.discrete_loss import discrete_flow_matching_loss

def check_identity():
    # Miniature model but Large Vocab
    V = 1000
    model = DiscreteFlowField(vocab_size=V, d_model=32, n_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Constant sequence "0 0 0 ..."
    # Batch size 32 to get enough samples
    x = torch.zeros(32, 10).long()
    
    print(f'Training on constant seqs (V={V}) with MASKED loss...')
    for i in range(1000): # More steps
        loss = discrete_flow_matching_loss(model, x, V)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 100 == 0:
            print(f'Step {i} Loss: {loss.item():.4f}')
        
    model.eval()
    t0 = torch.zeros(32) 
    with torch.no_grad():
        logits = model(x, t0) 
        probs = torch.softmax(logits, dim=-1)
        p_0 = probs[:, :, 0].mean().item()
        
    print(f'Probability of taking identity (0->0) with V={V}: {p_0:.4f}')
    
    if p_0 < 0.5:
        print("FAIL: Model failed to learn identity mapping for clean tokens!")
        exit(1)
    else:
        print("PASS: Model learned identity mapping.")
        exit(0)

if __name__ == "__main__":
    check_identity()
