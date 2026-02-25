import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ditsb.discrete_flow import DiscreteFlowField, CategoricalFlowMatcher
from ditsb.discrete_loss import discrete_flow_matching_loss

def test_loss_convergence():
    print("="*60)
    print("Miniature Character Dataset Loss Convergence")
    print("="*60)
    
    # 1. Setup a tiny dataset (e.g., repeating phrase)
    phrase = "the quick brown fox jumps over the lazy dog."
    chars = sorted(list(set(phrase)))
    vocab_size = len(chars)
    char2idx = {c: i for i, c in enumerate(chars)}
    
    seq_len = 16
    data = []
    # Create overlapping sequences
    text_repeated = phrase * 10
    for i in range(len(text_repeated) - seq_len):
        seq = [char2idx[c] for c in text_repeated[i:i+seq_len]]
        data.append(seq)
        
    data_tensor = torch.tensor(data, dtype=torch.long)
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. Setup tiny model
    model = DiscreteFlowField(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=32,
        n_heads=2,
        n_layers=2,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Track losses
    epochs = 50
    initial_loss = None
    final_loss = None
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            x1 = batch[0]
            optimizer.zero_grad()
            
            # Using discrete flow matching loss
            loss = discrete_flow_matching_loss(model, x1, vocab_size)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        if epoch == 0:
            initial_loss = avg_loss
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} (Initial)")
        elif epoch == epochs - 1:
            final_loss = avg_loss
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} (Final)")
        else:
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f}")

    print("-" * 60)
    if final_loss < initial_loss * 0.7:
        print(f"[SUCCESS] Loss converged successfully. Final loss {final_loss:.4f} < {initial_loss * 0.7:.4f}")
    else:
        print(f"[FAILED] Loss did not sufficiently converge. Final loss {final_loss:.4f} >= {initial_loss * 0.7:.4f}")
        
    assert final_loss < initial_loss * 0.7, "Model failed to converge on tiny dataset."

if __name__ == "__main__":
    test_loss_convergence()
