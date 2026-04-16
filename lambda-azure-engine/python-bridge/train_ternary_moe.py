import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os

class MoEConfig:
    vocab_size = 32000
    d_model = 512
    n_layers = 6
    n_heads = 8
    n_experts = 8
    expert_hidden_size = 2048
    top_k = 1

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.expert_hidden_size, bias=False)
        self.w2 = nn.Linear(config.expert_hidden_size, config.d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        self.top_k = config.top_k

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)
        
        router_logits = self.router(flat_x)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = torch.softmax(routing_weights, dim=-1)
        
        final_output = torch.zeros_like(flat_x)
        
        # Simple routing for top-1
        expert_mask = torch.nn.functional.one_hot(selected_experts[:, 0], num_classes=len(self.experts))
        
        for i, expert in enumerate(self.experts):
            idx = torch.where(expert_mask[:, i])[0]
            if len(idx) > 0:
                expert_input = flat_x[idx]
                expert_output = expert(expert_input)
                final_output[idx] += routing_weights[idx, 0].unsqueeze(-1) * expert_output
                
        return final_output.view(batch_size, seq_len, d_model)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MoELayer(config)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class TernaryMoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx_seqs):
        x = self.embed(idx_seqs)
        
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        return self.lm_head(x)

def train():
    print("Initializing model...")
    config = MoEConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TernaryMoEModel(config).to(device)
    
    # Standard FP16 training stub
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # In a real environment, download TinyStories here
    # For now, train on dummy data to ensure it runs
    print("Training on dummy data...")
    for step in range(10):
        dummy_input = torch.randint(0, config.vocab_size, (4, 128)).to(device)
        dummy_target = dummy_input.clone() # shift omitted for simplicity
        
        logits = model(dummy_input)
        loss = criterion(logits.view(-1, config.vocab_size), dummy_target.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 2 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            
    # Save checkpoint
    os.makedirs('/content/drive/MyDrive/LAE', exist_ok=True)
    out_path = '/content/drive/MyDrive/LAE/moe_fp16.pt'
    print(f"Saving FP16 checkpoint to {out_path}...")
    # Mocking actual save if not on Colab
    if os.path.exists('/content/drive/MyDrive'):
        torch.save(model.state_dict(), out_path)
    else:
        torch.save(model.state_dict(), 'moe_fp16.pt')
    print("Training complete.")

if __name__ == "__main__":
    train()
