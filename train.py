import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from miditoolkit import MidiFile, Instrument, Note
from tqdm import tqdm

# --- Dataset ---
class EMOPHIADataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.inputs = torch.tensor(data['x'], dtype=torch.float32)         # (N, T, F)
        self.targets = torch.tensor(data['y'], dtype=torch.long)           # (N, L, 8)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

# --- Transformer Encoder Layer ---
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# --- Transformer Decoder Layer ---
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        attn1, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout(attn1))
        attn2, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout(attn2))
        ff_output = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))
        return tgt

# --- Full Transformer Model ---
class Audio2MIDINetwork(nn.Module):
    def __init__(self, input_dim, d_model, vocab_size, max_len, n_heads, n_layers):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.input_proj = nn.Linear(d_model, d_model)
        self.pos_enc_in = PositionalEncoding(max_len, d_model)

        self.encoder = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads) for _ in range(n_layers)])
        #self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc_out = PositionalEncoding(max_len, d_model)
        self.decoder = nn.ModuleList([TransformerDecoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, y):
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_enc_in(x)
        for enc in self.encoder:
            x = enc(x)

        #y = self.token_embed(y) #each token is a vector of length 8, no need to create new embeddings of tokens
        y = self.pos_enc_out(y)
        for dec in self.decoder:
            y = dec(y, x)
        return self.out_proj(y)

# --- Training Setup ---
data_path = "/mnt/data/co-representation/co-representation/emopia_data.npz"
dataset = EMOPHIADataset(data_path)

train_len = int(0.6 * len(dataset))
val_len = int(0.1 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Audio2MIDINetwork(input_dim=8, d_model=8, vocab_size=256, max_len=1024, n_heads=4, n_layers=4).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# --- Training Loop ---
for epoch in range(10):
    model.train()
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        y_input = y[:, :-1, 0]  # use first dimension of token as class id
        y_target = y[:, 1:, 0]  # shifted target

        logits = model(x, y_input)
        logits = logits.view(-1, logits.size(-1))
        y_target = y_target.reshape(-1)

        loss = criterion(logits, y_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
