import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from miditoolkit import MidiFile, Instrument, Note
from tqdm import tqdm
import wandb

# --- Dataset ---
class EMOPHIADataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.inputs = torch.tensor(data['x'], dtype=torch.float32)       # (N, 1024, 64)
        self.targets = torch.tensor(data['y'], dtype=torch.long)         # (N, 1024, 8, 2)

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
    def __init__(self, input_dim, d_feature, vocab_sizes, max_len, d_model, n_heads, n_layers):
        super().__init__()
        assert len(vocab_sizes) == 8, "Expecting 8 vocab sizes (one for each feature)"
        assert d_model == 8 * d_feature, "d_model must be 8 × d_feature"

        self.conv1d = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.input_proj = nn.Linear(d_model, d_model)
        self.pos_enc_in = PositionalEncoding(max_len, d_model)

        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])

        self.feature_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, d_feature) for vocab_size in vocab_sizes
        ])
        self.concat_proj = nn.Linear(8 * d_feature, d_model)
        self.pos_enc_out = PositionalEncoding(max_len, d_model)

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])

        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for vocab_size in vocab_sizes
        ])

    def forward(self, x, y):
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_enc_in(x)
        for enc in self.encoder:
            x = enc(x)

        y_embeds = [embed(y[:, :, i]) for i, embed in enumerate(self.feature_embeds)]
        y_concat = torch.cat(y_embeds, dim=-1)
        y_proj = self.concat_proj(y_concat)
        y_proj = self.pos_enc_out(y_proj)
        for dec in self.decoder:
            y_proj = dec(y_proj, x)

        logits = [head(y_proj) for head in self.output_heads]  # List of 8 tensors
        return logits

# --- Training Setup ---
data_path = "emopia_data_structured.npz"
dataset = EMOPHIADataset(data_path)

train_len = int(0.6 * len(dataset))
val_len = int(0.1 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)
test_loader = DataLoader(test_set, batch_size=16)

# --- Automatically extract vocab sizes ---
# Extract vocab sizes from the dataset
print("Extracting vocab sizes...")
all_targets = []

for _, y in train_loader:
    all_targets.append(y[:, :, :, 0])  # shape: (B, T, 8)

all_targets = torch.cat(all_targets, dim=0)  # shape: (N, T, 8)

# Compute vocab size for each feature
vocab_sizes = [int(all_targets[:, :, i].max().item()) + 1 for i in range(8)]
print("Vocab sizes:", vocab_sizes)

# --- Hyperparameters ---
d_feature = 64
d_model = 8 * d_feature

# --- Model, Optimizer, Loss ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Audio2MIDINetwork(
    input_dim=64,
    d_feature=d_feature,
    vocab_sizes=vocab_sizes,
    max_len=1024,
    d_model=d_model,
    n_heads=8,
    n_layers=6
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
best_val_loss = float('inf')
patience = 3
patience_counter = 0

wandb.init(
  project="audio2midi-emopia",
  entity="dtian")

wandb.watch(model)

# --- Training Loop ---
for epoch in range(20):
    model.train()
    train_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        y_input = y[:, :-1, :, 0]
        y_target = y[:, 1:, :, 0]

        logits = model(x, y_input)
        loss = sum(
            criterion(logits[i].reshape(-1, logits[i].shape[-1]), y_target[:, :, i].reshape(-1))
            for i in range(8)
        ) / 8

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_input = y[:, :-1, :, 0]
            y_target = y[:, 1:, :, 0]
            logits = model(x, y_input)
            loss = sum(
                criterion(logits[i].reshape(-1, logits[i].shape[-1]), y_target[:, :, i].reshape(-1))
                for i in range(8)
            ) / 8
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        wandb.save("best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break

# --- Evaluation on Test Set ---
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        y_input = y[:, :-1, :, 0]
        y_target = y[:, 1:, :, 0]  # (B, T, 8)
        logits = model(x, y_input)  # List of 8 logits: each (B, T, vocab_i)

        for i in range(8):
            pred = logits[i].argmax(dim=-1)  # (B, T)
            target = y_target[:, :, i]
            mask = target != 0  # ignore padding (token 0)
            correct += ((pred == target) * mask).sum().item()
            total += mask.sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy (ignoring padding): {accuracy:.2f}%")
wandb.log({"test_accuracy": accuracy})
