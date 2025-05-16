import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from miditoolkit import MidiFile, Instrument, Note
from tqdm import tqdm
import wandb
import gc

# --- Dataset ---
class EMOPHIADataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.inputs = torch.tensor(data['x'], dtype=torch.float32)
        self.targets = torch.tensor(data['y'], dtype=torch.long)

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

# --- Embedding Module ---
class MIDIEmbedder(nn.Module):
    def __init__(self, vocab_sizes, d_feature, d_model):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(vocab_size, d_feature) for vocab_size in vocab_sizes
        ])
        self.proj = nn.Linear(8 * d_feature, d_model)

    def forward(self, y):
        y_embeds = [embed(y[:, :, i]) for i, embed in enumerate(self.embeds)]
        concat = torch.cat(y_embeds, dim=-1)
        return self.proj(concat)

# --- Full Transformer Model ---
class Audio2MIDINetwork(nn.Module):
    def __init__(self, input_dim, d_model, max_len, n_heads, n_layers, vocab_sizes):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.input_proj = nn.Linear(d_model, d_model)
        self.pos_enc_in = PositionalEncoding(max_len, d_model)
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.pos_enc_out = PositionalEncoding(max_len, d_model)
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for vocab_size in vocab_sizes
        ])

    def forward(self, x, y_emb):
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_enc_in(x)
        for enc in self.encoder:
            x = enc(x)
        y_proj = self.pos_enc_out(y_emb)
        for dec in self.decoder:
            y_proj = dec(y_proj, x)
        logits = []
        for head in self.output_heads:
            logits.append(head(y_proj))
            torch.cuda.empty_cache()
        return logits

# --- Load Data ---
data_path = "emopia_data_structured.npz"
dataset = EMOPHIADataset(data_path)
train_len = int(0.6 * len(dataset))
val_len = int(0.1 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4)
test_loader = DataLoader(test_set, batch_size=4)

# --- Vocab Sizes ---
all_targets = torch.cat([y[:, :, :, 0] for _, y in train_loader], dim=0)
vocab_sizes = [int(all_targets[:, :, i].max().item()) + 1 for i in range(8)]
del all_targets

# --- Hyperparameters ---
d_feature = 16 #32 64
d_model = 8 * d_feature
max_len = 200 #512

# --- Model Init ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = MIDIEmbedder(vocab_sizes, d_feature, d_model).to(device)
model = Audio2MIDINetwork(64, d_model, max_len, 4, 6, vocab_sizes).to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(embedder.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()
best_val_loss = float('inf')
patience = 3
patience_counter = 0

wandb.init(project="audio2midi-emopia", entity="dtian")
wandb.watch(model)

# --- Training Loop ---
for epoch in range(20):
    model.train()
    embedder.train()
    train_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        x = x[:, :max_len, :]
        y = y[:, :max_len + 1, :, :]
        y_input = y[:, :-1, :, 0]
        y_target = y[:, 1:, :, 0]

        y_input_emb = embedder(y_input)
        logits = model(x, y_input_emb)

        loss = 0
        for i in range(8):
            out = logits[i]
            loss += criterion(out.reshape(-1, out.shape[-1]), y_target[:, :, i].reshape(-1))
            del out
            torch.cuda.empty_cache()
        loss /= 8

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        del loss
        optimizer.step()
        del logits, y_input_emb, x, y, y_input, y_target
        torch.cuda.empty_cache()
        gc.collect()

    avg_train_loss = train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    embedder.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            x = x[:, :max_len, :]
            y = y[:, :max_len + 1, :, :]
            y_input = y[:, :-1, :, 0]
            y_target = y[:, 1:, :, 0]
            y_input_emb = embedder(y_input)
            logits = model(x, y_input_emb)
            loss = 0
            for i in range(8):
                out = logits[i]
                loss += criterion(out.reshape(-1, out.shape[-1]), y_target[:, :, i].reshape(-1))
                del out
                torch.cuda.empty_cache()
            loss /= 8
            val_loss += loss.item()
            del logits, y_input_emb, x, y, y_input, y_target
            torch.cuda.empty_cache()
            gc.collect()

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

# --- Evaluation ---
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
embedder.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        x = x[:, :max_len, :]
        y = y[:, :max_len + 1, :, :]
        y_input = y[:, :-1, :, 0]
        y_target = y[:, 1:, :, 0]
        y_input_emb = embedder(y_input)
        logits = model(x, y_input_emb)
        for i in range(8):
            pred = logits[i].argmax(dim=-1)
            target = y_target[:, :, i]
            mask = target != 0
            correct += ((pred == target) * mask).sum().item()
            total += mask.sum().item()
            del pred
        del logits, y_input_emb, x, y, y_input, y_target
        torch.cuda.empty_cache()
        gc.collect()

accuracy = 100.0 * correct / total
print(f"Test Accuracy (ignoring padding): {accuracy:.2f}%")
wandb.log({"test_accuracy": accuracy})
