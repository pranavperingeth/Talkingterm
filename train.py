import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_DIM   = 128
HIDDEN_DIM  = 256
MAX_LEN     = 20
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 0.001
CLIP        = 1.0          # gradient clipping
MODEL_PATH  = "models/model.pth"
DATA_PATH   = "dataset/dataset.csv"

PAD, SOS, UNK = "<PAD>", "<SOS>", "<UNK>"
EOS = "<EOS>"

# ── Vocabulary ─────────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self):
        self.word2idx = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build(self, sentences):
        for sentence in sentences:
            for word in tokenize(sentence):
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def encode(self, sentence, add_eos=False):
        tokens = [self.word2idx.get(w, self.word2idx[UNK]) for w in tokenize(sentence)]
        if add_eos:
            tokens.append(self.word2idx[EOS])
        return tokens[:MAX_LEN]  # truncate if too long

    def decode(self, indices):
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, UNK)
            if word == EOS:
                break
            if word not in (PAD, SOS):
                words.append(word)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


def tokenize(sentence):
    return sentence.lower().split()


def pad_sequence(seq, max_len):
    return seq[:max_len] + [0] * max(0, max_len - len(seq))


# ── Dataset ────────────────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, inputs, outputs, vocab):
        self.pairs = []
        for src, tgt in zip(inputs, outputs):
            src_enc = pad_sequence(vocab.encode(src), MAX_LEN)
            # target: decoder input starts with SOS, label ends with EOS
            tgt_tokens = vocab.encode(tgt, add_eos=True)
            dec_input  = pad_sequence([vocab.word2idx[SOS]] + tgt_tokens[:-1], MAX_LEN)
            dec_target = pad_sequence(tgt_tokens, MAX_LEN)
            self.pairs.append((
                torch.tensor(src_enc),
                torch.tensor(dec_input),
                torch.tensor(dec_target),
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ── Model ──────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell   # context vectors passed to decoder


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)                      # (B, seq, E)
        out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        logits = self.fc(out)                             # (B, seq, vocab)
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)

    def forward(self, src, tgt):
        hidden, cell = self.encoder(src)
        logits, _, _  = self.decoder(tgt, hidden, cell)
        return logits   # (B, seq, vocab_size)


# ── Training ───────────────────────────────────────────────────────────────────
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for src, dec_input, dec_target in loader:
        src, dec_input, dec_target = src.to(device), dec_input.to(device), dec_target.to(device)

        optimizer.zero_grad()
        logits = model(src, dec_input)                    # (B, seq, vocab)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),           # (B*seq, vocab)
            dec_target.reshape(-1)                         # (B*seq,)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP) # prevent exploding gradients
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def translate(sentence, model, vocab, device):
    model.eval()
    src = torch.tensor([pad_sequence(vocab.encode(sentence), MAX_LEN)]).to(device)
    hidden, cell = model.encoder(src)

    dec_input = torch.tensor([[vocab.word2idx[SOS]]]).to(device)
    result = []

    for _ in range(MAX_LEN):
        logits, hidden, cell = model.decoder(dec_input, hidden, cell)
        pred_idx = logits.argmax(-1).item()
        if pred_idx == vocab.word2idx[EOS]:
            break
        result.append(pred_idx)
        dec_input = torch.tensor([[pred_idx]]).to(device)

    return vocab.decode(result)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data    = pd.read_csv(DATA_PATH)
    inputs  = data["input"].astype(str).str.lower().tolist()
    outputs = data["output"].astype(str).str.lower().tolist()

    # Build vocab from both sides
    vocab = Vocabulary()
    vocab.build(inputs + outputs)
    print(f"Vocabulary size: {len(vocab)}")

    # Dataset & loader
    dataset = TranslationDataset(inputs, outputs, vocab)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, loss, optimizer
    model     = Seq2Seq(len(vocab)).to(device)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)   # ignore PAD tokens in loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Train
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(model, loader, optimizer, loss_fn, device)
        scheduler.step(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state":  model.state_dict(),
        "word2idx":     vocab.word2idx,
        "idx2word":     vocab.idx2word,
        "vocab_size":   len(vocab),
    }, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # Quick inference test
    test_sentence = inputs[0]
    prediction    = translate(test_sentence, model, vocab, device)
    print(f"\nTest translation\n  Input : {test_sentence}\n  Output: {prediction}")
