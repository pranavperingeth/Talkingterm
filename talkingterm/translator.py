



import torch
import torch.nn as nn
 
# ── Config (must match trainer.py) ────────────────────────────────────────────
EMBED_DIM  = 128
HIDDEN_DIM = 256
MAX_LEN    = 20
MODEL_PATH = r"C:\Users\prana\talkingterm\models\model.pth"
 
PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
 
# ── Model (must match trainer.py exactly) ─────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
 
    def forward(self, x):
        _, (hidden, cell) = self.lstm(self.embedding(x))
        return hidden, cell
 
 
class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)
 
    def forward(self, x, hidden, cell):
        out, (hidden, cell) = self.lstm(self.embedding(x), (hidden, cell))
        return self.fc(out), hidden, cell
 
 
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)
 
    def forward(self, src, tgt):
        hidden, cell = self.encoder(src)
        logits, _, _ = self.decoder(tgt, hidden, cell)
        return logits
 
 
# ── Load checkpoint ────────────────────────────────────────────────────────────
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
 
word2idx   = checkpoint["word2idx"]
idx2word   = checkpoint["idx2word"]
vocab_size = checkpoint["vocab_size"]
 
model = Seq2Seq(vocab_size)
model.load_state_dict(checkpoint["model_state"])  # key is "model_state", not "model"
model.eval()
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
 
# ── Helpers ────────────────────────────────────────────────────────────────────
def encode(sentence):
    tokens = sentence.lower().split()
    ids    = [word2idx.get(t, word2idx[UNK]) for t in tokens]
    ids    = ids[:MAX_LEN] + [0] * max(0, MAX_LEN - len(ids))  # truncate + pad
    return ids
 
 
@torch.no_grad()
def translate(sentence: str) -> str:
    # Encode source
    src = torch.tensor([encode(sentence)]).to(device)
 
    # Run encoder once
    hidden, cell = model.encoder(src)
 
    # Decode token by token
    dec_input = torch.tensor([[word2idx[SOS]]]).to(device)
    result    = []
 
    for _ in range(MAX_LEN):
        logits, hidden, cell = model.decoder(dec_input, hidden, cell)
        pred_idx = logits.argmax(-1).item()
 
        if pred_idx == word2idx[EOS]:   # stop at end token
            break
 
        word = idx2word.get(pred_idx, UNK)
        if word not in (PAD, SOS, EOS):
            result.append(word)
 
        dec_input = torch.tensor([[pred_idx]]).to(device)
 
    return " ".join(result)
 
 
# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Model loaded. Type 'quit' to exit.\n")
    while True:
        sentence = input("Input: ").strip()
        if sentence.lower() == "quit":
            break
        if not sentence:
            continue
        print(f"Output: {translate(sentence)}\n")
