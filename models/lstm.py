import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, args):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.hidden_dim = 512
        self.num_classes = 3

        self.lstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

        combined_size = self.hidden_dim * 4  # [u, v, |u−v|, u*v]
        self.mlp = nn.Sequential(
            nn.Linear(combined_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def encode_sentence(self, x):
        embedded = self.embedding(x)  # (batch, seq, emb_dim)
        _, (h_n, _) = self.lstm(embedded)  # h_n: (1, batch, hidden)
        return h_n.squeeze(0)  # → (batch, hidden)

    def forward(self, premise, hypothesis):
        u = self.encode_sentence(premise)
        v = self.encode_sentence(hypothesis)

        combined = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        return self.mlp(combined)
