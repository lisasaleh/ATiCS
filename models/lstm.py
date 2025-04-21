import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, args):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.hidden_dim = 512
        self.num_classes = 3
        self.dropout = nn.Dropout(p=0.2)

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

    def encode_sentence(self, x, lengths):
        embedded = self.embedding(x)  # (batch, seq, emb_dim)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (1, batch, hidden_dim)
        return self.dropout(h_n[-1])

    def forward(self, premise, prem_len, hypothesis, hypo_len):
        u = self.encode_sentence(premise, prem_len)
        v = self.encode_sentence(hypothesis, hypo_len)

        combined = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        return self.mlp(combined)
