import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, args):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.hidden_dim = 512
        self.num_classes = 3

        self.bilstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        sent_repr_dim = self.hidden_dim * 2
        combined_dim = sent_repr_dim * 4  # [u, v, |u−v|, u*v]
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def encode_sentence(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)
        _, (h_n, _) = self.bilstm(embedded)  # h_n: (2, batch, hidden)
        # Split directions
        forward_final = h_n[0]      # (batch, hidden)
        backward_final = h_n[1]     # (batch, hidden)
        return torch.cat([forward_final, backward_final], dim=1)  # (batch, 2*hidden)

    def forward(self, premise, hypothesis):
        u = self.encode_sentence(premise)
        v = self.encode_sentence(hypothesis)
        combined = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        return self.mlp(combined)