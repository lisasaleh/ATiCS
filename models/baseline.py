import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineClassifier(nn.Module):
    def __init__(self, embedding_matrix, args):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.hidden_dim = 512
        self.num_classes = 3

        combined_size = args.embedding_dim * 4  # [u, v, |u-v|, u*v]
        self.mlp = nn.Sequential(
            nn.Linear(combined_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def average_embeddings(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, emb_dim)
        mask = (x != 0).unsqueeze(-1)  # (batch_size, seq_len, 1), pad token = 0
        summed = torch.sum(embedded * mask, dim=1)  # sum over valid tokens
        lengths = torch.sum(mask, dim=1)  # count of non-pad tokens
        avg = summed / lengths.clamp(min=1e-9)  # avoid div by 0
        return avg  # (batch_size, emb_dim)

    def forward(self, premise, prem_len, hypothesis, hypo_len):
        u = self.average_embeddings(premise)
        v = self.average_embeddings(hypothesis)

        combined = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        return self.mlp(combined)