import torch
import torch.nn as nn

class BiLSTMMaxPoolClassifier(nn.Module):
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

        # Sentence vector = max pooled BiLSTM output → 2 * hidden_dim
        sent_repr_dim = self.hidden_dim * 2
        combined_dim = sent_repr_dim * 4  # [u, v, |u−v|, u*v]

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def encode_sentence(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Max pooling over time
        return torch.max(unpacked, dim=1).values  # (batch, 2*hidden)
    
    def forward(self, premise, prem_len, hypothesis, hypo_len):
        u = self.encode_sentence(premise, prem_len)
        v = self.encode_sentence(hypothesis, hypo_len)

        combined = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        return self.mlp(combined)