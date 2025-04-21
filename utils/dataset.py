from collections import Counter
import torch
from torch.utils.data import Dataset
import nltk
import numpy as np
from nltk.tokenize import TreebankWordTokenizer

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
tokenizer = TreebankWordTokenizer()

class SNLIDataset(Dataset):
    def __init__(self, data, vocab=None, max_len=50):
        self.data = [ex for ex in data if ex["label"] != -1]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        premise =tokenizer.tokenize(sample["premise"].lower())
        hypothesis =tokenizer.tokenize(sample["hypothesis"].lower())

        if self.vocab:
            premise_ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in premise]
            hypothesis_ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in hypothesis]
        else:
            premise_ids = premise
            hypothesis_ids = hypothesis

        # Truncate and pad
        premise_ids = premise_ids[:self.max_len]
        hypothesis_ids = hypothesis_ids[:self.max_len]

        premise_ids += [self.vocab["<pad>"]] * (self.max_len - len(premise_ids))
        hypothesis_ids += [self.vocab["<pad>"]] * (self.max_len - len(hypothesis_ids))

        premise_len = min(len(premise), self.max_len)
        hypothesis_len = min(len(hypothesis), self.max_len)

        label = sample["label"]
        return torch.tensor(premise_ids), torch.tensor(hypothesis_ids), torch.tensor(label), premise_len, hypothesis_len


def build_vocab(dataset, min_freq=2):
    all_tokens = []

    for sample in dataset:
        if sample["label"] == -1:
            continue
        prem =tokenizer.tokenize(sample["premise"].lower())
        hypo =tokenizer.tokenize(sample["hypothesis"].lower())
        all_tokens.extend(prem)
        all_tokens.extend(hypo)

    freq = Counter(all_tokens)
    vocab = {"<pad>": 0, "<unk>": 1}

    for word, count in freq.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def load_glove_embeddings(glove_path, vocab, dim=300):
    # random init for words not in GloVe
    embeddings = np.random.normal(0, 0.1, (len(vocab), dim))
    embeddings[vocab["<pad>"]] = np.zeros(dim)

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)

            if word in vocab:
                idx = vocab[word]
                embeddings[idx] = vec
                found += 1

    print(f"Found {found} vectors out of {len(vocab)} words.")
    return torch.tensor(embeddings, dtype=torch.float)
