import argparse
import torch
import pickle
import os
import json
from datetime import datetime
from torch.utils.data import DataLoader
from datasets import load_dataset

from utils.dataset import SNLIDataset, load_glove_embeddings
from models import get_model

class SentEvalWrapper:
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_len = 50
        self.model_type = type(model).__name__

        # Safe defaults
        self.pad_id = self.vocab.get("<pad>", 0)
        self.unk_id = self.vocab.get("<unk>", 0)

        # Assert critical vocab keys exist
        assert "<pad>" in self.vocab, "Missing <pad> in vocab"
        assert "<unk>" in self.vocab, "Missing <unk> in vocab"

    def prepare(self, params, samples):
        return

    def batcher(self, params, batch):
        from nltk.tokenize import TreebankWordTokenizer
        tokenizer = TreebankWordTokenizer()

        vecs = []
        lengths = []

        for sent in batch:
            if isinstance(sent, list):
                tokens = [token.lower() for token in sent]
            else:
                sent = sent.strip()
                tokens = tokenizer.tokenize(sent.lower()) if sent else []

            if len(tokens) == 0:
                tokens = ["<pad>"]

            length = min(len(tokens), self.max_len)
            lengths.append(length)

            ids = [self.vocab.get(w, self.unk_id) for w in tokens[:self.max_len]]
            ids += [self.pad_id] * (self.max_len - len(ids))
            vecs.append(ids)

        if len(vecs) == 0:
            raise ValueError("Batcher received empty batch")

        input_tensor = torch.tensor(vecs).to(self.device)
        lengths_tensor = torch.tensor(lengths).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, 'encode_sentence'):
                reps = self.model.encode_sentence(input_tensor, lengths_tensor)
            elif hasattr(self.model, 'average_embeddings'):
                reps = self.model.average_embeddings(input_tensor)
            else:
                raise AttributeError(f"Model {self.model_type} doesn't have a sentence encoding method")

        return reps.cpu().numpy()

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for premise, hypothesis, labels, prem_len, hypo_len in dataloader:
            premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)
            prem_len, hypo_len = prem_len.to(device), hypo_len.to(device)

            logits = model(premise, prem_len, hypothesis, hypo_len)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['baseline', 'lstm', 'bilstm', 'bilstm_max'])
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--glove_path', type=str, default='./glove/glove.840B.300d.txt')
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("Loading SNLI dev/test splits...")
    snli = load_dataset("snli")
    raw_dev = snli["validation"]
    raw_test = snli["test"]

    print("Loading vocab...")
    with open("./checkpoints/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    print("Loading GloVe...")
    embedding_matrix = load_glove_embeddings(args.glove_path, vocab, dim=args.embedding_dim)

    print("Preparing datasets...")
    dev_dataset = SNLIDataset(raw_dev, vocab, max_len=args.max_len)
    test_dataset = SNLIDataset(raw_test, vocab, max_len=args.max_len)

    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print("Initializing model...")
    model = get_model(args.model_type, embedding_matrix, args).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device, weights_only=True))

    print("Evaluating SNLI dev...")
    dev_acc = evaluate(model, dev_loader, args.device)
    print(f"Dev Accuracy: {dev_acc:.4f}")

    print("Evaluating SNLI test...")
    test_acc = evaluate(model, test_loader, args.device)
    print(f"Test Accuracy: {test_acc:.4f}")

    print("Running SentEval transfer tasks...")
    import senteval
    params_senteval = {
        'task_path': './SentEval/data',
        'usepytorch': True,
        'kfold': 10,
        'batch_size': args.batch_size,
        'classifier': {
            'nhid': 0,
            'optim': 'adam',
            'batch_size': args.batch_size,
            'tenacity': 5,
            'epoch_size': 4,
            'device': args.device
        }
    }

    se_model = SentEvalWrapper(model, vocab, args.device)
    se = senteval.engine.SE(params_senteval, se_model.batcher, se_model.prepare)

    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    results = se.eval(transfer_tasks)

    accs = [(t, results[t]['acc'], results[t]['nexamples']) for t in results]
    macro = sum(a for _, a, _ in accs) / len(accs)
    micro = sum(a * n for _, a, n in accs) / sum(n for _, _, n in accs)

    print("\n=== Final Evaluation Summary ===")
    print(f"Model: {args.model_type}")
    print(f"SNLI dev accuracy : {dev_acc:.2f}")
    print(f"SNLI test accuracy: {test_acc:.2f}")
    print(f"SentEval macro    : {macro:.2f}")
    print(f"SentEval micro    : {micro:.2f}")

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/{args.model_type}_eval_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "model": args.model_type,
            "SNLI_dev_accuracy": round(dev_acc, 4),
            "SNLI_test_accuracy": round(test_acc, 4),
            "SentEval_macro": round(macro, 4),
            "SentEval_micro": round(micro, 4),
            "task_accuracies": {t: round(results[t]['acc'], 4) for t in results}
        }, f, indent=4)

    print(f"\n Results saved to: {results_file}")

if __name__ == "__main__":
    main()
