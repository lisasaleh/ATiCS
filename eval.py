import argparse
import torch
import pickle
from torch.utils.data import DataLoader
from datasets import load_dataset

from utils.dataset import SNLIDataset, build_vocab, load_glove_embeddings
from models import get_model

class SentEvalWrapper:
    def __init__(self, encoder, vocab, device):
        self.encoder = encoder
        self.vocab = vocab
        self.device = device
        self.max_len = 50

    def prepare(self, params, samples):
        return
    
    def batcher(self, params, batch):
        from nltk.tokenize import TreebankWordTokenizer
        tokenizer = TreebankWordTokenizer()

        vecs = []
        lengths = []
        
        for sent in batch:
            # Process tokens based on input type
            if isinstance(sent, list):
                tokens = [token.lower() for token in sent]
            else:
                tokens = tokenizer.tokenize(sent.lower())
            
            # Calculate length once
            length = min(len(tokens), self.max_len)
            lengths.append(length)
            
            # Create ids
            ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in tokens[:self.max_len]]
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
            vecs.append(ids)

        input_tensor = torch.tensor(vecs).to(self.device)
        lengths_tensor = torch.tensor(lengths).to(self.device)

        with torch.no_grad():
            reps = self.encoder.encode_sentence(input_tensor, lengths_tensor)
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
            'epoch_size': 4
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

if __name__ == "__main__":
    main()