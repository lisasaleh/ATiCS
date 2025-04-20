import argparse
import torch
from models import get_model
from utils.train_utils import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='baseline', choices=['baseline', 'lstm', 'bilstm', 'bilstm_max'])
    parser.add_argument('--glove_path', type=str, default='glove/glove.6B.300d.txt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
    parser.add_argument('--embedding_dim', type=int, default=300, help="Dimensionality of GloVe embeddings (e.g., 100 or 300)")
    parser.add_argument('--use_part_data', action='store_true', help="Use small subset of SNLI for local testing")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()