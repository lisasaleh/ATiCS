import nltk
nltk.download("punkt", quiet=True)

from nltk.tokenize import word_tokenize
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import datetime

from utils.dataset import SNLIDataset, build_vocab, load_glove_embeddings
from models import get_model


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for premise, hypothesis, labels in dataloader:
            premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)
            
            outputs = model(premise, hypothesis)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total



def train(args):
    # Load SNLI subset
    print("Loading SNLI...")
    if args.use_part_data:
        raw_train = load_dataset("snli")["train"].select(range(1000))
        raw_val = load_dataset("snli")["validation"].select(range(500))
    else:
        raw_train = load_dataset("snli")["train"]
        raw_val = load_dataset("snli")["validation"]

    # Build vocab
    print("Building vocab...")
    vocab = build_vocab(raw_train, min_freq=1)

    # Load GloVe embeddings
    print("Loading GloVe...")
    embedding_matrix = load_glove_embeddings(args.glove_path, vocab, dim=args.embedding_dim)

    # Prepare datasets
    train_dataset = SNLIDataset(raw_train, vocab, max_len=args.max_len)
    val_dataset = SNLIDataset(raw_val, vocab, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Init model, loss, optimizer
    print("Initializing model...")
    model = get_model(args.model_type, embedding_matrix, args).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Train loop
    print("Training...")
    log_dir = os.path.join("runs", f"{args.model_type}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for premise, hypothesis, labels, prem_len, hypo_len in train_loader:
            premise = premise.to(args.device)
            hypothesis = hypothesis.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            logits = model(premise, prem_len, hypothesis, hypo_len)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5) #add clipping
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, LR:{optimizer.param_groups[0]['lr']:.5f} ")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

        # do the LR decay if nessecary
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.checkpoint_path, f"{args.model_type}_valacc{best_val_acc:.4f}.pt")
            print(f"Saving best model to {save_path}")
            torch.save(model.state_dict(), save_path)
        elif val_acc < best_val_acc:
            for g in optimizer.param_groups:
                g["lr"] *= args.lr_decay

        # Stop if LR < 1e-5
        if optimizer.param_groups[0]["lr"] < 1e-5:
            print("Early stopping: learning rate below 1e-5.")
            break
    
    writer.close()
    print("Training finished.")