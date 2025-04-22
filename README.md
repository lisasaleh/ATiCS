# ATiCS: Sentence Representations from Natural Language Inference

This repository implements several sentence encoders, **Baseline**, **LSTM**, **BiLSTM**, and **BiLSTM-Max**, trained on the **SNLI** (Stanford Natural Language Inference) dataset. The models are evaluated both on the SNLI classification task and on a diverse set of transfer tasks using **SentEval** (Conneau & Kiela, 2018), following the framework introduced by Conneau et al. (2017).


## Setup
### 1. Create and activate the enviroment

With conda:
```
conda env create -f enviroment.yml
conda activate atics
```
Or with pip:
```
pip install -r requirements.txt
```
### 2. Download GloVe embeddings
Download `glove.840B.300d.txt` from https://nlp.stanford.edu/projects/glove/ and place it in the following folder:
```
ATiCS/glove/glove.840B.300d.txt
```
### 3. Download SentEval transfer tasks
Go to the `SentEval/data/downstream/` folder and extract the files":
```
cd SentEval/data/downstream
bash get_transfer_data.bash
```
Alternatively, manually download and extract the data from https://dl.fbaipublicfiles.com/SentEval/data.zip into the `SentEval/data` folder.

### 4. Pretrained Models

You can download all trained model checkpoints and supporting files from the following Google Drive folder:

**[Google Drive: Model Checkpoints & Logs](https://drive.google.com/drive/folders/10hVweNZeJXwwwbJbBT6hsfhO1jGOAfXc?usp=sharing)**

The folder contains:
- Trained `.pt` files for each of the four sentence encoder models: `baseline`, `lstm`, `bilstm`, `bilstm_max`
- A `vocab.pkl` file containing the vocabulary used during training (needed in the jupyter notebook)
- Screenshots from TensorBoard showing validation curves

## Training
To train a model:
```
python train.py \
  --model_type bilstm_max \
  --glove_path glove/glove.840B.300d.txt \
  --epochs 20 \
  --checkpoint_path checkpoints/
```
Add `--use_part_data` to run on a small debug subset.`
TensorBoard logs will be written to the `runs/` directory.

## Evaluation
To evaluate a trained model on the SNLI test set and SentEval transfer tasks:
```
python eval.py \
  --model_type bilstm_max \
  --checkpoint_path checkpoints/bilstm_max.pt \
  --glove_path glove/glove.840B.300d.txt
```
### Example output
```
SNLI dev accuracy : 82.41
SNLI test accuracy: 82.01
SentEval macro    : 83.70
SentEval micro    : 83.41
```

## Folder Structure
```
ATiCS/
├── train.py                # Training script
├── eval.py                 # Evaluation script (SNLI + SentEval)
├── utils/
|   |__ __init__.py
│   ├── dataset.py
│   └── train_utils.py
├── models/
|   |__ __init__.py
│   ├── baseline.py
│   ├── lstm.py
│   ├── bilstm.py
│   └── bilstm_max.py
├── checkpoints/            # Saved model weights (.pt)
├── glove/                  # GloVe embeddings
├── SentEval/               # SentEval repo and data/downstream/
├── runs/                   # TensorBoard logs
├── environment.yml
├── requirements.txt
└── README.md
```

## Notes
- Models are trained using SGD with a learning rate decay factor of 0.99 per epoch
- Early stopping is triggered when the learning rate falls below 1e-5
- LSTM models include dropout (p=0.2) to improve convergence
- Padding is handled using `pack_padded_sequence` for all LSTM-based encoders
- Sentence embeddings are evaluated using macro and micro accuracy metrics on SentEval tasks, as introduced by Conneau et al. (2017).

### References

- Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015).  
  *A large annotated corpus for learning natural language inference*.  
  arXiv preprint [arXiv:1508.05326](https://arxiv.org/abs/1508.05326)

- Conneau, A., Kiela, D., Schwenk, H., Barrault, L., & Bordes, A. (2017).  
  *Supervised learning of universal sentence representations from natural language inference data*.  
  In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.  
  [arXiv:1705.02364](https://arxiv.org/abs/1705.02364)

- Conneau, A., & Kiela, D. (2018).  
  *SentEval: An evaluation toolkit for universal sentence representations*.  
  In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.  
  [ACL Anthology L18-1269](https://aclanthology.org/L18-1269/)
