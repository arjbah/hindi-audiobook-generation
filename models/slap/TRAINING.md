# SLAP Training Setup

## Prerequisites

Install PyTorch with CUDA support (replace `cu121` with your CUDA version from `nvidia-smi`):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install remaining dependencies:

```bash
pip install fire librosa mir_eval numpy pandas requests rich rootutils scikit-learn timm tokenizers torchaudio torchlibrosa torchvision tqdm transformers treetable omegaconf submitit retrying torchmetrics lazy-loader huggingface_hub hydra-colorlog umap-learn plotly==5.24.1 "dora-search==0.1.12" "hydra-core==1.3.2" "wandb==0.17.6" "lightning==2.4.0" "lightning_utilities==0.11.9" datasets soundfile
```

---

## Required Files and Folders

```
SLAP/
├── .project-root                          # empty marker file, required by rootutils
├── pretrained/
│   └── HTSAT_AudioSet_Saved_6.ckpt        # HTS-AT pretrained checkpoint
├── src/
│   ├── data/
│   │   └── rasa_dataset.py                # Rasa HuggingFace DataModule
│   ├── networks/
│   │   ├── audio/
│   │   │   └── htsat.py                   # HTS-AT audio encoder (unchanged)
│   │   ├── text/
│   │   │   └── bert.py                    # text encoder, extended for MuRIL
│   │   ├── siamese_arm.py                 # BYOL arm wrapper (unchanged)
│   │   └── mlp.py                         # MLP projection/predictor heads (unchanged)
│   ├── models/
│   │   └── slap.py                        # SLAP Lightning module (unchanged)
│   └── utils/
│       └── losses/
│           ├── byol.py                    # BYOL loss (unchanged)
│           └── __init__.py                # fixed to remove missing imports
└── configs/
    ├── data/
    │   └── rasa.yaml                      # Rasa dataset config
    ├── model/
    │   ├── slap.yaml                      # SLAP model config (unchanged)
    │   ├── audio_encoder/
    │   │   ├── htsat_audioset.yaml        # HTS-AT tiny + AudioSet checkpoint
    │   │   └── htsat_audioset_slap.yaml   # same + BYOL predictor head
    │   └── text_encoder/
    │       ├── muril.yaml                 # MuRIL encoder + projector
    │       └── muril_slap.yaml            # same + BYOL predictor head
    └── trainer/
        └── hindi.yaml                     # GPU, bf16, 120 epochs, accum=8
```

---

## Setup

Create the required marker file (once only):

```bash
cd SLAP
New-Item -ItemType File -Path .project-root   # PowerShell
# or on Linux/Mac:
touch .project-root
```

---

## Training Command

```bash
cd SLAP
python src/train.py data=rasa model=slap "model/audio_encoder=htsat_audioset_slap" "model/text_encoder=muril_slap" trainer=hindi logger=wandb
```

---

## Training Configuration

| Setting | Value |
|---|---|
| Audio encoder | HTS-AT tiny (embed_dim=96), pretrained on AudioSet |
| Text encoder | MuRIL (google/muril-base-cased), pretrained |
| Embedding dim | 512 |
| Dataset | ai4bharat/Rasa, Hindi split |
| Audio sample rate | 32 kHz |
| Loss | BYOL (no negatives) |
| Global batch size | 128 |
| Effective batch size | 128 (`accumulate_grad_batches=1`) |
| Epochs | 120 |
| Precision | bf16-mixed |
| EMA tau | 0.995 (fixed) |
| Optimizer | Adam, lr=1e-4 |
| Scheduler | Linear warmup (10 epochs) + cosine annealing |
