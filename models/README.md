# CLAP Models

Install the dependencies:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r models/requirements.txt
sudo apt update
sudo apt install ffmpeg
export HF_HOME=/root/.cache/huggingface
hf auth login
```

Train a model:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python models/train.py --model all --language hindi --gender male --batch_size 256
```

Use `--model all` to train all four models sequentially.
Pass `--indicvoices_limit 1500` to choose the number of IndicVoices validation samples.

Evaluate a trained model:

```bash
python models/evaluate_clap_model.py --model laion_clap --language hindi --gender male
```

Pass `--path checkpoint.pt` to evaluate a specific checkpoint.
Pass `--indicvoices_limit 1500` to choose the number of IndicVoices samples.

Valid models are `laion_clap`, `voiceclap`, `mga_clap`, `slap`, and `all`. Language defaults to `all`, gender defaults to `male`, and evaluation results are written to `models/evaluation.txt`.

Best checkpoints are saved as `checkpoints/<language>_best_rasa.pt` in each model folder. Processed dataset caches are stored in `models/.cache`.
