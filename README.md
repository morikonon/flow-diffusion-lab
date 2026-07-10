# Flow Diffusion Lab

Research project exploring:
- Diffusion models
- Flow matching
- Transformer-based generative models

## Goal
Compare diffusion vs flow matching for image generation.

## Dataset
CIFAR10 (initial)

## Models
- Diffusion Transformer
- Flow matching vector field model

## Setup
```bash
pip install -r requirements.txt
```

## Training
Trains the flow-matching DiT on CIFAR10 and saves a checkpoint to `checkpoints/`:
```bash
python train.py --epochs 10
```
Useful flags for a quick run: `--max-steps 200 --subset-size 512 --checkpoint-name my_run`.

## Comparing checkpoints
Launch a Gradio app to pick two saved checkpoints and compare their generations,
starting from the same noise seed:
```bash
python app.py
```
