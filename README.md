# Final Project: Build & Fine-tunning GPT-2

This project is part of the Natural Language Processing course (CSE5321, CSEG321) at Sogang University.

## Project Overview

- **Core Implementation:**
  - 12-layer GPT-2 architecture with masked multi-head self-attention, positional encoding, and residual connections
  - AdamW optimizer with efficient bias correction and decoupled weight decay
  - Causal masking for autoregressive decoding

- **Downstream Tasks:**
  - **Sentiment Analysis** (CFIMDB, SST):
    - Last-layer: CFIMDB 0.869, SST 0.476
    - Full-model: CFIMDB 0.882, SST 0.397
  - **Paraphrase Detection** (Quora Question Pairs):
    - Full-model accuracy: 0.898
  - **Sonnet Generation** (Shakespeare Sonnets):
    - Full-model CHRF: 41.259

- **Main Experiment: Short Query Intent Classification**
  - Dataset: MASSIVE (SetFit/amazon_massive_intent_en-US), 60 intent labels, 11,500 train / 2,030 val / 2,970 test
  - Full-model: 85.3% accuracy, 81.2% micro F1 (en-US subset)
  - Competitive with mT5/XLM-R on the MASSIVE benchmark, despite being monolingual and lightweight
  - See report for detailed training and validation curves, and F1/accuracy plots

- **Training Efficiency:**
  - All experiments conducted on a single NVIDIA RTX 3080 GPU
  - Full-model fine-tuning achieves strong results in under 10 minutes (10 epochs)


[Fine-tuning GPT-2 for Short Query Intent Classification](report/report.pdf)

## Implementation Details

The project comprises two main parts:

### Part 1: Core Implementation
* modules/attention.py: Missing code blocks.
* modules/gpt2_layer.py: Missing code blocks.
* models/gpt2.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

### Part 2: Downstream Tasks
* Paraphrase detection using cloze-style classification
* Sonnet generation via autoregressive language modeling

## Testing Instructions

To test Part 1:
* `optimizer_test.py`: Test optimizer implementation
* `sanity_check.py`: Test GPT models implementation
* `classifier.py`: Perform sentiment classification

To test Part 2:
* `paraphrase_detection.py`: Perform paraphrase detection
* `sonnet_generation.py`: Perform sonnet generation

## Setup

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Acknowledgement

This project is adapted from a prior year's CS 224N project [Implement BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf).

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).