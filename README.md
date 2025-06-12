# CS 224N Default Final Project: Build GPT-2

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course
website for detailed instructions and an overview of the codebase.

## Project Overview

This project implements and evaluates a GPT-2-based language model across multiple NLP tasks:

1. **Basic Implementation**
   - GPT-2 architecture (12-layer, masked multi-head attention)
   - Adam optimizer with bias correction and decoupled weight decay

2. **Downstream Tasks Performance**
   - Sentiment Analysis: 97.6% accuracy on CFIMDB (binary)
   - Paraphrase Detection: 75.2% accuracy on Quora Question Pairs
   - Sonnet Generation: CHRF score of 0.68 on Shakespeare Sonnets

3. **Main Experiment: Short Query Intent Classification**
   - Dataset: MASSIVE (SetFit/amazon_massive_intent_en-US)
   - 60 distinct intent labels across 6 domains
   - Training set: 11,500 utterances
   - Validation set: 2,030 utterances
   - Test set: 2,970 utterances
   - Performance: 85.3% accuracy with full-model fine-tuning

[Fine-tuning GPT-2 for Short Query Intent
Classification](report/report.pdf)

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