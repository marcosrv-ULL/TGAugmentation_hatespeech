# From Slurs to Slots: LLM Masking and Telephone-Game Augmentation for Multiclass Hate Speech Detection

This repository contains the code, models, and results for the paper **"From slurs to slots: LLM masking and Telephone-Game Augmentation for multiclass hate speech detection"** (Online Social Networks and Media, 2026).

## Overview

We present an offline/online hybrid framework for multiclass hate speech detection designed for efficiency and robustness:
* **Offline Phase:** Large Language Models (LLMs) are used only at training time to mask volatile slurs and targets into normalized placeholders (e.g., `[SLUR:]`, `[TARGET:]`). We also introduce **Telephone-Game Augmentation (TGA)**, which generates label-preserving paraphrases to expand the training data, particularly boosting minority classes and low-resource regimes.
* **Online Phase:** Lightweight classifiers (such as Linear SVM, Logistic Regression, etc.) operate on these masked inputs during inference. This allows for predictable, stable latency and highly efficient moderation at scale.

Our evaluation shows that masking improves binary macro-F1 by +0.073 and accuracy by +0.071 on average across small models. In multiclass setups, Linear SVM improves by +0.141 macro-F1. Under equal budgets, our TGA framework outperforms a back-translation baseline by +0.109 macro-F1. 

## ⚠️ Important Note on Datasets

**Please note that the original and augmented datasets (except for the lexicon) are not included in this repository. However, they are available upon request. You can contact the authors at mrodrive@ull.edu.es, and we will respond quickly.**

This approach aligns with responsible release practices to prevent the direct exposure of harmful or hateful text strings. The only dataset included natively is the redacted lexicon (`datasets/lexicon/lexicon.jsonl`), which contains masked forms, placeholder taxonomies, and aggregated counts.

## Repository Structure

**This version is not aim to serve as a framework some filepath are changed due to changes in the structure of the code to achieve better legibility, further updates will aim to achieve a unified framework architecture**

* **`datasets/`**: Expected directory for the dataset splits (ETHOS, SuperTweetEval, MultilingualTweetEval2024). Raw and augmented files (`.jsonl`, `.csv`, `.txt`) are omitted but can be requested and placed here.
  * `lexicon/`: Contains the generated `lexicon.jsonl` resource.
* **`models/`**: Stores the output artifacts from training and evaluating our lightweight classifiers (e.g., Logistic Regression, Linear SVM, SGD, Passive-Aggressive, and Complement Naive Bayes) using TF-IDF and RoBERTa features. Includes `.joblib` model weights, classification reports, and confusion matrices across different data regimes and random seeds.
* **`results/`**: Contains CSV summaries of the experiments across different datasets and the figures exported for the paper (e.g., threshold sensitivity analysis plots and SVM vs. Transformer comparisons).
* **`src/`**: The main source code directory containing scripts for the full experimental pipeline.
  * **Masking Pipeline**: Scripts like `mask_with_lexicon.py`, `mask_with_llm.py`, and `apply_lexicon.py`.
  * **Augmentation**: Scripts such as `augment_qwen.py`, `extract_augmentations.py`, `backtranslate_dataset.py`, and the `tga/` folder containing notebooks for the Telephone-Game Augmentation paraphrase chain.
  * **Experimentation & Evaluation**: Scripts like `experiments.sh`, `make_splits.py`, `low_regime.py`, and error analysis modules.
* **`enviroment.yml` & `enviroment-gpu.yml`**: Conda environment files to reproduce the software dependencies.

## Installation

You can replicate the project's environment using the provided Conda YAML files:

```bash
# For a standard CPU environment
conda env create -f enviroment.yml
conda activate <env_name>

# For a GPU-enabled environment
conda env create -f enviroment-gpu.yml
conda activate <env_name>
```

## Reproducibility

To run the experiments, use the scripts provided in the src/ directory. For instance, src/experiments.sh contains the main bash commands used to execute the evaluation loops. Make sure you have requested and placed the dataset files in the datasets/ hierarchy before running the complete pipeline.

## Citation

If you find this code, lexicon, or methodology useful in your research, please cite our paper:

```
@article{RODRIGUEZVEGA2026100346,
title = {From slurs to slots: LLM masking and Telephone-Game Augmentation for multiclass hate speech detection},
journal = {Online Social Networks and Media},
volume = {53},
pages = {100436},
year = {2026},
issn = {2468-6964},
doi = {[https://doi.org/10.1016/j.osnem.2026.100436](https://doi.org/10.1016/j.osnem.2026.100436)},
url = {[https://www.sciencedirect.com/science/article/pii/S2468696426000029](https://www.sciencedirect.com/science/article/pii/S2468696426000029)},
author = {Marcos Rodriguez-Vega and Carlos Rosa-Remedios and Pino Caballero-Gil},
keywords = {Online moderation, Hate speech detection, Data augmentation, LLM masking, Efficiency, Low-resource}
}
```