# Evaluating State-Space Models and LLM Sentiment in Financial Time-Series: A SHAP and Regime-Based Study

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **A PyTorch implementation evaluating hybrid forecasting architectures, integrating state-space models (SAMBA, MambaStock) and traditional baselines with FinBERT sentiment analysis, enhanced by SHAP explainability.**

---

## 📌 Overview

This repository provides the codebase for a comprehensive comparative study evaluating how different sequence models tackle non-stationary financial time-series forecasting. 

Rather than focusing on a single architecture, this framework addresses the limitations of standard networks by evaluating the combination of:

1.  **State-Space Model Backbones:** Implementations of Mamba-based architectures (SAMBA, MambaStock) alongside standard baselines (LSTM, DLinear) for the efficient, linear-time sequence modelling of quantitative market data (OHLCV).
2.  **FinBERT Integration:** A pre-trained financial LLM to process news headlines, injecting qualitative sentiment signals to contextualise price movements.
3.  **Explainability & Regime Analysis:** Utilisation of SHAP (Shapley Additive Explanations) to interpret feature importance, coupled with performance evaluation across varying market volatility regimes.
4.  **Walk-Forward Validation:** A rigorous testing protocol that eliminates look-ahead bias, ensuring all models are evaluated under realistic "live trading" conditions.

---

## 🏗️ System Architecture

The evaluation pipeline processes two distinct data streams across multiple predictive engines:

* **Quantitative Stream:** Historical price data and technical indicators are processed via the chosen predictive engine (e.g., SAMBA, MambaStock, LSTM). The state-space models utilise a Selective Scan Mechanism to capture long-term market dependencies without the quadratic computational cost of standard attention mechanisms.
* **Qualitative Stream:** News headlines are processed by **FinBERT** to generate a bounded daily sentiment score $[-1, 1]$. This score serves as a dynamic covariate, acting alongside structural price data to evaluate how different architectures weight narrative sentiment during high-volatility periods.

---

## 🚀 Installation

### Prerequisites

* Python 3.9+
* CUDA-enabled GPU (Highly recommended for Mamba training)

### 1. Clone the Repository

```bash
git clone [https://github.com/SL8Entropy/Evaluating-State-Space-Models-and-LLM-Sentiment-in-Financial-Time-Series.git](https://github.com/SL8Entropy/Evaluating-State-Space-Models-and-LLM-Sentiment-in-Financial-Time-Series.git)
cd Evaluating-State-Space-Models-and-LLM-Sentiment-in-Financial-Time-Series

```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust version as needed)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install Mamba and other requirements
pip install packaging
pip install mamba-ssm causal-conv1d
pip install -r requirements.txt

```

---

## 💻 Usage

### 1. Data Preparation

This project requires two datasets:

* **News Data:** S&P 500 News Headlines (or a similar financial news corpus).
* **Price Data:** OHLCV data and technical indicators for target indices (e.g. S&P 500).

Run the preprocessing script to tokenise news and generate daily sentiment scores:

```bash
python scripts/preprocess_finbert.py --input data/raw_news.csv --output data/sentiment_scores.csv

```

### 2. Training and Testing

To train and evaluate the models on your dataset:

```bash
python main.py --model samba --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 5

```

**Arguments:**

* `--model`: Choose which model to train: `samba` (adaptive graph state-space), `lstm` (recurrent baseline), `mambastock` (pure Mamba baseline), or `dlinear` (simple linear baseline). Default is `samba`.
* `--dataset`: The file path to your target dataset CSV (e.g., `Dataset/sp500_with_indicators.csv`).
* `--num_features`: The number of input parameters/features in your dataset. If not provided, it will attempt to default to the dataset's configuration.
* `--seed`: Random seed for reproducibility. Overrides the default seed in the configuration file.

---

## 📊 Performance Metrics

The implementation tracks four key metrics:

* **IC (Information Coefficient):** Pearson correlation between predicted and actual returns.
* **RIC (Rank IC):** Spearman rank correlation (robust to outliers).
* **MAE:** Mean Absolute Error.
* **RMSE:** Root Mean Square Error.

Refer to the `results/` directory for detailed logs, SHAP summary plots, and visualisations comparing the hybrid models across distinct volatility regimes.

---

## 📂 Project Structure

SAMBA-FINBERT/
|   abc.PNG
|   main.py
|   paper_config.py
|   README.md
|   requirements.txt
|   runall.bat
|   test_system.py
|
+---config
|   |   model_config.py
|   |   __init__.py
|   |
|   \---__pycache__
|           model_config.cpython-312.pyc
|           __init__.cpython-312.pyc
|
+---Dataset
|       analyze_sentiment.py
|       calculateIndicators.py
|       create_price_graph.py
|       daily_sentiment_clean.csv
|       merge_sentiment_data.py
|       sp500_headlines_2008_2024.csv
|       sp500_index.csv
|       sp500_index_plot.png
|       sp500_with_indicators.csv
|       sp500_with_indicators_llm.csv
|
+---models
|   |   graph_layers.py
|   |   lstm.py
|   |   mamba.py
|   |   mambaStock.py
|   |   DLinear.py
|   |   mamba_block.py
|   |   normalization.py
|   |   samba.py
|   |   __init__.py
|   |
|   \---__pycache__
|           graph_layers.cpython-312.pyc
|           lstm.cpython-312.pyc
|           mamba.cpython-312.pyc
|           mamba_block.cpython-312.pyc
|           normalization.cpython-312.pyc
|           samba.cpython-312.pyc
|           __init__.cpython-312.pyc
|
+---results
|       create_graphs.py
|       look.py
+---trainer
|   |   trainer.py
|   |   __init__.py
|   |
|   \---__pycache__
|           trainer.cpython-312.pyc
|           __init__.cpython-312.pyc
|
+---utils
|   |   data_utils.py
|   |   logger.py
|   |   metrics.py
|   |   model_utils.py
|   |   __init__.py
|   |
|   \---__pycache__
|           data_utils.cpython-312.pyc
|           logger.cpython-312.pyc
|           metrics.cpython-312.pyc
|           model_utils.cpython-312.pyc
|           __init__.cpython-312.pyc
|

```

---

## 📄 Citation

If you use this codebase or find our framework helpful in your research, please cite our paper:

@article{sambathkumar2025evaluating,
  title={Evaluating State-Space Models and LLM Sentiment in Financial Time-Series: A SHAP and Regime-Based Study},
  author={Sambathkumar, Sudharshan and Abirami, K.},
  year={2025},
  journal={arXiv preprint arXiv:24XX.XXXXX}
}

```

```

```
