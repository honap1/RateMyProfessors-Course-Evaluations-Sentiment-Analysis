# RateMyProfessors-Course-Evaluations-Sentiment-Analysis
A binary sentiment classification of professor reviews using Traditional Machine Learning Learning and Deep Learning Approaches with Hyperparameter Optimization

---

## Project Overview
**Goal:** Classify RateMyProfessors reviews as positive (1) or negative (0) sentiment

**Target Variable:**
- Primary target: `student_star` (continuous rating 1.0–5.0)  
- Derived target: `sentiment` (binary: Negative ≤2.0, Positive ≥4.0)

**Dataset:** 20,000 sampled professor reviews from RateMyProfessors.com
- **Training:** 11,580 samples (70%)
- **Validation:** 2,482 samples (15%)
- **Test:** 2,483 samples (15%)
- **Class Distribution:** 71.3% positive, 28.7% negative

**Best Model Performance:** Baseline + Optuna
- **Accuracy:** 87.11%
- **F1-Score:** 90.64%
- **Precision:** 93.99%
- **Recall:** 87.52%

---

## Prerequisites
**Python 3.8+**

**Required libraries:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
torch>=1.10.0
optuna>=3.0.0
spacy>=3.2.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

## Installation

### Clone repository
```bash
git clone 
cd ratemyprofessors-sentiment-analysis
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download spaCy model
```bash
python -m spacy download en_core_web_sm
```
## Run Analysis

### Open Jupyter Notebook
```bash
jupyter notebook NLP_Capstone_Project_Code_Repository.ipynb
```

---

## Methodology
### A. Data Preprocessing Pipeline
**Tool:** spaCy `en_core_web_sm` model

**Pipeline:**
1. Lowercase normalization
2. Remove punctuation and special characters
3. **Lemmatization** using spaCy (e.g., "classes" → "class")
4. Remove English stopwords
5. Remove tokens ≤2 characters

---

### B. Model Implementations

#### **Baseline: TF-IDF + Logistic Regression**
- **Features:** TF-IDF (5,000 features, unigrams + bigrams)
- **Model:** Logistic Regression with balanced class weights
- **Performance (Manual):** 86.35% accuracy, 89.90% F1-score
- **Performance (Optuna):** 87.11% accuracy, 90.64% F1-score ⭐
- **Training Time:** ~30 seconds

#### **Model 2: MLP with TF-IDF**
- **Features:** TF-IDF (3,000 features)
- **Architecture:** 512 → 256 → 128 neurons
- **Regularization:** BatchNorm, Dropout (0.4)
- **Performance (Manual):** 85.14% accuracy, 89.17% F1-score
- **Performance (Optuna):** 85.86% accuracy, 89.66% F1-score
- **Training Time:** ~5 minutes

#### **Model 3: MLP with GloVe Embeddings**
- **Features:** GloVe 6B 100d (pre-trained word embeddings)
- **Architecture:** 128 → 64 neurons
- **Performance (Manual):** 83.85% accuracy, 88.43% F1-score
- **Performance (Optuna):** 83.61% accuracy, 88.31% F1-score
- **Training Time:** ~3 minutes

---

### C. Hyperparameter Optimization (Optuna)

**Method:** Bayesian optimization using Tree-structured Parzen Estimator (TPE)

**Parameters Tuned:**
- **Baseline:** max_features, ngram_range, min_df, max_df, C, max_iter
- **MLP-TF-IDF:** max_features, architecture (3 layers), dropout, learning rate, batch size
- **MLP-GloVe:** architecture (2 layers), dropout, learning rate, batch size

**Total Trials:** 110 (50 for Baseline, 30 for each MLP)

**Results:**
- **Baseline:** +0.76% F1-score improvement
- **MLP-TF-IDF:** +0.49% F1-score improvement
- **MLP-GloVe:** -0.13% F1-score (marginal decrease)

---

### D. Evaluation Metrics

- **Accuracy:** Overall correctness
- **Precision:** Of predicted positives, how many are correct
- **Recall:** Of actual positives, how many we found
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the ROC curve

---

### E. Interpretability Analysis

**Techniques Used:**
- Feature importance (coefficient analysis)
- SHAP values (Shapley Additive Explanations)
- Calibration curves (Expected Calibration Error < 0.05)
- Error analysis (false positives/negatives)

---

## Summary of Results
### Final Model Comparison (Test Set)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
|-------|---------------|----------------|-------------|---------|
| **Baseline (Optuna)**  | **87.11%** | **93.99%** | **87.52%** | **90.64%** |
| Baseline (Manual) | 86.35% | 95.15% | 85.21% | 89.90% |
| MLP-TF-IDF (Optuna) | 85.86% | 93.77% | 85.88% | 89.66% |
| MLP-TF-IDF (Manual) | 85.14% | 92.85% | 85.77% | 89.17% |
| MLP-GloVe (Manual) | 83.85% | 90.44% | 86.50% | 88.43% |
| MLP-GloVe (Optuna) | 83.61% | 89.84% | 86.84% | 88.31% |

---

## Findings:
1. **Baseline + Optuna achieved best results**
   - 87.11% accuracy, 90.64% F1-score
   - +0.76% improvement over manual tuning (significant)

2. **Traditional ML outperforms deep learning for this task**
   - Simple model + good features > complex models
   - 16K samples insufficient for deep learning to excel

3. **Feature engineering is crucial**
   - TF-IDF representation more important than model complexity
   - Good features enable simpler, faster, more interpretable models

4. **High precision across all models**
   - less than 89% precision ensures reliable positive predictions
   - Critical for real-world deployment

5. **Dataset size limitation**
   - ~16K samples too small for 1.6M parameter neural networks
   - Would need 100K+ samples for deep learning advantages

---


## References
### Libraries
- **spaCy:** Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. https://spacy.io/
- **scikit-learn:** Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR 12, pp. 2825-2830.
- **PyTorch:** Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS.
- **Optuna:** Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. KDD.
- **SHAP:** Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.

### Pre-trained Models
- **GloVe:** Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global vectors for word representation. EMNLP.
  - Model used: GloVe 6B (100-dimensional, trained on 6 billion tokens)
  - https://nlp.stanford.edu/projects/glove/

### Datasets
- **RateMyProfessors:** Professor review platform
  - Source: Kaggle dataset / RateMyProfessors.com
  - Collected: 2023-2024
  - Size: 16,545 reviews after cleaning
  - License: Educational use only

### Related Works
- Wen, M., et al. (2014). Sentiment Analysis in MOOC Discussion Forums. EDM.
- Jia, R., & Liang, P. (2017). Adversarial Examples for Evaluating Reading Comprehension Systems. EMNLP.
- Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL.

---

## Team
**Team Members:**
- Kimberly Venice Alon
- Aaron John Pacis
- Allyana Coleen Reyes
- Juliane Moira Roldan

**Contributions:** All members contributed equally to code review, testing, and presentation

---

## License
This project is for **academic purposes only**.

- **Code:** MIT License (team's original work)
- **Dataset:** RateMyProfessors.com (educational use only)
- **Pre-trained models:** GloVe (Public Domain)

**Disclaimer:** This project is a student capstone and not intended for commercial deployment. RateMyProfessors.com trademark belongs to its respective owners.

---

## Acknowledgments
- **Professor Dr. Felix Muga** for course guidance and mentorship
- **RateMyProfessors.com** for making review data available
- **Open-source community** for excellent libraries (spaCy, scikit-learn, PyTorch, Optuna)
- **Stanford NLP Group** for pre-trained GloVe embeddings
- **Kaggle community** for dataset curation

---


**Best Model:** Baseline (TF-IDF + Logistic Regression) with Optuna Optimization  
**Performance:** 87.11% Accuracy | 90.64% F1-Score | 93.99% Precision  
**Course:** MATH 103.1 - Natural Language Processing
