# Sentiment Analysis: A Comparative Study of ML Models

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-scikit--learn%20%7C%20pandas%20%7C%20Matplotlib-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A project comparing Naive Bayes, Logistic Regression, and SVM for sentiment analysis on the DailyDialog dataset. The SVM, aided by data augmentation, was identified as the optimal model.

---

## Project Workflow

This flowchart illustrates the complete end-to-end process of the project, from data collection to final analysis.

> **Note:** To add your flowchart image here, upload the diagram to your GitHub repository (e.g., in a folder named `img/`) and replace the path below.

![Flow diagram](Flow.png)

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Methodology](#2-methodology)
3.  [Model Implementation](#3-model-implementation)
4.  [Results and Analysis](#4-results-and-analysis)
5.  [Conclusion](#5-conclusion)
6.  [Future Work](#6-future-work)
7.  [How to Run](#7-how-to-run)
8.  [Potential Applications](#8-potential-applications)

## 1. Introduction

### Background
Sentiment analysis, also known as opinion mining, is a field of Natural Language Processing (NLP) that involves identifying and categorizing the emotional tone expressed in a piece of text. Instead of focusing on product reviews, this project applies sentiment analysis to the more nuanced domain of everyday human conversation using the **DailyDialog dataset**. The primary goal is to determine the underlying attitude within a dialogue snippet, classifying it as **positive, negative, or neutral**.

### Motivation
The motivation for this project stems from the challenge of teaching a machine to understand the nuances of human emotion. The **DailyDialog dataset** provides a rich but complex source of text, where sentiments are expressed through emotions like 'joy,' 'sadness,' 'anger,' and 'fear'.

A primary challenge discovered during preprocessing was a significant **class imbalance**, with negative sentiments far outnumbering positive and neutral ones. This project was motivated by the need to compare different machine learning approaches—Naive Bayes, Logistic Regression, and Support Vector Machine—to build a robust model that could accurately classify sentiment while being resilient to the biasing effects of imbalanced data.

### Problem Statement
The problem is to develop a machine learning model that accurately classifies text from the DailyDialog dataset into **positive, negative, or neutral** categories. The core task is to compare the performance of the three classification algorithms and identify an optimal model that effectively overcomes the challenge of the imbalanced dataset.

## 2. Methodology

### Data Collection
The dataset used is the **DailyDialog dataset**, a corpus containing high-quality, multi-turn dialogues reflecting everyday human communication.

### Data Pre-processing
Data pre-processing was a foundational phase to transform the raw data into a clean, structured format suitable for machine learning.
* **Label Mapping:** The original, detailed emotion labels ('joy', 'sadness', 'anger', etc.) were consolidated into three primary sentiment categories: `positive`, `negative`, and `neutral`.
* **Text Cleaning:** All text was converted to lowercase, and all punctuation and special characters were removed.
* **Data Augmentation:** To address the class imbalance, a strategic data augmentation technique was employed for the SVM model. A dozen new, high-quality examples of positive, negative, and especially neutral sentences were manually added to the training data.

### Feature Extraction (TF-IDF)
To convert the cleaned text into a numerical format that models can understand, the **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization technique was used. TF-IDF assigns a weight to each word that reflects its importance in a sentence relative to the entire corpus, making it highly effective for text classification.

## 3. Model Implementation

Three different machine learning algorithms were implemented to compare their effectiveness.

* **Multinomial Naive Bayes:** A fast, probabilistic classifier commonly used as a baseline for text classification. It operates by calculating the probability of each word appearing in each sentiment class.
* **Logistic Regression:** A robust statistical model that learns a specific weight for each word, determining how much that word contributes to a sentence being classified as positive, negative, or neutral.
* **Support Vector Machine (SVM):** The optimal model for this project. A `LinearSVC` was used, which is highly efficient for high-dimensional text data. It works by finding the optimal hyperplane that best separates the different sentiment classes.

## 4. Results and Analysis

### Model Performance Comparison
The SVM model demonstrated the most notable success, achieving the highest accuracy and the most balanced performance across all classes.

> **Note:** To add your bar chart image here, upload it to your GitHub repository and replace the path below.

![Model comparison](Model%20Comparison.png)

### Comparative Metrics
The table below provides a side-by-side comparison of all evaluation metrics for the three models.

| Metric | Naive Bayes | Logistic Regression | **Support Vector Machine (SVM)** |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 72.24% | 80.63% | **80.73%** |
| **F1-Score** (Weighted Avg) | 69% | 80% | **80%** |
| **F1-Score (Negative)** | 0.82 | 0.87 | **0.88** |
| **F1-Score (Neutral)** | 0.43 | 0.71 | **0.71** |
| **F1-Score (Positive)** | 0.54 | 0.66 | **0.68** |

### Optimal Model: SVM (Classification Report)
The SVM's superior performance is further detailed in its classification report, showing a strong balance between precision and recall, especially for the challenging minority classes.

| | precision | recall | f1-score | support |
| :--- | :---: | :---: | :---: | :---: |
| **negative** | 0.85 | 0.91 | 0.88 | 1350 |
| **neutral** | 0.72 | 0.71 | 0.71 | 452 |
| **positive** | 0.77 | 0.61 | 0.68 | 466 |
| | | | | |
| **accuracy** | | | 0.81 | 2268 |
| **weighted avg**| 0.80 | 0.81 | 0.80 | 2268 |

## 5. Conclusion
The outcomes of this project demonstrate that strategic data handling is as critical as algorithm choice for effective sentiment analysis. The project successfully identified the **Support Vector Machine (SVM)**, enhanced with data augmentation, as the most optimal approach, achieving the highest accuracy of **80.73%**. The results prove the significant impact that a targeted strategy like data augmentation can have on improving a model's performance and its ability to generalize across all sentiment classes.

## 6. Future Work
While the SVM performed well, it sometimes struggled with complex sentences containing both positive and negative cues. Future work will be directed toward:

* **Exploring Advanced Models:** Implementing deep learning architectures like LSTMs or Transformers (e.g., BERT), which are designed to better understand the sequence and context of words.
* **Sophisticated Data Sampling:** Using automated over-sampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create a more balanced training dataset.
* **Model Generalizability:** Validating the final model on larger and more diverse conversational datasets to ensure its reliability.

## 7. How to Run
This project is contained within Jupyter Notebooks (`.ipynb` files).

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/YourUsername/Sentiment-Analysis-Project.git](https://github.com/YourUsername/Sentiment-Analysis-Project.git)
    ```
2.  **Install dependencies:**
    ```sh
    pip install pandas numpy scikit-learn matplotlib
    ```
3.  **Run the notebooks:**
    You can run the `NaiveBayes.ipynb`, `Logistic_Regression.ipynb`, and `SVM.ipynb` notebooks in any environment that supports Jupyter, such as VS Code or Google Colab. To run in Google Colab, simply upload the notebook and the `DailyDialog.csv` file.

## 8. Potential Applications
* **Customer Feedback Analysis:** Automatically analyze customer support chats and emails to gauge satisfaction and identify recurring issues.
* **Social Media Monitoring:** Track public sentiment towards brands or campaigns by analyzing conversations and comments.
* **Market Research:** Analyze online discussions and forums to understand consumer opinions and identify emerging trends.
* **Building Emotionally Aware AI:** Use these insights to develop more sophisticated and empathetic AI, such as emotionally aware chatbots.
