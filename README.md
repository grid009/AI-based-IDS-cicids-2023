## AI-Based Intrusion Detection System (IDS) using CICIDS 2023

📌 **Overview**

This project aims to identify malicious network traffic using the CICIDS 2023 dataset. It involves training and testing multiple machine learning models on the Backdoor Malware subset, including Logistic Regression, Perceptron, Random Forest, and a Feedforward Neural Network (MLP). Model performance is assessed using metrics such as Accuracy, Precision, Recall, and F1-score.

📂 **Dataset**

Source: https://www.unb.ca/cic/datasets/iotdataset-2023.html

Contains both normal and malicious network traffic.

Attacks included: DoS, DDoS, Brute Force, Botnet, Backdoor, Web Attacks, Infiltration, etc

## Features
**Accurate Intrusion Detection:**

The system uses machine learning models to accurately detect various types of network intrusions (e.g., DDoS attacks, Brute Force attacks, etc.).

Achieves high accuracy and precision in distinguishing between normal and malicious traffic flows.

**Machine Learning Models:**

The system employs multiple machine learning algorithms, including:

**Random Forest**: A robust ensemble learning model for classification and regression tasks.

**Logistic Regression**: A simple yet effective model for binary classification tasks (normal vs. malicious).

**Perceptron**: A type of neural network used for classifying network traffic.

**False Positive Reduction:**

Isolation Forest: Used to reduce false positives in the system, improving the reliability of the detection system and reducing unnecessary alerts.

Enhances the system’s ability to classify normal traffic while minimizing false alarms correctly.

**Feature Selection with PCA (Principal Component Analysis):**

PCA is used for dimensionality reduction and feature selection to optimize model performance, focusing on the most relevant features of network traffic.

PCA improves the efficiency and speed of the IDS by reducing the number of features without losing critical information.

## Techniques
**Principal Component Analysis (PCA)**:

PCA is used for feature extraction and dimensionality reduction in the project. By transforming the high-dimensional dataset into a lower-dimensional space, PCA retains the most important features and removes redundant ones, improving model training and performance.

**Random Forest Classifier**:

A popular ensemble machine learning algorithm used for classification tasks. The Random Forest model aggregates predictions from multiple decision trees to improve accuracy and reduce overfitting.

**Logistic Regression**:

A statistical method used for binary classification. In this project, it is applied to classify network traffic as either normal or malicious.

**Isolation Forest Algorithm**:

Used for anomaly detection, particularly effective in identifying rare or outlying points in data (i.e., anomalies such as attacks). Isolation Forest helps reduce false positives by identifying and isolating anomalous network traffic patterns.

**Neural Networks (Perceptron)**:

A simple neural network model used for classification, particularly for binary outcomes (e.g., normal vs. malicious traffic).


## Evaluation

Accuracy, Precision, Recall, F1-score

ROC-AUC, Confusion Matrix

Deployment-Ready (planned)

Integrate into a real-time IDS pipeline

## Results

Random Forest achieved 97.67% accuracy

Isolation Forest integration reduced false positives by 15%

Models demonstrated strong performance in detecting a wide range of attack types

## Tech Stack

Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Environment: Jupyter Notebook / Google Colab

🚀 **Getting Started**
1. Clone the repository:
   ```bash
   git clone https://github.com/grid009/AI-based-IDS-cicids-2023.git
   cd AI-based-IDS-cicids-2023



