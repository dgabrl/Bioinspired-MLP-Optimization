"""
MLP Manual Architecture Exploration (2024)
------------------------------------------
This script represents an empirical approach to defining a Multi-Layer Perceptron.
It was used as a baseline to compare manual tuning against automated heuristic searches.

Key features:
- Data normalization using MinMaxScaler.
- Fixed SGD solver with adaptive learning rate.
"""

# ==========================================
# 1. LIBRARIES AND DEPENDENCIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

# ==========================================
# 2. DATA LOADING AND CLEANING
# ==========================================
# Loading data from Google Drive (Environment setup)
from google.colab import drive
drive.mount('/content/drive')
data = pd.read_csv('/content/drive/MyDrive/Dados/dados.csv')

df = pd.DataFrame(data)
df_cleaned = df.dropna()

# ==========================================
# 3. FEATURE SELECTION AND PREPROCESSING
# ==========================================
# Defining target (class) and features
y = df_cleaned[['class']].values
x = df_cleaned.iloc[:, 2:].values

# Class Balance Analysis
# Checking for class imbalance to ensure robust model training
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique, counts))

total_samples = class_counts.get(0, 0) + class_counts.get(1, 0)
print(f"Class 0 percentage: {100 * class_counts.get(0, 0) / total_samples:.2f}%")
print(f"Class 1 percentage: {100 * class_counts.get(1, 0) / total_samples:.2f}%")

# Data Standardization using MinMaxScaler (Scaling features to 0-1 range)
scale = MinMaxScaler()
x_norm = scale.fit_transform(x)

# Splitting dataset into training and testing sets
x_norm_train, x_norm_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2,random_state=42)

# ==========================================
# 4. MODEL ARCHITECTURE AND TRAINING
# ==========================================
# Manual configuration of the Multi-Layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(8),
                    learning_rate_init= 0.1,
                    max_iter = 2000,
                    activation = 'relu',
                    momentum = 0.9,
                    solver = 'sgd',
                    learning_rate='adaptive',
                    random_state = 42)
mlp.fit(x_norm_train, y_train)

# ==========================================
# 5. PERFORMANCE EVALUATION
# ==========================================
y_pred = rna.predict(x_norm_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")
