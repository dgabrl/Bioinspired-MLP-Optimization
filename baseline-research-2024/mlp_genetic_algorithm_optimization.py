"""
MLP Hyperparameter Optimization via Genetic Algorithms (2024)
-----------------------------------------------------------
This script automates the search for the optimal Multi-Layer Perceptron 
configuration using Evolutionary Strategies.

Methodology:
- Library: sklearn-genetic-opt
- Search Space: Learning Rate, Momentum, Activation, and Hidden Layers.
- Strategy: eaMuPlusLambda (Evolutionary Selection).
"""

# ==========================================
# 1. LIBRARIES AND EVOLUTIONARY TOOLS
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Specific tools for Genetic Optimization
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
from sklearn.model_selection import StratifiedKFold

# ==========================================
# 2. DATA LOADING AND CLEANING
# ==========================================
# Note: Path adjusted for local execution (ensure dados.csv is in the folder)
data = pd.read_csv('dados.csv')
df = pd.DataFrame(data)
df_cleaned = df.dropna()

# Defining target (class) and features
y = df_cleaned[['class']].values
x = df_cleaned.iloc[:, 2:].values

# ==========================================
# 4. PREPROCESSING AND STANDARDIZATION
# ==========================================
scale = MinMaxScaler()
x_norm = scale.fit_transform(x)

x_norm_train, x_norm_test, y_train, y_test = train_test_split(
    x_norm, y, test_size=0.2, random_state=42
)

# ==========================================
# 5. GENETIC SEARCH SPACE DEFINITION
# ==========================================
# Defining the base MLP model for optimization
mlp = MLPClassifier(max_iter=2000, random_state=42)

# Defining the architecture and hyperparameter boundaries
param_grid = {
    'hidden_layer_sizes': Categorical([
        (10,), (20,), (40,), (80,), (100,),
        (10, 10), (20, 20), (40, 40), (80, 80), (100, 100),
        (10, 10, 10), (20, 20, 20), (40, 40, 40), (80, 80, 80), (100, 100, 100),
    ]),
    'learning_rate_init': Continuous(0.001, 0.1),
    'momentum': Continuous(0.1, 0.9),
    'activation': Categorical(['logistic', 'tanh', 'relu'])
}

# ==========================================
# 6. EVOLUTIONARY SEARCH EXECUTION
# ==========================================
# Running GASearchCV with 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True)

ga_estimator = GASearchCV(estimator=mlp,
                               cv=cv,
                               scoring='accuracy',
                               population_size=15,
                               generations=50,
                               tournament_size=3,
                               elitism=True,
                               crossover_probability=0.6,
                               mutation_probability=0.1,
                               param_grid=param_grid,
                               criteria='max',
                               algorithm='eaMuPlusLambda',
                               n_jobs=-1,
                               verbose=True,
                               keep_top_k=4)

# ==========================================
# 7. EXECUTION AND RESULTS
# ==========================================
ga_estimator.fit(x_norm_train,y_train)

# Best configuration found during evolution
# Best configuration found during evolution
print(f"Best Parameters: {ga_estimator.best_params_}")

y_pred_ga = ga_estimator.predict(x_norm_test)
final_acc = accuracy_score(y_test, y_pred_ga)
print(f"Final Optimized Accuracy: {final_acc}")

# Visualization of the evolutionary progress
plot_fitness_evolution(ga_estimator)
plt.show()
