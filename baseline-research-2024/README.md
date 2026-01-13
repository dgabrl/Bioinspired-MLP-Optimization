# Baseline Research (2024) - Bioinspired MLP Optimization

This folder documents the experimental phase of my research on Multi-Layer Perceptrons (MLP) for high-energy physics data classification. These scripts provided the foundational results and methodological testing that preceded my Master's Thesis.

## Data Processing Methodology
The dataset contains 28 low-level and high-level attributes used to distinguish between "background" and "signal" events.

* **Cleaning:** Rows with missing values (NaN) were removed. Given the large dataset size, this cleaning did not impact model representation.
* **Class Balance:** The distribution is naturally balanced (~46.87% background vs. 53.19% signal), requiring no additional resampling.
* **Normalization:** **Min-Max Scaling** was applied to input features to optimize gradient descent convergence.
* **Data Split:** Used the **Hold-out method** (80% training / 20% testing). Class proportions were verified to remain consistent across both subsets.



## Model Architecture & Heuristics

### 1. Manual Exploration (`mlp_manual_experiment.py`)
Architecture and training parameters were adjusted based on scientific literature and empirical analysis:

* **Neuron Heuristics:** The first hidden layer search space was defined by the rules:
    * $n = 2 \cdot n_{a} + 1$
    * $n = n_{c}$
    * Resulting in an analytical range between 2 and 57 neurons.
* **Architecture Depth:** Following *Krishnan (2021)*, we analyzed up to **three hidden layers**. While one layer is often sufficient, deeper architectures (up to 3 layers) were tested for comparison.
* **Layer Optimization:** We used a greedy approach—varying the number of neurons in the current layer (between the input size of 28 and output size of 1) while keeping previous layers constant.
* **Training Specs:** Stochastic Gradient Descent (SGD) for 2000 epochs with **Step Decay** for the learning rate.



### 2. Genetic Algorithm Optimization (`mlp_genetic_algorithm_optimization.py`)
To validate the manual results, we implemented an evolutionary search to find the optimal global configuration.
* **Library:** `sklearn-genetic-opt`.
* **Algorithm:** `eaMuPlusLambda` (Evolutionary Strategy).
* **Automated Search:** Simultaneously optimized activation functions (ReLU, Tanh, Logistic), momentum (0.5 to 0.99), and layer configurations.
* **Validation:** Used **Stratified 5-Fold Cross-Validation** to ensure the robustness of the discovered hyperparameters.



## How to Run (Execution Guide)
These scripts were developed and tested in **Google Colab**. To reproduce the results:

1. **Environment Setup:** Open the scripts in Google Colab and install the evolutionary optimization library:
   ```python
   !pip install sklearn-genetic-opt
   ```
2. **Database Configuration (Crucial):** Mount your Google Drive.
* **Path:** Update the `pd.read_csv()` path to point to your `dados.csv` file.
* **Features and Target:** Ensure you review the input (`x`) and output (`y`) definitions in the code. In these scripts, `y` is assigned to the first column (`class`) and `x` starts from the 3rd column onwards (`iloc[:, 2:]`). Adjust these indices if your dataset structure differs.

3. **Execution:** Run the cells sequentially.
