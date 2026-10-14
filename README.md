# Data-Classification-Using-AI-supervised-learning-the-Iris-dataset-model-training-and-testing.
# Iris Flower Classification Using Machine Learning 🌸

## 📌 Project Overview

This project was developed as **Project 2: Data Classification Using AI** as part of my Artificial Intelligence internship at DecodeLabs.

The objective is to build a basic machine learning classification model that learns from flower measurements and predicts the species of an Iris flower.

The project uses the Iris dataset provided by Scikit-learn and implements the **K-Nearest Neighbors (KNN)** algorithm to classify flowers into three species: Setosa, Versicolor, and Virginica.

## 🎯 Project Objectives

- Load and understand the Iris dataset.
- Split the dataset into training and testing sets.
- Apply feature scaling to standardize the measurements.
- Train a K-Nearest Neighbors classification model.
- Evaluate model performance using a confusion matrix and weighted F1 score.
- Visualize classification results and feature relationships.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Scikit-learn** — Dataset loading, preprocessing, model training, and evaluation
- **Pandas** — Data handling
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization

## 📊 Dataset Information

The Iris dataset contains 150 flower samples, four numerical features, and three target classes.

### Features

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

### Target Classes

- Setosa
- Versicolor
- Virginica

The dataset is loaded directly from `sklearn.datasets` using the `load_iris()` function, so a separate dataset file is not required.

## ⚙️ Project Workflow

### 1. Load the Dataset

The Iris dataset is loaded using Scikit-learn. The feature data (`X`) contains flower measurements, while the target data (`y`) contains the numerical species labels.

### 2. Split the Dataset

The dataset is divided into:

- **80% training data** — Used to train the model.
- **20% testing data** — Used to evaluate the model on unseen samples.

The `random_state=42` parameter makes the split reproducible.

### 3. Feature Scaling

`StandardScaler` standardizes the features before training the model.

The scaler is fitted only on the training data and then applied to the testing data to avoid data leakage.

### 4. Train the Classification Model

The project uses the **K-Nearest Neighbors (KNN)** algorithm with `n_neighbors=5`.

The model predicts a flower's species based on the labels of its five nearest neighbors in the scaled feature space.

### 5. Evaluate the Model

The model's predictions are evaluated using:

- **Confusion Matrix:** Shows the number of correct and incorrect predictions for each species.
- **Weighted F1 Score:** Measures the balance between precision and recall while accounting for the number of samples in each class.

### 6. Visualize the Results

Two visualizations are generated:

- **Confusion Matrix Heatmap:** Compares the actual flower species with the predicted species.
- **Petal Length vs. Petal Width Scatter Plot:** Shows the distribution of flower species based on their petal measurements.

##

## 🚀 Getting Started

### Prerequisites

Install Python 3 and Git on your computer.

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Navigate to the Project Folder

```bash
cd iris-flower-classification
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Project

```bash
python main.py
```

The terminal displays the dataset sample counts, model training status, confusion matrix, and weighted F1 score. Two visualization windows display the evaluation results.

## 📈 Results

The program reports the confusion matrix and weighted F1 score after making predictions on the test dataset.

The exact evaluation results are generated when the program runs. The confusion matrix heatmap and scatter plot provide additional insight into the model's predictions and the distribution of the Iris species.

## 📚 Key Learnings

Through this project, I practiced:

- Understanding a dataset and separating features from target labels.
- Splitting data into training and testing sets.
- Applying feature scaling correctly.
- Training a supervised machine learning classification model.
- Making predictions on unseen data.
- Evaluating classification performance using a confusion matrix and weighted F1 score.
- Visualizing data and model evaluation results.

## 👨‍💻 Internship Project

**Artificial Intelligence Internship — DecodeLabs**

**Project 2:** Data Classification Using AI

This project strengthened my understanding of supervised learning and the fundamental machine learning workflow.

---

⭐ If you find this project useful, feel free to explore the code and learn from it.
