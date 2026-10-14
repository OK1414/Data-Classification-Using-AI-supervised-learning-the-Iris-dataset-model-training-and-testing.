from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Load the dataset
iris = load_iris()

# X represents our features (Sepal/Petal length and width)
X = iris.data

# y represents our target answers (Setosa, Versicolor, Virginica)
y = iris.target

# 2. STRUCTURAL INTEGRITY: The Split
# We split the data into 80% for training (pattern recognition) and 20% for testing (validation)
# random_state=42 shuffles the data to remove order bias before splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Total dataset samples: {len(X)}")
print(f"Training samples (80%): {len(X_train)}")
print(f"Testing samples (20%): {len(X_test)}")

# 3. THE GATEKEEPER RULE: Feature Scaling
scaler = StandardScaler()
# We fit the scaler to the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
# We ONLY transform the test data (do not fit it, to prevent data leakage)
X_test_scaled = scaler.transform(X_test) 

# 4. THE PROCESS: Train the K-Nearest Neighbors Algorithm
model = KNeighborsClassifier(n_neighbors=5)
# This is where the machine actually learns the patterns
model.fit(X_train_scaled, y_train) 

print("Data successfully scaled.")
print("Machine learning model trained successfully.")


# 5. THE OUTPUT: Make predictions on the unseen 20% test data
predictions = model.predict(X_test_scaled)

# 6. OUTPUT VALIDATION: Confusion Matrix & F1 Score
conf_matrix = confusion_matrix(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')


print("Confusion Matrix:\n", conf_matrix)
print(f"Weighted F1 Score: {f1 * 100:.2f}%")

# Graph 1: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix: Iris Classification (KNN)', fontsize=14)
plt.ylabel('Actual Species', fontsize=12)
plt.xlabel('Predicted Species', fontsize=12)
plt.tight_layout()
plt.show() 

# Graph 2: Feature Distribution (Petal Length vs Width)
# Convert raw data into a DataFrame for easier plotting
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = [iris.target_names[i] for i in y]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', 
                hue='Species', palette='viridis', s=100)
plt.title('Iris Dataset: Petal Length vs Petal Width', fontsize=14)
plt.tight_layout()
plt.show() 
