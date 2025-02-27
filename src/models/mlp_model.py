import torch
import torch.nn as nn
import torch.optim as optim
import optuna  # Import Optuna for hyperparameter tuning
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import sys
import nltk
import ssl
import matplotlib.pyplot as plt

# Fix SSL issue for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Fix import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import preprocessing & feature extraction
from src.data.preprocess import Preprocessing
from src.data.feature_extraction import FeatureExtraction

# Step 1: Load and preprocess the data
pre_proc = Preprocessing()
df = pre_proc.read_CSV('test.csv')

# Ensure the column names exist
if 'text' not in df.columns or 'sentiment' not in df.columns:
    raise ValueError("Error: Columns 'text' or 'sentiment' not found in CSV file.")

# Preprocess text
df['text'] = df['text'].fillna("")  # Ensure no NaN values
texts = df['text'].apply(pre_proc.preprocess).values
labels = df['sentiment'].values  # Labels

# Convert Labels to Numeric
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Feature extraction using TF-IDF
feature_extractor = FeatureExtraction()
X_tfidf = feature_extractor.tfidf_vectorize(texts)

# Split dataset
X_train, X_val, X_test, y_train, y_val, y_test = feature_extractor.split_dataset(X_tfidf, labels)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the Complex MLP model
class ComplexMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(ComplexMLP, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout3 = nn.Dropout(p=dropout_rate)

        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x

# Define Optuna Optimization Function

input_dim = X_train_tensor.shape[1]  # Number of features
hidden_dim = 256  # Increased hidden dimension
output_dim = len(np.unique(y_train))  # Number of classes


def objective(trial):
    # Suggest hyperparameters
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)  # Dropout range: 0.2 - 0.5
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)   # L2 Regularization range
    
    # Initialize model
    model = ComplexMLP(input_dim, hidden_dim=256, output_dim=len(np.unique(y_train)), dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_reg)
    
    # Train model
    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Validation Step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        _, predicted_val = torch.max(val_outputs, 1)
        val_accuracy = accuracy_score(y_val_tensor.numpy(), predicted_val.numpy())

    return val_accuracy  # Optuna tries to maximize validation accuracy

# Run Optuna Hyperparameter Optimization
study = optuna.create_study(direction="maximize")  # We want to maximize validation accuracy
study.optimize(objective, n_trials=20)  # Run 20 trials

# Print best hyperparameters
best_hyperparams = study.best_params
print(f"Best Hyperparameters: {best_hyperparams}")

# Train Best Model on Full Dataset
print(f"\nTraining Best Model with Dropout: {best_hyperparams['dropout_rate']}, L2 Regularization: {best_hyperparams['l2_reg']}")
best_model = ComplexMLP(input_dim, hidden_dim=256, output_dim=len(np.unique(y_train)), dropout_rate=best_hyperparams['dropout_rate'])
criterion = nn.CrossEntropyLoss()
best_optimizer = optim.Adam(best_model.parameters(), lr=0.001, weight_decay=best_hyperparams['l2_reg'])

# Train final model
epochs = 15
for epoch in range(epochs):
    best_model.train()
    best_optimizer.zero_grad()
    outputs = best_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    best_optimizer.step()

# Final Evaluation on Test Set
best_model.eval()
with torch.no_grad():
    test_outputs = best_model(X_test_tensor)
    _, predicted_test = torch.max(test_outputs, 1)
    test_accuracy = accuracy_score(y_test_tensor.numpy(), predicted_test.numpy())

print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
