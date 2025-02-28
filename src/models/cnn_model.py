import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import sys
import nltk
import ssl
import matplotlib.pyplot as plt

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

# Load and preprocess the data
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

# Convert Text to Sequences using TF-IDF
feature_extractor = FeatureExtraction()
X_tfidf = feature_extractor.tfidf_vectorize(texts)

# Split dataset
X_train, X_val, X_test, y_train, y_val, y_test = feature_extractor.split_dataset(X_tfidf, labels)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32).unsqueeze(1).unsqueeze(-1)[:500]
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32).unsqueeze(1).unsqueeze(-1)[:500]
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32).unsqueeze(1).unsqueeze(-1)[:500]

y_train_tensor = torch.tensor(y_train, dtype=torch.long)[:500]
y_val_tensor = torch.tensor(y_val, dtype=torch.long)[:500]
y_test_tensor = torch.tensor(y_test, dtype=torch.long)[:500]

# Create DataLoaders for Batch Training
batch_size = 2
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class TextCNN(nn.Module):
    def __init__(self, input_dim, num_classes, num_filters, dropout_rate, kernel_sizes):
        super(TextCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=(kernel_sizes[0], 1), stride=1, padding=(1, 0))
        self.conv2 = nn.Conv2d(1, num_filters, kernel_size=(kernel_sizes[1], 1), stride=1, padding=(2, 0))
        self.conv3 = nn.Conv2d(1, num_filters, kernel_size=(kernel_sizes[2], 1), stride=1, padding=(2, 0))

        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(num_filters * 3, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Ensure input is (batch_size, 1, seq_length, 1)
        x = x.unsqueeze(1) if x.dim() == 3 else x

        x1 = self.global_max_pool(self.relu(self.conv1(x))).squeeze(3).squeeze(2)
        x2 = self.global_max_pool(self.relu(self.conv2(x))).squeeze(3).squeeze(2)
        x3 = self.global_max_pool(self.relu(self.conv3(x))).squeeze(3).squeeze(2)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x



    

# Model Parameters
input_dim = X_train_tensor.shape[2]  # Number of features
num_classes = len(np.unique(y_train))

# Define Optuna Optimization Function
def objective(trial):
    # Suggest hyperparameters
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)
    num_filters = trial.suggest_int("num_filters", 64, 256)
    kernel_sizes = [trial.suggest_int(f"kernel_size_{i}", 2, 5) for i in range(3)]

    # Initialize model
    model = TextCNN(input_dim, num_classes=len(np.unique(y_train)), num_filters=num_filters, dropout_rate=dropout_rate, kernel_sizes=kernel_sizes)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_reg)
    criterion = nn.CrossEntropyLoss()

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
study.optimize(objective, n_trials=20, timeout = 600)  # Run 20 trials

# Print best hyperparameters
best_hyperparams = study.best_params
print(f"Best Hyperparameters: {best_hyperparams}")

# Train Best Model on Full Dataset
print(f"\nTraining Best Model with {best_hyperparams}")
best_model = TextCNN(input_dim, num_classes=len(np.unique(y_train)), num_filters=best_hyperparams['num_filters'], dropout_rate=best_hyperparams['dropout_rate'], kernel_sizes=[best_hyperparams[f'kernel_size_{i}'] for i in range(3)])
best_optimizer = optim.Adam(best_model.parameters(), lr=0.001, weight_decay=best_hyperparams['l2_reg'])
criterion = nn.CrossEntropyLoss()

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
