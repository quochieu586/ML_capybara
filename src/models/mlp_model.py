import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import os
import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))



from src.data.preprocess import Preprocessing
from src.data.feature_extraction import FeatureExtraction

pre_proc = Preprocessing()
df = pre_proc.read_CSV('test.csv')

texts = df['text'].apply(pre_proc.preprocess).values
labels = df['sentiment'].values

feature_extractor = FeatureExtraction()
X_tfidf = feature_extractor.tfidf_vectorize(texts)

X_train, X_val, X_test, y_train, y_val, y_test = feature_extractor.split_dataset(X_tfidf, labels)

X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class ComplexMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ComplexMLP, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout3 = nn.Dropout(p=0.3)

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

# Model Parameters
input_dim = X_train_tensor.shape[1]  # Number of features
hidden_dim = 256  # Increased hidden dimension
output_dim = len(np.unique(y_train))  # Number of classes

# Initialize the model, loss function, and optimizer
model = ComplexMLP(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the model
epochs = 15
for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 3 == 0:  # Print every 3 epochs
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Step 6: Evaluate the model
model.eval()

# Validation
val_outputs = model(X_val_tensor)
_, predicted = torch.max(val_outputs, 1)
val_accuracy = accuracy_score(y_val_tensor.numpy(), predicted.numpy())
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Test
test_outputs = model(X_test_tensor)
_, predicted = torch.max(test_outputs, 1)
test_accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
print(f"Test Accuracy: {test_accuracy:.4f}")
