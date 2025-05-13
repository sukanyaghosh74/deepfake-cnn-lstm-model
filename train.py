import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import DeepfakeDataset
from cnn_lstm_model import DeepfakeDetector

# Example data
video_paths = ["data/real1.mp4", "data/fake1.mp4"]  # to be updated with real paths
labels = [0, 1]  # 0 = real, 1 = fake

dataset = DeepfakeDataset(video_paths, labels)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    total_loss = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
