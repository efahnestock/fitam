import torch
import torch.nn as nn
import copy
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.v2 as v2

def preprocess_image_pyramid_batch(batch, device="cpu"):
    transforms = v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    patches = torch.from_numpy(np.array(batch)).permute(0, 3, 1, 2).to(device) / 255.0
    return transforms(patches)


class H5ImagePatchDataset(Dataset):
    def __init__(self, h5_path):
        self.file = h5py.File(h5_path, 'r')
        self.labels = torch.from_numpy(np.array(self.file['labels']).astype(int)).long()
        self.patches = preprocess_image_pyramid_batch(self.file['patches'])

    def __getitem__(self, idx):

        return self.patches[idx], self.labels[idx]

    def __len__(self):
        return len(self.patches)

    def close(self):
        self.file.close()


class SpatialLabelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(3, 16, 6, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, 5),
            nn.Flatten()
        ])

        self.linear = nn.LazyLinear(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x 
    

def load_spatial_label_model(model_path):
    params = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
    model = SpatialLabelModel()
    model.load_state_dict(params)
    return model




def evaluate(model, dataloader, criterion, device="cuda"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device); labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_spatial_label_model(
    file_path,
    train_split=0.8,
    epochs=100,
    batch_size=512,
    lr=0.001,
    patience=6,
    device="cuda",
    num_workers = 15
):
    dataset = H5ImagePatchDataset(file_path)
    
    train_subset, val_subset = random_split(
        dataset,
        [train_split, 1-train_split],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SpatialLabelModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Accuracy:   {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = copy.deepcopy(model)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    dataset.close()
    return best_model




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("model_save_path", type=str)

    args = parser.parse_args()
    best_model = train_spatial_label_model(args.dataset_path)

    torch.save(best_model.state_dict(), args.model_save_path)

