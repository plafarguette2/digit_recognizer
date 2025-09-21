"""
Training script for the handwritten digit recognition CNN model.

This module defines a `Trainer` class to manage training and validation 
loops : 
    1. Load MNIST dataset with normalization.
    2. Initialize the CNN model.
    3. Train the model using Adam optimizer and CrossEntropyLoss.
    4. Validate performance on test dataset after each epoch.
    5. Save the trained model weight to disk.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import my_model


class Trainer:
    """
    Class to train and validate the PyTorch model.
    """

    def __init__(self, model, device, lr=0.001):
        """
        Initialize the Trainer.

        Args:
            model (torch.nn.Module): Model to be trained.
            device (torch.device): Device to run computations on.
            lr (float): Learning rate for Adam optimizer.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_one_epoch(self, train_loader):
        """
        Run one training epoch over the dataset.

        Args:
            train_loader (DataLoader): DataLoader for training data.

        Returns:
            float: Average training loss for this epoch.
        """
        self.model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Evaluate the model on validation/test data.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            float: Validation accuracy in percentage.
        """
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def fit(self, train_loader, val_loader, epochs=5, save_path="model_weights.pt"):
        """
        Run the training loop for a defined number of epochs, with validation.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Number of epochs.
            save_path (str): Path to save model weights.
        """
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_acc = self.validate(val_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved at {save_path}")


def main():
    """
    Train the CNN model on the MNIST dataset.

        - Configure device.
        - Apply image transformations.
        - Load MNIST training and test sets.
        - Initialize model and Trainer.
        - Train the model and save weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = my_model()
    trainer = Trainer(model, device)

    trainer.fit(train_loader, test_loader, epochs=5, save_path="model/model_weights.pt")


if __name__ == "__main__":
    main()