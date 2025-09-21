"""
Script for loading a trained MNIST CNN model and getting predictions.
"""

import torch
import sys
import os

# Add project root to path in order that "model" can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model import my_model 


class Predictor:
    """
    Output predictions using a trained MNIST model.
    """

    def __init__(self, model_path, device=None):
        """
        Initialize the Predictor by loading a trained model.

        Args:
            model_path (str): Path to the saved model weights (the .pt file).
            device (torch.device): Device to use ("cuda" or "cpu").
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = my_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  

    def predict(self, tensor):
        """
        Run inference on a preprocessed input tensor.

        Args:
            tensor (torch.Tensor): Preprocessed image tensor of shape [1, 1, 28, 28].

        Return:
            int: Predicted digit (0–9).
        """
        tensor = tensor.to(self.device)
        with torch.no_grad():  # to avoid backpropagation
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).item()
        return pred