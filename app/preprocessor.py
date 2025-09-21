"""
Script to preprocess images to make them MNIST-compatible.
"""

import numpy as np
from PIL import Image
import torch

class Preprocessor:
    """
    Preprocess images for MNIST-trained CNN.

    Steps:
        1. Convert input to PIL Image.
        2. Convert to grayscale.
        3. Resize.
        4. Convert to numpy array.
        5. Normalize pixel values to [-1, 1].
        6. Convert to PyTorch tensor By adding batch and channel dimensions.
    """

    def __init__(self, image_size=(28, 28)):
        """
        Initialize the Preprocessor.

        Args:
            image_size (tuple): Desired image size.
        """
        self.image_size = image_size

    def preprocess(self, image):
        """
        Convert an input image into a PyTorch tensor ready for MNIST models.

        Args:
            image (np.ndarray): Input image.

        Returns:
            torch.tensor: Preprocessed image tensor of shape [1, 1, 28, 28].
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("L")  # grayscale
        image = image.resize(self.image_size)
        image = np.array(image)
        image = image / 255.0  # scaling to  0-1
        image = (image - 0.5) / 0.5  # scaling to (-1) - 1
        tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) #adding batch size and channel dim
        return tensor