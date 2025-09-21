"""
Scritp to test preprocessor.
"""

import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.preprocessor import Preprocessor

def test_preprocessor_output_shape_and_range():
    """
    Test the Preprocessor output shape and value range.
        1. Create a fake 200x200 random image as a NumPy array.
        2. Preprocess the image using the Preprocessor to obtain a PyTorch tensor.
        3. Assert that the output tensor has the expected shape [1, 1, 28, 28].
        4. Assert that the tensor values are normalized between -1 and 1.

    Raises:
        AssertionError: If the tensor shape is incorrect or values are outside [-1,1].
    """

    fake_image = np.random.randint(0, 256, size=(300, 300), dtype=np.uint8)
    
    preprocessor = Preprocessor()
    tensor = preprocessor.preprocess(fake_image)
    
    assert tensor.shape == (1, 1, 28, 28), f"Expected (1,1,28,28), got {tensor.shape}"
    assert torch.min(tensor) >= -1.0 and torch.max(tensor) <= 1.0, "Values not in [-1,1]"
