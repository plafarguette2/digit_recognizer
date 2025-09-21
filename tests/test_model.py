"""
Script to test model architecture.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from model.model import my_model

def test_model_forward_shape():
    """
    Test that my_model produces output of correct shape.
    
    Given a batch of 1x28x28 grayscale images, the output should be [batch size, 10].
    """
    model = my_model()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10), f"Expected shape (4,10), got {out.shape}"
