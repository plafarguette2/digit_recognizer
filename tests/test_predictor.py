"""
Script to test the predictor.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.preprocessor import Preprocessor
from app.predictor import Predictor

def test_predictor_returns_digit():
    """
    Test that the Predictor correctly returns an integer digit between 0 and 9.
        1. Create a fake 28x28 image as a NumPy array.
        2. Preprocess the image.
        3. Load the trained model with Predictor.
        4. Predict the digit.
        5. Assert that the output is an integer within the valid range (0-9).

    Raises:
        AssertionError: If the prediction is not an integer or is out of range.
    """
    fake_image = np.zeros((28, 28), dtype=np.uint8)
    
    preprocessor = Preprocessor()
    tensor = preprocessor.preprocess(fake_image)
    
    predictor = Predictor("model/model_weights.pt", device="cpu")
    pred = predictor.predict(tensor)
    
    assert isinstance(pred, int), f"Prediction is not int: {type(pred)}"
    assert 0 <= pred <= 9, f"Prediction out of range: {pred}"
