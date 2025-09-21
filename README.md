## Overview

This repository implements a **handwritten digit recognizer**.
It provides a **Streamlit interface** where the user can draw a digit to make the app predict it.
A `Dockerfile` is included to containerize the application.

The recognition model is a small **convolutional neural network** (CNN) (see `model/model.py` for more details on architecture) trained on the MNIST dataset.

This README describes the project requirements, how to run the Streamlit app locally (with or without Docker), and how the model was trained.

```arduino 
digit_recognizer
|   Dockerfile
|   README.md
|   requirements.txt
|               
+---app
|   |   predictor.py
|   |   preprocessor.py
|   |   streamlit_app.py
|                        
+---model
|   |   model.py
|   |   model_weights.pt
|   |   train.py
|           
\---tests
    |   test_model.py
    |   test_predictor.py
    |   test_preprocessor.py
```

## Requirements

For containerization, CPU-only versions of Torch and Torchvision are installed by default.
This choice was made because the model is lightweight and using the CPU-only versions significantly reduces the Docker image build time.

If you have a GPU and want to leverage it, you can modify the dependencies:

Uncomment the following lines in `requirements.txt`:
```python
#torch==2.8.0
#torchvision==0.23.0
```
And comment out the corresponding line in the `Dockerfile`:
```python
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you prefer to install dependencies directly on your machine (without Docker), run:
```bash
pip install -r requirements.txt
```

## Running the Streamlit app

**Option 1: Using Docker**

Clone this repository:
```bash 
git clone https://github.com/plafarguette2/digit_recognizer.git
cd digit_recognizer
```

Build the Docker image:
```bash
docker build -t digit_recognizer .
```

Run the container:
```bash
docker run -d -p 0.0.0.0:8080:8501 digit_recognizer
```

Open your browser and go to:

http://localhost:8080/

**Option 2: Run locally without Docker**

After cloning the repository, simply run:
```bash
streamlit run app/streamlit_app.py
```
## Training and data

The inference phase of the app uses pretrained weights stored in`model/model_weights.pt`.
Since the trained model is very small (< 2 MB), the weights are directly included in this repository.
The training script is available at `model/train.py`.

The per-trained model was trained for 5 epochs on the MNIST dataset (60,000 training images and 10,000 test images of handwritten digits). After 5 epochs the accuracy on validation set reached 99.2%. The train and test data are directly downloaded by the training script.

If you wish to retrain the model yourself, you can run the training script and overwrite the weights. Save the new weights under the same name (`model_weights.pt`) to be used by the app.