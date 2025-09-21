"""
Streamlit interface for handwritten digit recognition.

Workflow:
    1. Draw a digit on a canvas on the app interface.
    2. Preprocess the canvas image into a [1, 1, 28, 28] PyTorch tensor.
    3. Pass the tensor through the trained model to get a prediction.
    4. Display the predicted digit on the app interface.
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from preprocessor import Preprocessor
from predictor import Predictor

# Canvas size in pixels
size_canvas = 300


class StreamlitApp:
    """
    Streamlit application class for digit recognition.

    Attributes:
        preprocessor (Preprocessor): Manage preprocessing of canvas images.
        predictor (Predictor): Loads the trained model and makes predictions.
    """

    def __init__(self, model_path="model/model_weights.pt"):
        """
        Initialize the Streamlit application.

        Args:
            model_path (str): Path to the saved model weights.
        """
        self.preprocessor = Preprocessor()
        self.predictor = self.load_model(model_path)
    
    @st.cache_resource
    def load_model(_self, model_path):
        """
        Load the Predictor model once and cache it.

        Args:
        model_path (str): Path to the saved model weights.

        Return:
        Predictor: Predictor with loaded model.
        """
        return Predictor(model_path)

    def run(self):
        """
        Run the Streamlit application.

        Displays:
            - A drawing canvas for user input.
            - A "Predict" button that triggers preprocessing and prediction.
            - Prediction result or warning message.
        """
        st.title("Handwritten Digit Recognizer")

        st.markdown(
    "✏️ **Draw your digit inside the black square below using your mouse**, "
    "then press the button on the right to **read your digit**.")

        # Centered canvas
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            canvas_result = st_canvas(
                fill_color="#000000", 
                stroke_width=23,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=size_canvas,
                width=size_canvas,
                drawing_mode="freedraw",
                key="canvas"
            )

        # Prediction button
        with col3:
            if st.button("Read my digit"):
                if canvas_result.image_data is not None:
                    img = canvas_result.image_data[..., 0]  # take one channel
                    if img.max() == 0:
                        st.warning("Please draw a digit before predicting!")
                    else :
                        tensor = self.preprocessor.preprocess(img)
                        pred = self.predictor.predict(tensor)
                        st.success(f"I think you wrote a {pred} !")
                else:
                    st.warning("Please draw a digit before predicting!")


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()