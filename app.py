import gradio as gr
import numpy as np
from tensorflow import keras
from keras.models import load_model

# Load the trained model
model = load_model('skin_model.h5')

# Define a function to make predictions
def predict(image):
    # Preprocess the image
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make a prediction using the model
    prediction = model.predict(image)
    
    # Get the sigmoid percentage
    sigmoid_percentage = prediction[0][0] * 100
    
    # Get the predicted class label
    if prediction[0][0] < 0.5:
        label = 'Benign'
    else:
        label = 'Malignant'
    
    return f"{label} ({sigmoid_percentage:.2f}%)"

examples = [["benign.jpg"], ["malignant.jpg"]]

# Define input and output components
image_input = gr.inputs.Image(shape=(150, 150))
label_output = gr.outputs.Label()

# Define a Gradio interface for user interaction
iface = gr.Interface(
    fn=predict,
    inputs=image_input,
    outputs=label_output,
    examples=examples,
    title="Skin Cancer Classification",
    description="Predicts whether a Skin Lesion is Cancerous or not.",
    theme="default",  # Choose a theme: "default", "compact", "huggingface"
    layout="vertical",  # Choose a layout: "vertical", "horizontal", "double"
    live=False 
)

