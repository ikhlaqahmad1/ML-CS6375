"""
This is the test file for model. Just change the path for model_path
"""
import tensorflow as tf
import os
from glob import glob

# Load the trained model
model_path = 'web-interface/deepfake_detector_model.h5'
#model_path = Change the path ^


model = tf.keras.models.load_model(model_path)

# Function to preprocess a single image
def preprocess_image(file_path):
    try:
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Read image with 3 channels (RGB)
        image = tf.image.resize(image, [224, 224])       # Resize image to match input shape
        image = image / 255.0                            # Normalize pixel values to [0, 1]
        return image
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return None

# Function to predict whether images are real or fake
def predict_images(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Get list of image files
    image_paths = glob(f"{folder_path}/*.jpg") + glob(f"{folder_path}/*.png") + glob(f"{folder_path}/*.jpeg")
    if not image_paths:
        print(f"No images found in folder '{folder_path}'.")
        return

    print(f"Found {len(image_paths)} images in '{folder_path}'.")

    for img_path in image_paths:
        image = preprocess_image(img_path)
        if image is not None:
            # Add batch dimension for model prediction
            image = tf.expand_dims(image, axis=0)
            prediction = model.predict(image)  # Get the sigmoid output score
            score = prediction[0][0]          # Extract the score from the prediction

            # Determine the class label based on the score
            label = "Real" if score < 0.5 else "Fake"

            # Print the prediction result
            print(f"Image: {img_path}")
            print(f"  Score: {score:.4f} -> Prediction: {label}")

# Specify the folder path containing images for testing
test_folder = 'test-data'

# Run the prediction
predict_images(test_folder)
