import sys
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import os
import cv2


# Function to load the model weights
def load_model_weights(model_path):
    my_model = tf.keras.models.load_model(model_path)
    my_model.summary()
    return my_model


# Function to process images and their corresponding labels
def get_images_labels(df, classes, img_height, img_width):
    ################### Modified ########################
    test_images = []
    test_labels = []

    # Create a mapping from class names to indices
    class_to_index = {cls.strip(): idx for idx, cls in enumerate(classes)}

    # Loop through the rows of the dataframe to load images and labels
    for index, row in df.iterrows():
        img_path = row['image_path']
        label = row['label'].strip()  # Strip any extra spaces from the label

        # Read and decode the image
        img = tf.io.read_file(img_path)
        img = decode_img(img, img_height, img_width)

        # Append the image and the label (converted to integer)
        test_images.append(img)
        test_labels.append(class_to_index[label])

    # Convert lists to numpy arrays
    test_images = np.stack(test_images)
    test_labels = np.array(test_labels)
    ####################################################
    return test_images, test_labels


# Function to decode and resize the image
def decode_img(img, img_height, img_width):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


if __name__ == "__main__":
    # Argument parser to accept command-line arguments
    parser = argparse.ArgumentParser(description="Transfer Learning Test")
    parser.add_argument('--model', type=str, default='my_model.h5', help='Saved model')
    parser.add_argument('--weights', type=str, default=None, help='weight file if needed')
    parser.add_argument('--test_csv', type=str, default='flowers_test.csv', help='CSV file with true labels')

    args = parser.parse_args()
    model = args.model
    weights = args.weights
    test_csv = args.test_csv

    # Load the test CSV and inspect columns
    test_df = pd.read_csv(test_csv)

    # Check for column names and strip any leading/trailing whitespaces
    test_df.columns = test_df.columns.str.strip()
    print("Columns in the CSV:", test_df.columns)

    # Classes: List of all possible flower classes in your dataset
    classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy',
               'carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose',
               'sunflower', 'tulip']

    # Load images and labels using the get_images_labels function
    test_images, test_labels = get_images_labels(test_df, classes, 224, 224)

    # Load the model
    my_model = load_model_weights(model)

    # Evaluate the model on the test set
    loss, acc = my_model.evaluate(test_images, test_labels, verbose=2)
    print('Test model accuracy: {:5.5f}%'.format(100 * acc))
