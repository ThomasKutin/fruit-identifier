import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 1. Load the Saved Brain 🧠
# We don't need to rebuild the model; just load the file we saved earlier.
print("Waking up the robot...")
model = tf.keras.models.load_model('my_fruit_model.keras')

# Define the labels (Must be in alphabetical order because that's how Keras learned them!)
class_names = ['Apple', 'Banana', 'Orange']

print("Robot is ready!")
print("------------------------------------------------")

# 2. The Input Function ⌨️
while True:
    print("\nSelect a fruit image to test!")
    img_path = input("Attach a picture of Your Fruit with a plain background (Paste the file path here, or type 'exit'): ")

    # Allow the user to quit
    if img_path.lower() == 'exit':
        break

    # Remove quotes if the user pasted them (common on Windows)
    img_path = img_path.strip('"').strip("'")

    try:
        # 3. Pre-process the Image 🖼️
        # We must treat this image exactly like the training images:
        # Resize it to 64x64 pixels
        img = image.load_img(img_path, target_size=(64, 64))
        
        # Convert the image pixels to a math array
        img_array = image.img_to_array(img)
        
        # Add a "fake" batch dimension. 
        # The model expects a list of photos, even if we only give it one.
        # It turns (64, 64, 3) into (1, 64, 64, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # 4. Make the Prediction 🔮
        predictions = model.predict(img_array)
        
        # predictions looks like: [[0.05, 0.90, 0.05]] -> 90% chance it's index 1
        score = tf.nn.softmax(predictions[0])
        
        # Find the highest score
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)

        print(f"I am {confidence:.2f}% sure this is a... {predicted_class.upper()}! 🍎🍌🍊")
    
    except Exception as e:
        print(f"Oops! Couldn't read that file. Error: {e}")