import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 1. Load the PRO Brain 🧠
# We are loading the Transfer Learning model this time.
print("Waking up the GENIUS robot (MobileNetV2)...")
try:
    model = tf.keras.models.load_model('my_fruit_model_pro.keras')
    print("Robot is ready!")
except:
    print("❌ Error: Could not find 'my_fruit_model_pro.keras'. Did you run the training script?")
    exit()

# Define the labels (Alphabetical order)
class_names = ['Apple', 'Banana', 'Orange']

print("------------------------------------------------")

# 2. The Input Function ⌨️
while True:
    print("\nSelect a fruit image to test!")
    img_path = input("Attach Your Fruit (Paste the file path here, or type 'exit'): ")

    if img_path.lower() == 'exit':
        break

    # Clean up the path string
    img_path = img_path.strip('"').strip("'")

    try:
        # 3. Pre-process the Image 🖼️
        # ⚠️ CRITICAL CHANGE: We must use (160, 160) because MobileNet is bigger!
        img = image.load_img(img_path, target_size=(160, 160))
        
        # Convert to array (0-255)
        # Note: We don't need manual math here because we put a Rescaling layer 
        # inside the model itself!
        img_array = image.img_to_array(img)
        
        # Add the fake batch dimension: (160, 160, 3) -> (1, 160, 160, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # 4. Make the Prediction 🔮
        predictions = model.predict(img_array)
        
        # Get the confidence score
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)

        print(f"I am {confidence:.2f}% sure this is a... {predicted_class.upper()}! 🍎🍌🍊")
    
    except Exception as e:
        print(f"Oops! Couldn't read that file. Error: {e}")