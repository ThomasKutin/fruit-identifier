import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the Brain (Cached so it doesn't reload every time you click)
@st.cache_resource
def load_model():
    # Make sure this matches the name of your best model!
    return tf.keras.models.load_model('my_fruit_model_finetuned.keras')

model = load_model()

# Define our labels
class_names = ['Apple', 'Banana', 'Orange', 'Unrelated']

# 2. Build the Website UI 🖥️
st.title("🍎🍌🍊 Kutin's Fruit Identifier")
st.write("Upload a photo of a fruit, and I'll tell you what it is!")

# File Uploader Button
file = st.file_uploader("Choose a fruit image...", type=["jpg", "png", "jpeg"])

# 3. The Magic Logic 🪄
if file is not None:
    # Display the user's image
    image = Image.open(file)
    image = image.convert('RGB')  # Ensure 3 color channels
    st.image(image, caption='Your Image', use_column_width=True)
    
    st.write("Thinking... 🧠")

    # PRE-PROCESS (Make it look like the training data)
    # Resize to 160x160 (The size MobileNet expects)
    img = image.resize((160, 160))
    
    # Convert to array
    img_array = np.array(img)
    
    # Add batch dimension: (160, 160, 3) -> (1, 160, 160, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # PREDICT
    predictions = model.predict(img_array)
    
    # Get the best guess
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    # SHOW RESULTS
    st.write("---")

    # TIER 1: Low Confidence (The "Rejection" Zone) 🔴
    if confidence < 50:
        st.error("🚫 Unknown Object Detected")
        st.write(f"My highest guess was **{predicted_class}** ({confidence:.2f}%), but that is way too low.")
        st.write("⚠️ This image is **neither an Apple, an Orange, nor a Banana**.")
        st.info("Please upload a clear photo of one of the 3 supported fruits.")

    # TIER 2: Medium Confidence (The "Unsure" Zone) 🟡
    elif confidence < 70:
        st.warning("🤔 I am confused...")
        st.write(f"It looks a bit like a **{predicted_class}**, but I'm only **{confidence:.2f}%** sure.")
        st.write("The lighting might be bad, or the angle is tricky. Try again?")

    # TIER 3: High Confidence (The "Success" Zone) 🟢
    else:
        st.header(f"It's a... **{predicted_class}**!")
        st.write(f"Confidence: **{confidence:.2f}%**")
        
        st.progress(int(confidence))

    
        
        # Custom Messages
        if predicted_class == 'Apple':
            st.balloons()
            st.success("🍎 Did you know? Apples float in water because they are 25% air!")
        elif predicted_class == 'Banana':
            st.balloons()
            st.success("🍌 Did you know? Bananas are technically berries!")
        elif predicted_class == 'Orange':
            st.balloons()
            st.success("🍊 Did you know? There are over 600 varieties of oranges!")
        elif predicted_class == 'Unrelated':
            st.warning("🚫 This doesn't look like an apple, banana, or orange.")
            st.info("Please upload a clear photo of one of the 3 supported fruits.")    
