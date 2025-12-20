import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from tensorflow.keras.utils import image_dataset_from_directory

print("Starting Fruit-Identifier 3000...")

# --- PART 1: LOAD THE DATA (The Lunchbox) ---
# This looks into the 'dataset' folder and automatically splits data.
# 80% for studying (Training), 20% for testing (Validation).

print("Loading images...")

# The Training Data (Studying)
train_ds = image_dataset_from_directory(
    directory='dataset/',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(64, 64),
    batch_size=32
)

# The Testing Data (The Exam)
val_ds = image_dataset_from_directory(
    directory='dataset/',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(64, 64),
    batch_size=32
)

# ... (Keep the imports and loading part the same) ...

# --- PART 2: BUILD THE BRAIN (The Architecture) ---
print("Building the CNN...")

model = Sequential()

# === 🆕 NEW: The Hall of Mirrors (Data Augmentation) ===
# These layers ONLY run during training. They flip and rotate images randomly
# so the robot learns to recognize fruits from all angles.
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

model.add(RandomFlip("horizontal", input_shape=(64, 64, 3)))
model.add(RandomRotation(0.1)) # Rotate slightly (10%)
model.add(RandomZoom(0.1))     # Zoom in/out slightly
# =======================================================

# Pre-processing (Scale the pixels)
model.add(Rescaling(1./255))

# Layer 1: Edge Detection
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2: Shape Detection
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3: Complex Feature Detection
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten & Decide
model.add(Flatten())


# Let's add 'Dropout'. It randomly turns off neurons so they don't get lazy!
from tensorflow.keras.layers import Dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) 

# Output Layer
model.add(Dense(3, activation='softmax'))

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- PART 3: TRAIN (The Gym) ---
# Since the problem is harder now (moving targets!), let's train longer.
print("Training in progress... (Training longer this time!) 🏋️")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25  # Increase from 10 to 25 loops
)

# ... (Save the model as before) ...

print("--- ✅ Training Complete! ---")

# Optional: Save the brain so we don't have to retrain it later
model.save('my_fruit_model.keras')
print("Model saved as 'my_fruit_model.keras'")