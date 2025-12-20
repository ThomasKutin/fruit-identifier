import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers

print("--- 🚀 Starting Fruit-Identifier PRO (Transfer Learning) ---")

# --- PART 1: LOAD DATA (Bigger Size!) ---
# MobileNet likes images around 160x160 pixels or larger.
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

print("Loading images...")

train_ds = image_dataset_from_directory(
    'dataset/',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    'dataset/',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# --- PART 2: DOWNLOAD THE GENIUS BRAIN 🧠 ---
# We use MobileNetV2. 
# include_top=False means "Cut off the last layer". 
# (Because the original model was trained for 1000 things, we only want 3 fruits).
print("Downloading the Genius Brain (MobileNetV2)...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)

# FREEZE THE BASE! 🧊
# This is crucial. We don't want to retrain the millions of parameters inside MobileNet.
# We just want to use them.
base_model.trainable = False

# --- PART 3: ADD OUR CUSTOM LAYERS ---
# This is the "Tape on the forehead" part.

model = tf.keras.Sequential([
    # Input Layer
    tf.keras.Input(shape=(160, 160, 3)),
    
    # Pre-processing: MobileNet expects pixels between -1 and 1
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    
    # Data Augmentation (The Hall of Mirrors is still good!)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    
    # The Genius Brain
    base_model,
    
    # The Summarizer (GlobalAveragePooling2D is like Flatten but smarter)
    layers.GlobalAveragePooling2D(),
    
    # A Dropout layer to prevent overconfidence
    layers.Dropout(0.2),
    
    # THE FINAL DECISION: 3 Fruits
    layers.Dense(3, activation='softmax')
])

# Compile
# We use a slightly lower learning rate (0.0001) to be gentle with the new layers.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ... (Keep Parts 1, 2, and 3 exactly the same) ...

# --- PART 4: ROUND 1 - TRAIN THE HEAD (The Warm-Up) 🏃 ---
print("--- Round 1: Training the new head... ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5  # Short warm-up
)

# --- PART 5: ROUND 2 - FINE TUNING (The Deep Study) 🧘 ---
print("--- Round 2: Unfreezing the Professor (Fine-Tuning)... ---")

# 1. Unfreeze the Genius Brain
base_model.trainable = True

# 2. But wait! We don't want to unfreeze ALL of it (too risky).
# Let's verify how many layers are in the base model.
print(f"Number of layers in the base model: {len(base_model.layers)}")

# Freeze all the bottom layers (keep the basic shapes/lines knowledge locked)
# We only unfreeze the top 50 layers to learn "Orange Texture".
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 3. RE-COMPILE (Crucial Step!)
# We use a TINY learning rate (1e-5). 
# If we learn too fast, we might break the pre-trained knowledge.
model.compile(loss='sparse_categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
              metrics=['accuracy'])

# 4. Train again
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # Study specifically for the difficult fruits
)

# Save as the "Fine Tuned" version
model.save('my_fruit_model_finetuned.keras')
print("✅ Model saved as 'my_fruit_model_finetuned.keras'")