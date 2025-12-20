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

# --- PART 4: TRAIN 🏋️ ---
print("Training the new layers...")

# Since the base is already smart, we don't need many epochs!
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save as the "Pro" version
model.save('my_fruit_model_pro.keras')
print("✅ Model saved as 'my_fruit_model_pro.keras'")