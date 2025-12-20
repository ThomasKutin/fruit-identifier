import os
import shutil

# 1. Define where the messy data is (The folder we just cloned)
# The repository has a 'Training' folder inside it
source_folder = "Fruit-Images-Dataset/Training"

# 2. Define where we want the clean data to go
destination_folder = "dataset"

# 3. Define which fruits we want to keep (The "Shopping List")
# We map the "Messy Name" to the "Clean Name"
fruit_map = {
    "Apple Braeburn": "apples",       # We'll take Braeburn apples
    "Banana": "bananas",              # Standard Bananas
    "Orange": "oranges"               # Standard Oranges
}

# 4. The Sorting Machine 🏗️
print("🤖 Starting the Data Sorter...")

# Make sure our destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for messy_name, clean_name in fruit_map.items():
    # Construct the full path to the source folder
    src_path = os.path.join(source_folder, messy_name)
    
    # Construct the full path to the new home
    dest_path = os.path.join(destination_folder, clean_name)
    
    # Check if the source fruit actually exists
    if os.path.exists(src_path):
        print(f"found {messy_name}... moving copies to {clean_name} folder!")
        
        # If the destination folder already exists, delete it first (start fresh)
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        
        # Copy the entire folder over
        shutil.copytree(src_path, dest_path)
    else:
        print(f"⚠️ Warning: Could not find {messy_name} in the downloaded data.")

print("✅ Done! Your 'dataset' folder is packed and ready.")