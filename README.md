# 🍎🍌🍊Kutin's Fruit Identifier 

A Deep Learning application that can classify fruits (Apples, Bananas, Oranges) from images. 
Built with Python, TensorFlow/Keras, and Streamlit.

## 🧠 How it Works
This project uses **Transfer Learning** with the **MobileNetV2** architecture. 
* **Base Model:** Pre-trained on ImageNet (millions of images).
* **Fine-Tuning:** The top layers were retrained specifically to distinguish between fruit textures (e.g., Apple skin vs. Orange peel).
* **Accuracy:** Achieves ~95% accuracy on the test set.

## 🛠️ Tech Stack
* **Language:** Python
* **AI Engine:** TensorFlow & Keras
* **Interface:** Streamlit
* **Data Source:** [Fruits-360 Dataset](https://github.com/Horea94/Fruit-Images-Dataset)

## 🚀 How to Run it locally

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/fruit-identifier-cnn.git](https://github.com/YOUR_USERNAME/fruit-identifier-cnn.git)
    cd fruit-identifier-cnn
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Get the Data:**
    Run the setup script to download and sort the Kaggle dataset automatically:
    ```bash
    python setup_data.py
    ```

4.  **Train the Brain (Optional):**
    If you want to retrain the model yourself:
    ```bash
    python fruit_brain_pro.py
    ```

5.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

