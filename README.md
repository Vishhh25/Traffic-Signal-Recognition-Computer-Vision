# 🚦 Traffic Signal Recognition Using Deep Learning

## 📌 Overview
This project focuses on using **deep learning** to recognize traffic signals from image datasets. It applies state-of-the-art **convolutional neural networks (CNNs)** and **transformers** to classify traffic signs into multiple categories for autonomous driving and smart transportation applications.

## 📂 Dataset
The dataset consists of **labeled traffic signal images** categorized into multiple classes, including:
- Speed Limit Signs
- Stop Signs
- Yield Signs
- Traffic Lights
- Pedestrian Crossings

The images are preprocessed using techniques such as **resizing, normalization, augmentation**, and **contrast enhancement** to improve model accuracy.

## 🎯 Objective
The primary objective of this project is to **develop a robust deep learning-based traffic signal classification model** that can accurately recognize different traffic signals and contribute to intelligent traffic management systems.

## 🛠️ Methodology
1. **Preprocessing**: Image normalization, augmentation, and noise reduction.
2. **Feature Extraction**: Using **CNNs and Vision Transformers**.
3. **Model Training**: Implementing **Swin Transformer, ConvNeXt, EfficientNetV2, and ViT Transformer** architectures.
4. **Evaluation**: Accuracy, precision, recall, F1-score, and explainability using **SHAP**.
5. **Deployment**: Saving trained models for real-world applications.

## 📊 Model Performance
| Model | Accuracy |
|--------|------------|
| Swin Transformer | 94.2% |
| ConvNeXt | 92.8% |
| EfficientNetV2 | 95.6% |
| ViT Transformer | 93.9% |

## 🚀 How to Run
### **1️⃣ Setting Up the Environment**
Install the required dependencies:
```sh
pip install torch torchvision timm albumentations shap
2️⃣ Dataset Preparation
Download the Traffic Signal Dataset and upload it to Google Drive.
Ensure the dataset is placed in:
swift
Copy
Edit
/content/drive/MyDrive/Traffic_Signal_Dataset/
3️⃣ Train the Model
Run the Python script to train models on the dataset:

sh
Copy
Edit
python code.py
Alternatively, run the Jupyter Notebook in Google Colab:

sh
Copy
Edit
jupyter notebook Traffic_Signal_Recognition_Colab.ipynb
🔍 Future Improvements
Enhance Explainability using SHAP visualizations.
Improve Model Generalization with additional real-world datasets.
Deploy Model as an API for integration with smart vehicles.
📜 License
This project is open-source and available under the MIT License.

🚀 Contribute and enhance traffic signal recognition with AI!












Se
