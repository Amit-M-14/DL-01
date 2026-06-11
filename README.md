# 🩺 Deep Learning for Automated X-Ray Diagnostics

## 📌 Project Overview
This project focuses on building an image classification model to identify diseases, specifically pneumonia, from chest X-ray images. By leveraging Convolutional Neural Networks (CNNs) and Transfer Learning, this model assists in automated medical diagnostics, optimizing for high recall to minimize dangerous false negatives in patient care.

## 🗄️ The Data
The model is trained on the open-source **Kaggle Chest X-Ray Images (Pneumonia)** dataset, which contains over 5,000 high-resolution pediatric chest X-rays.
* **Dataset Link:** [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## 🛠️ Tech Stack & Methodology
* **Framework:** PyTorch (Torchvision)
* **Architecture:** ResNet50 (Transfer Learning)
* **Hardware:** Google Colab (NVIDIA T4 GPU)
* **UI/UX:** Gradio (Interactive Web Interface)
* **Methodology:** The pre-trained ResNet50 model's base layers were frozen to preserve foundational visual feature extraction. The final fully connected layer was replaced and fine-tuned on the medical dataset using Cross-Entropy Loss and the Adam optimizer.

## 🚀 How to Run the App
This project is built to run entirely in the cloud via Google Colab. 
1. Open the `.ipynb` notebook in Google Colab.
2. Navigate to the **Secrets** tab (key icon) on the left sidebar.
3. Add a new secret named `KAGGLE_API_TOKEN` and paste your Kaggle API key as the value. Ensure **Notebook access** is toggled on.
4. Run all cells sequentially. 
5. The final cell will generate a public Gradio web link. Click it to launch the drag-and-drop diagnostic interface!

## 🧪 Real-World Application (Domain Shift)
During testing, the model accurately classifies images native to its training distribution (the Kaggle dataset). However, passing random adult X-rays from Google Images into the UI may result in varied confidence scores due to **Domain Shift** (differences in cropping, contrast, and hospital-specific text markers). This highlights the importance of training medical models on diverse, globally representative datasets before clinical deployment.
