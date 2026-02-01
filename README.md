# 😊 Emotion Detection System Using CNN

## 📌 Internship Task – 2 (CODTECH)

- COMPANY- CODTECH IT SOLUTIONS
- NAME- MAYANK SRIVASTAVA
- INTERN ID- CTIS4257
- DOMAIN- DATA SCIENCE
- DURATION - 12 WEEKS
- MENTOR - NEELA SANTHOSH KUMAR

A **Deep Learning based Emotion Detection project** that identifies human emotions from facial images using a **Convolutional Neural Network (CNN)**.

📂 **GitHub Notebook:** https://github.com/ms00000ms0000/DL_Project_Emotion_Detection_System_Using_CNN/blob/main/Emotion_Detection_CNN.ipynb  
🔍 **Emotion classes include:** Angry, Disgust, Fear, Happy, Pain, Sad.  

---

## 📌 Project Overview

Emotion detection from facial expressions is an important task in **computer vision and human-computer interaction**.  
This project builds a **CNN model** trained on facial expression data and performs classification of emotions from images. The model learns spatial features from grayscale face images to classify emotions accurately.

CNNs are widely used for image-based recognition tasks due to their ability to detect local and hierarchical features in visual data. :contentReference[oaicite:1]{index=1}

---

## 📂 Repository Contents

```
DL_Project_Emotion_Detection_System_Using_CNN
│
├── Train_Data_and_Test_Data
├── Emotion_Detection_CNN.ipynb                                                # Jupyter notebook with data preprocessing, model training & evaluation
└── README.md                                                                  # Project documentation

```
---
## 🧠 Key Features

- 🤖 **Deep Learning Model:** A Convolutional Neural Network trained to classify facial emotions  
- 📊 **Exploratory Data Analysis (EDA):** Visualization and understanding of the dataset  
- 🧪 **Training & Evaluation:** Model built, trained, and validated in the notebook  
- 💡 **Classification of 6 Emotions:** Angry, Disgust, Fear, Happy, Pain, Sad 

---

## 🛠️ Tech Stack

- **Frameworks:** Python, TensorFlow/Keras  
- **Libraries:** NumPy, Pandas, Matplotlib, MobileNetV2   
- **Development:** Jupyter Notebook  
- **Deep Learning Model:** CNN (Convolutional Neural Network)

---

## 🧪 Dataset Description

The project typically uses a facial expression dataset consisting of **grayscale face images (48x48 pixels)** labeled with one of seven emotions.  
Commonly used datasets for this problem have thousands of images with balanced emotion classes.  

Each input image conveys one emotion and the CNN learns to classify these during training.

---

## 🧠 CNN Model Workflow

1. **Data Preprocessing**
   - Load image data and labels  
   - Normalize pixel values  
   - Split into training and validation sets

2. **Model Architecture**
   - Sequence of convolutional and pooling layers  
   - Batch normalization and dropout for regularization  
   - Dense layers with softmax output for classification

3. **Training**
   - Multiple epochs with loss and accuracy tracking  
   - Optimization using suitable optimizer (e.g., Adam)

4. **Evaluation**
   - Model performance evaluated on validation/test data  
   - Visualizations of loss/accuracy trends

---

## 📈 Results & Performance

* The model is trained to differentiate between multiple facial emotion classes and typically shows promising performance during validation.  
* Feel free to inspect the training curves, classification accuracy, and confusion matrices included inside the notebook.

---

## 🧩 Usage

To **run this project locally**:

1. **Clone the repository**
   
```bash

git clone https://github.com/ms00000ms0000/DL_Project_Emotion_Detection_System_Using_CNN.git
```
2. **Install dependencies**

```bash
pip install numpy pandas tensorflow opencv-python matplotlib seaborn

```

3. **Open the Jupyter Notebook**

```bash
jupyter notebook Emotion_Detection_CNN.ipynb

```

4. **Execute cells from top to bottom to train and evaluate the model.**

---

## 💡 Future Improvements

*Add real-time webcam emotion detection using OpenCV

*Improve accuracy via data augmentation and hyperparameter tuning

*Save the trained model for deployment in web or mobile apps

*Integrate the model with a UI (e.g., Streamlit or Flask)

---
