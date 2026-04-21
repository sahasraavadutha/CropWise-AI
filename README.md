# CropWise-AI


🌾 CropWise AI – Crop Disease Detection System

An AI-powered web application that detects crop diseases from leaf images using Deep Learning and provides actionable insights like treatment, prevention, and risk analysis.


 🚀 Features

* 🔍 Disease Detection using AI (CNN)
* 🏆 Top-3 Predictions with Confidence Scores
* 📅 Seasonal Risk Analysis
* 🔥 Grad-CAM Heatmap (Explainable AI)
* 📄 Downloadable PDF Report
* 📊 Prediction History Dashboard
* 🧾 CSV Export of Predictions
* 📖 Disease Encyclopedia
* 📝 Farmer Notes Support


 🧠 Tech Stack

 🔹 Backend

* Python
* Flask

 🔹 Machine Learning

* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)

 🔹 Image Processing

* OpenCV
* Pillow

 🔹 Data Handling

* NumPy

🔹 Database

* SQLite

- Report Generation

* FPDF



 📊 Dataset

This project uses the PlantVillage Dataset:

🔗 https://www.kaggle.com/datasets/emmarex/plantdisease

* Contains labeled images of plant leaves
* Includes healthy and diseased samples
* Used for training the model



 🗂️ Project Structure

SmartCrop_Project/
│
├── app.py                 # Main Flask app
├── train.py               # Model training script
├── dataset_reduced.py     # Dataset preparation script
├── requirements.txt       # Dependencies
├── history.db             # SQLite database
│
├── dataset/               # Original dataset (PlantVillage)
├── dataset_reduced/       # Reduced dataset for training
├── model/
│   ├── crop_disease_model.h5
│   ├── classes.txt
│
├── static/
│   ├── uploads/
│   ├── heatmaps/
│
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   ├── encyclopedia.html
│
└── venv/




 ⚙️ Installation & Setup

 1️⃣ Clone the Repository

git clone https://github.com/your-username/cropwise-ai.git
cd cropwise-ai




2️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate   # Windows


3️⃣ Install Dependencies


pip install -r requirements.txt


🧪 Model Training

 Prepare dataset


python dataset_reduced.py


 Train model


python train.py

Model will be saved in:


model/crop_disease_model.h5
 ▶️ Run the Application


python app.py


Open browser:

http://127.0.0.1:5000
📸 How to Use

1. Upload a leaf image
2. Click Predict
3. View:

   * Disease name
   * Confidence score
   * Heatmap visualization
   * Treatment suggestions
4. Download report (PDF)

📈 Algorithms Used

* Convolutional Neural Networks (CNN)
* Transfer Learning (MobileNetV2)
* Softmax Classification
* Grad-CAM (Explainable AI)

 ⚠️ Limitations

* Works only on trained classes (limited crops)
* Accuracy depends on image quality
* May misclassify unseen diseases

🚀 Future Improvements

* Add more crops and diseases
* Improve accuracy with larger dataset
* Deploy on cloud (AWS / Render)
* Mobile app integration
* Real-time camera detection

 👨‍💻 Author

* Name: A.Sahasra
* Project: CropWise AI
* Year: 2026

 📜 License

This project is for educational purposes only.


⭐ Acknowledgements

* PlantVillage Dataset (Kaggle)
* TensorFlow & Keras
* Open-source community


