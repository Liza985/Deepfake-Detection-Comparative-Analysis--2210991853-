# Deepfake Detection Using CNN and Capsule Networks: A Comparative Analysis and Performance Evaluation

## 👨‍🎓 Student Details
- Name: Liza   
- Roll No: 2210991853  

---

## 📌 Project Type
Research Project (COOP-II)

---

## 📖 Project Description
This project focuses on detecting deepfake images using deep learning. 
The system combines Convolutional Neural Networks (CNN) for hierarchical 
feature extraction with capsule-inspired dense layers that preserve spatial 
relationships between facial features.

The model is trained and evaluated on benchmark datasets FaceForensics++ 
and DFDC, performing binary classification to distinguish real images from 
manipulated (fake) ones.

---

## 🎯 Objectives
- Detect fake (manipulated) images from real images  
- Implement CNN-based hierarchical feature extraction  
- Preserve spatial relationships between facial features using 
  capsule-inspired dense layers  
- Evaluate model using accuracy, precision, recall, F1-score, 
  and confusion matrix  

---

## ⚙️ Implementation Details
- Dataset: FaceForensics++ and DFDC benchmark datasets  
- Train-Test Split: 80:20  
- Preprocessing: Image resizing (128×128) and normalization  
- Model: Hybrid CNN + Capsule-inspired architecture (~8.5M parameters)  
- Optimizer: Adam | Loss: Binary Cross-Entropy | Epochs: 25  
- Classification: Binary (Real vs Fake)  

---

## 📊 Results
- Accuracy: 92.4%  
- Precision: 91.2%  
- Recall: 90.8%  
- F1-Score: 91.0%  
- Confusion Matrix: 185/200 samples correctly classified  
- Outperforms standalone CNN (89.1%) and Capsule Network (90.7%)  

---

## 📂 Repository Structure
- **Code/** → Python implementation (deepfake_detection_complete.py)  
- **Dataset/** → Real and fake images  
- **Results/** → Output graphs and evaluation  
- **Report and PPT/** → Final report and presentation  
- **Research Paper and Submission Proof/** → Paper + submission screenshot  

---

## 📑 IPR Submission Proof
The research paper has been submitted to:

**PEC Chandigarh Conference (CHANDICON 2026)**  

✔ Screenshot of submission is included in the repository.

---

## 📌 Current Status
- Project implementation: ✅ Completed  
- Report and PPT: ✅ Completed  
- Research Paper: ✅ Submitted to conference  
- GitHub Repository: ✅ Completed  

---

## 🛠️ Technologies Used
- Python 3.x  
- NumPy  
- OpenCV (cv2)  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- TensorFlow / Keras (conceptual architecture)  

---

## 👩‍💻 Author
Liza  
Roll No: 2210991853  
Department of Computer Science and Engineering  
Chitkara University Institute of Engineering and Technology, Punjab  

---

## ⚠️ Note
The research paper associated with this project has been submitted to 
PEC Chandigarh Conference (CHANDICON 2026). Submission proof is 
included in the Research Paper and Submission Proof folder.