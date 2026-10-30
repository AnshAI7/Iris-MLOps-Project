# 🌸 Iris MLOps Project – AI Flower Classification System

<p align="center">
  <b>A Machine Learning Web Application for Iris Flower Species Prediction</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Library-Scikit--learn-orange?logo=scikitlearn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Status-Development-success" alt="Project Status">
</p>

---

## 📌 Project Overview

The **Iris MLOps Project** is a Machine Learning-based web application that predicts the species of an Iris flower using four physical measurements.

The application uses a trained **Random Forest Classifier** to classify flowers into three different Iris species:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

Users can enter flower measurements manually or upload a CSV file to generate predictions for multiple flower records.

The application also provides data visualization, model performance analysis, prediction history, and downloadable prediction results through an interactive Streamlit dashboard.

---

## 🎯 Project Objectives

The main objectives of this project are:

* To understand the complete Machine Learning workflow.
* To explore and analyze the Iris dataset.
* To train a classification model using Random Forest.
* To evaluate the trained model using different performance metrics.
* To develop an interactive web application using Streamlit.
* To predict Iris flower species using user-provided measurements.
* To support multiple predictions through CSV file upload.

---

## 📊 Dataset Information

This project uses the **Iris Dataset**, a popular dataset for learning Machine Learning classification.

The dataset contains:

| Dataset Information | Details                   |
| ------------------- | ------------------------- |
| Total Records       | 150                       |
| Total Features      | 4                         |
| Target Classes      | 3                         |
| Problem Type        | Classification            |
| Dataset Source      | Scikit-learn Iris Dataset |

### Input Features

The model uses the following four measurements:

| Feature           | Description         |
| ----------------- | ------------------- |
| Sepal Length (cm) | Length of the sepal |
| Sepal Width (cm)  | Width of the sepal  |
| Petal Length (cm) | Length of the petal |
| Petal Width (cm)  | Width of the petal  |

### Target Classes

1. Setosa
2. Versicolor
3. Virginica

---

## 🤖 Machine Learning Model

The project uses the **Random Forest Classifier**, a supervised Machine Learning algorithm used for classification tasks.

Random Forest combines multiple Decision Trees and uses their predictions to produce the final classification.

### Model Configuration

| Parameter            | Value                              |
| -------------------- | ---------------------------------- |
| Algorithm            | Random Forest Classifier           |
| Number of Estimators | 100                                |
| Test Size            | 20%                                |
| Random State         | 42                                 |
| Cross-Validation     | 5-Fold Stratified Cross-Validation |

### Model Training Workflow

1. Load the Iris dataset.
2. Separate input features and target labels.
3. Split the dataset into training and testing sets.
4. Train the Random Forest Classifier.
5. Generate predictions using test data.
6. Evaluate the model performance.
7. Save the trained model using Joblib.

The trained model is saved as:

`iris_rf_model.joblib`

---

## 📈 Model Evaluation

The model is evaluated using different classification metrics:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix
* 5-Fold Cross-Validation

These evaluation methods help understand how well the model classifies the three Iris flower species.

The application also provides visual representations of model performance.

---

## ✨ Application Features

The project includes an interactive dashboard with the following sections:

### 1. 🏠 Home

* Project introduction.
* Overview of the Iris classification system.
* Information about the three Iris species.
* Quick navigation to application features.

### 2. 🔍 Flower Prediction

* Enter four flower measurements.
* Predict the Iris flower species.
* Display prediction confidence.
* View prediction probabilities.
* Display the predicted flower image.

### 3. 📊 Data Visualization

* Dataset preview.
* Species distribution.
* Feature histograms.
* Box plots.
* Correlation heatmap.
* PCA visualization.

### 4. 📈 Model Performance

* Model accuracy.
* Cross-validation results.
* Classification report.
* Confusion matrix.
* Feature importance visualization.

### 5. 📜 Prediction History

* View previous predictions during the current session.
* Check flower measurements.
* View predicted species and confidence.
* Download prediction history as a CSV file.
* Clear prediction history.

### 6. 📁 Bulk Prediction

* Upload a CSV file.
* Validate uploaded data.
* Preview input records.
* Predict multiple flower species.
* Display prediction results and confidence scores.
* Download results as a CSV file.

### 7. 📚 About Project

* Project introduction.
* Technologies used.
* Input feature information.
* Explanation of the Random Forest algorithm.

---

## 🛠️ Technologies Used

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| Python           | Main Programming Language             |
| Pandas           | Data Manipulation                     |
| NumPy            | Numerical Operations                  |
| Matplotlib       | Data Visualization                    |
| Seaborn          | Statistical Visualization             |
| Scikit-learn     | Machine Learning                      |
| Random Forest    | Classification Algorithm              |
| Streamlit        | Web Application                       |
| Joblib           | Model Saving and Loading              |
| Jupyter Notebook | Model Development and Experimentation |

---

## 📂 Project Structure

```text
Iris-MLOps-Project/
│
├── Iris-MLOps-Project.ipynb
│
├── app.py
│
├── iris_rf_model.joblib
│
├── requirements.txt
│
└── README.md
```

**File Description:**

* `Iris-MLOps-Project.ipynb` – Notebook containing dataset analysis, model training, and evaluation.
* `app.py` – Streamlit application source code.
* `iris_rf_model.joblib` – Saved trained Random Forest model.
* `requirements.txt` – Required Python libraries.
* `README.md` – Project documentation.

---

## ⚙️ Installation and Setup

Follow these steps to run the project on your local system.

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### Step 2: Open the Project Folder

```bash
cd Iris-MLOps-Project
```

### Step 3: Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### Step 4: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 📁 Bulk Prediction – CSV Format

To use the Bulk Prediction feature, upload a CSV file containing the following exact column names:

```csv
sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2
6.3,3.3,6.0,2.5
5.8,2.7,5.1,1.9
7.1,3.0,5.9,2.1
```

### How to Use

1. Open the Streamlit application.
2. Navigate to **Bulk Prediction**.
3. Upload your CSV file.
4. Check the uploaded dataset preview.
5. Click on **Predict All Flowers**.
6. View the prediction results.
7. Download the results as a CSV file.

**Note:** All four required columns must be present and contain valid numeric values.

---

## 🚀 Future Improvements

The following improvements can be considered in future versions:

* Deploy the application on a cloud platform.
* Add automated model retraining.
* Implement model monitoring.
* Improve prediction reports.
* Add more Machine Learning algorithms.
* Improve application performance and user experience.

---

## 🎓 Academic Project Information

| Details    | Information                   |
| ---------- | ----------------------------- |
| University | Invertis University, Bareilly |
| Department | Computer Applications         |
| Course     | BCA (Artificial Intelligence) |
| Semester   | 3rd Semester                  |
| Section    | A                             |
| Project    | Iris MLOps Project            |
| Domain     | Machine Learning              |

---

## 👥 Team Members – GitHub Profiles

The following students are associated with this project:

| S. No. | Team Member     | GitHub Profile                                                        |
| ------ | --------------- | --------------------------------------------------------------------- |
| 1      | Muskan Gupta    | [muskan-gupta01](https://github.com/muskan-gupta01)                   |
| 2      | Ansh Mishra     | [AnshAI7](https://github.com/AnshAI7)                                 |
| 3      | Ritu Kushwaha   | [mizuritu](https://github.com/mizuritu)                               |
| 4      | Ishika Mathur   | [ishikamathur-coder](https://github.com/ishikamathur-coder)           |
| 5      | Pari Pal        | [paripal1812](https://github.com/paripal1812)                         |
| 6      | Vibhor Saini    | [sainivibhor235-dotcom](https://github.com/sainivibhor235-dotcom)     |
| 7      | Unnati Verma    | [vunnati163-dotcom](https://github.com/vunnati163-dotcom)             |
| 8      | Hansika Arora   | [hansikaarora726-ship-it](https://github.com/hansikaarora726-ship-it) |
| 9      | Kashifa Malik   | [Kashifamalik](https://github.com/Kashifamalik)                       |
| 10     | Palak Saxena    | [palak7002](https://github.com/palak7002)                             |
| 11     | Abhishek Tiwari | [abhitiwari639273-cell](https://github.com/abhitiwari639273-cell)     |
| 12     | Somendra        | [Somendra47](https://github.com/Somendra47)                           |
| 13     | Kashish         | [Kashish320](https://github.com/Kashish320)                           |
| 14     | Sonali Mishra   | [sonalimishra](https://github.com/sonalimishra)                       |

---

## 📝 Conclusion

The Iris MLOps Project demonstrates how Machine Learning can be used to solve a classification problem through a practical web application.

By combining dataset analysis, Random Forest classification, model evaluation, and Streamlit development, this project provides an interactive platform for predicting Iris flower species.

It also helps students understand important concepts of Machine Learning, data preprocessing, model training, evaluation, and application development.

---

<p align="center">
  <b>🌸 Iris MLOps Project</b><br>
  Machine Learning | Data Analysis | Streamlit
</p>

<p align="center">
  <i>Developed as an academic project at Invertis University, Bareilly.</i>
</p>
