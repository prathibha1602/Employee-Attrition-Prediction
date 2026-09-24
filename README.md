# Employee Attrition Prediction using Data Analytics & Machine Learning

**AICTE | IBM SkillsBuild Data Analytics with AI — Internship Program 2026 (BharatCares)**
**Internship Duration:** 17th August 2026 – 30th September 2026

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-Educational--Use-lightgrey)

---

## 📌 Project Description

This project analyzes the **IBM HR Analytics Employee Attrition dataset** to understand why employees leave an organization and to build a machine learning model that predicts the likelihood of an employee leaving. It follows a complete data analytics lifecycle — data cleaning, exploratory data analysis (EDA), visualization, machine learning modeling, and business insight generation.

## 🎯 Problem Statement

Employee attrition is costly for organizations due to recruitment, training, and productivity losses. HR teams often cannot easily identify which employees are at risk of leaving or why. This project uses data analytics to predict attrition and uncover its key drivers, enabling proactive, targeted retention strategies.

## 📊 Dataset Information

| Detail | Description |
|---|---|
| **Source** | IBM HR Analytics Employee Attrition & Performance dataset (Kaggle / IBM Watson Analytics sample data) |
| **Records** | 1,470 employees |
| **Features** | 35 columns (demographic, job-related, compensation, satisfaction, performance) |
| **Target Variable** | `Attrition` (Yes / No) |
| **File** | `data/HR-Employee-Attrition.csv` |

## 🛠️ Technologies Used

- **Python 3.10+**
- **Pandas, NumPy** — data manipulation
- **Matplotlib, Seaborn** — data visualization
- **Scikit-learn** — machine learning (Logistic Regression, Random Forest)
- **Jupyter Notebook** — interactive development and reporting

## ⚙️ Installation Steps

1. Clone or download this project folder.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ How to Run

1. Ensure the dataset file `HR-Employee-Attrition.csv` is inside the `data/` folder.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `notebook/Employee_Attrition_Prediction.ipynb` and run all cells sequentially (**Cell → Run All**).
4. Alternatively, run the plain Python script version:
   ```bash
   python notebook/Employee_Attrition_Prediction.py
   ```

## 📁 Project Structure

```
Employee_Attrition_Project/
│
├── data/
│   └── HR-Employee-Attrition.csv                 # Raw dataset
│
├── notebook/
│   ├── Employee_Attrition_Prediction.ipynb        # Main analysis notebook
│   └── Employee_Attrition_Prediction.py           # Script version of the notebook
│
├── outputs/
│   └── charts/                                    # Saved EDA & model visualizations (PNG)
│
├── report/
│   └── Employee_Attrition_Project_Report.docx     # Full project report
│
├── requirements.txt                               # Python dependencies
└── README.md                                      # Project documentation (this file)
```

## 📈 Results

- **Overall attrition rate:** ~16.1%
- **Top attrition drivers identified:** Monthly Income, Age, Total Working Years, OverTime, Years at Company
- **Best model (by recall):** Logistic Regression — Accuracy ~75%, Recall ~74%, ROC-AUC ~0.80
- **Random Forest:** Accuracy ~82%, ROC-AUC ~0.78, with feature importance ranking for interpretability

Full visualizations (distribution plots, box plots, correlation heatmap, confusion matrix, ROC curve, feature importance) are available in the notebook and the project report.

## 🚀 Future Improvements

- Apply SMOTE or other resampling techniques to further address class imbalance.
- Experiment with advanced models (XGBoost, LightGBM, Gradient Boosting).
- Build an interactive HR dashboard (Streamlit / Power BI) for real-time attrition monitoring.
- Incorporate longitudinal/time-series data to predict *when* attrition is likely to occur.

## 👤 Author Information

- **Name:** Prathibha M
- **College:** Panimalar Engineering College
- **Roll No:** 2023PECIT207
- **Year:** Final Year
- **Internship Program:** AICTE | IBM SkillsBuild Data Analytics with AI — Internship 2026 (BharatCares)
- **Internship Duration:** 17th August 2026 – 30th September 2026

## 📄 License

This project is created for educational purposes as part of an AICTE | IBM SkillsBuild (BharatCares) internship submission. The dataset is a publicly available sample dataset provided by IBM for analytics practice.
