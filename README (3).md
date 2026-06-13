# 🍷 Wine Quality Machine Learning Project

**Data Science Project | SEP-2025**  
**Author:** Arif Bin Mushtaq  
**Program:** MSc Statistics, University of Delhi

---

## 📌 Overview

This project develops and evaluates a **Random Forest-based classification model** to predict wine quality from physicochemical data. By analyzing chemical properties such as alcohol content, acidity, sulphates, and pH, the model classifies wines as **Good** or **Bad** quality with approximately **97% accuracy**.

---

## 🎯 Objectives

- Understand key factors affecting wine quality through Exploratory Data Analysis (EDA)
- Build a binary classification model (Good vs. Bad wine) based on chemical properties
- Identify the most influential physicochemical features using feature importance analysis
- Evaluate model performance using accuracy, precision, recall, and F1-score

---

## 📂 Project Structure

```
wine-quality-ml/
│
├── WineQT.csv                  # Dataset
├── wine_quality_analysis.ipynb # Main notebook
└── README.md                   # Project documentation
```

---

## 📊 Dataset

- **Source:** Kaggle — Wine Quality Dataset
- **Total Records:** 1,144 wine samples
- **Features:** 12 physicochemical properties + quality score + ID

| Feature | Description |
|---|---|
| fixed acidity | Concentration of non-volatile acids (e.g., tartaric acid) |
| volatile acidity | Acetic acid content; high levels produce vinegar flavor |
| citric acid | Contributes to freshness and flavor |
| residual sugar | Sugar remaining after fermentation |
| chlorides | Salt content |
| free sulfur dioxide | Free SO₂ acting as antimicrobial/antioxidant |
| total sulfur dioxide | Combined free and bound SO₂ |
| density | Correlated with alcohol and sugar content |
| pH | Acidity/alkalinity of the wine |
| sulphates | Potassium sulfate; contributes to flavor and stability |
| alcohol | Alcohol percentage by volume |
| quality | Expert sensory score (0–10 scale) — **target variable** |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, preprocessing |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots (heatmaps, boxplots, pairplots) |
| `scikit-learn` | Model building and evaluation |

---

## 🔍 Methodology

### 1. Data Preprocessing
- Checked for and handled missing values
- Explored distributions and detected outliers
- Converted quality scores into binary labels: **Good (1) / Bad (0)**

### 2. Exploratory Data Analysis (EDA)
- **Correlation Heatmap** — identified key predictors of quality
- **Bar Plots** — compared feature means across quality categories
- **Box Plots** — visualized distributions and outliers for all features
- **Pair Plot** — examined pairwise relationships colored by quality
- **Distribution Plots** — analyzed the shape of each feature's distribution

### 3. Key Correlations Found

| Feature | Correlation with Quality | Direction |
|---|---|---|
| Alcohol | 0.5 | ✅ Positive (strongest) |
| Volatile Acidity | -0.4 | ❌ Negative |
| Sulphates | 0.3 | ✅ Positive |
| Citric Acid | 0.2 | ✅ Slightly positive |
| Residual Sugar / Chlorides | ~0.0 | ➖ Negligible |

### 4. Model Development

- **Algorithm:** Random Forest Classifier (`RandomForestClassifier`)
- **Trees:** 200 decision trees (`n_estimators=200`)
- **Train/Test Split:** 80% training / 20% testing (`random_state=2`)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(x_train, y_train)
```

### 5. Model Evaluation

```python
from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_test, rfc.predict(x_test))
```

**Model Accuracy: ~97%**

---

## 📈 Feature Importance (Random Forest)

Ranked by importance score:

1. 🥇 **Alcohol** — Most influential predictor
2. 🥈 **Volatile Acidity** — Strong negative impact
3. 🥉 **Sulphates** — Positive contribution
4. Citric Acid
5. Density
6. Total Sulfur Dioxide
7. Chlorides, Fixed Acidity, pH, Residual Sugar, Free Sulfur Dioxide

---

## 🔮 Making a Prediction

```python
import numpy as np

# Example wine sample: (fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
#                        chlorides, free_SO2, total_SO2, density, pH, sulphates, alcohol, Id)
input_data = (7.3, 0.65, 0.0, 1.2, 0.065, 15.0, 21.0, 0.9946, 3.39, 0.47, 10.0, 11)

input_array = np.asarray(input_data).reshape(1, -1)
prediction = rfc.predict(input_array)

if prediction[0] == 1:
    print("Good Quality Wine ✅")
else:
    print("Bad Quality Wine ❌")
```

---

## 📝 Key Findings

1. **Alcohol content** is the single most important driver of wine quality — higher alcohol consistently correlates with better ratings.
2. **Volatile acidity** has a strong negative impact; elevated levels produce vinegar-like aromas and reduce acceptability.
3. **Sulphates and citric acid** are positively associated with quality, contributing to freshness and microbial stability.
4. **Residual sugar, chlorides, and pH** have minimal predictive power and can be deprioritized in future models.
5. Most wines in the dataset are of **average quality** (score 5 or 6), indicating a class imbalance worth addressing in future work.

---

## ✅ Actionable Recommendations

- **Reduce volatile acidity** during fermentation to prevent off-flavors
- **Optimize sulphate and citric acid levels** to enhance freshness within regulatory limits
- **Monitor alcohol content** closely — it is the most impactful quality lever
- **Deprioritize** residual sugar and chloride monitoring for quality-control efforts

---

## 🔭 Future Work

- Build a **regression model** to predict continuous quality scores (0–10) for finer granularity
- Deploy the trained model into a **real-time production monitoring** system
- Perform **periodic feature importance analysis** to track shifts in quality drivers over time
- Apply **oversampling techniques** (e.g., SMOTE) to address class imbalance

---

## 📚 References

1. [Intro to Machine Learning — Kaggle](https://www.kaggle.com/learn/intro-to-machine-learning)
2. Wine Quality Dataset — UCI Machine Learning Repository / Kaggle
