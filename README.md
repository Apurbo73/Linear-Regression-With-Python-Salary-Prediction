### Salary Prediction



---

## 1. **What is pandas?**

* **pandas** is an open-source Python library for **data manipulation and analysis**.
* Built on top of **NumPy**.
* Core objects:

  * **`Series`** → 1D labeled array.
  * **`DataFrame`** → 2D labeled table with rows & columns.

In ML, pandas is used **before modeling** (and sometimes after) for data cleaning, exploration, and transformation.

---

## 2. **Why pandas is important in ML**

Machine learning models are only as good as the data you feed them. Pandas is like the “Swiss Army knife” for:

* **Reading datasets** from CSV, Excel, SQL, JSON, Parquet, etc.
* **Cleaning** messy data.
* **Exploratory Data Analysis (EDA)**.
* **Feature engineering** (creating new variables).
* **Merging/joining** multiple datasets.
* **Handling missing values**.
* **Converting data** into formats suitable for scikit-learn, TensorFlow, etc.

---

## 3. **Common pandas workflow in ML**

### **Step 1 — Load the data**

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

Formats: `.read_csv()`, `.read_excel()`, `.read_sql()`, `.read_parquet()`.

---

### **Step 2 — Explore the data**

```python
df.head()          # first 5 rows
df.info()          # column types & non-null counts
df.describe()      # summary stats for numerical data
df['column'].value_counts()  # frequency counts
```

---

### **Step 3 — Clean the data**

* Missing values:

```python
df.isnull().sum()
df['age'].fillna(df['age'].median(), inplace=True)
df.dropna(subset=['salary'], inplace=True)
```

* Duplicates:

```python
df.drop_duplicates(inplace=True)
```

* Type conversion:

```python
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
```

---

### **Step 4 — Feature engineering**

* Create new features:

```python
df['BMI'] = df['weight_kg'] / (df['height_m'] ** 2)
```

* Encode categorical variables:

```python
pd.get_dummies(df, columns=['gender'], drop_first=True)
```

* Binning:

```python
df['age_group'] = pd.cut(df['age'], bins=[0,18,35,60,100],
                         labels=['child','young_adult','adult','senior'])
```

---

### **Step 5 — Combine datasets**

```python
df_merged = pd.merge(df1, df2, on='id', how='left')
df_concat = pd.concat([df1, df2], axis=0)  # stack rows
```

---

### **Step 6 — Prepare for ML models**

* Selecting features & target:

```python
X = df.drop('target', axis=1)
y = df['target']
```

* Convert to NumPy (scikit-learn friendly):

```python
X.values
```

---

## 4. **Key pandas functions for ML work**

| Task           | Function                                                |
| -------------- | ------------------------------------------------------- |
| Read data      | `pd.read_csv()`, `pd.read_excel()`, `pd.read_parquet()` |
| Summary        | `.head()`, `.info()`, `.describe()`                     |
| Missing values | `.isnull()`, `.fillna()`, `.dropna()`                   |
| Selection      | `df[['col1','col2']]`, `.loc[]`, `.iloc[]`              |
| Grouping       | `.groupby()`                                            |
| Joining        | `.merge()`, `.concat()`, `.join()`                      |
| Encoding       | `pd.get_dummies()`                                      |
| Time series    | `.dt` accessor                                          |
| Apply function | `.apply()`, `.map()`, `.applymap()`                     |

---

## 5. **Best practices in ML with pandas**

* Always **check data types** before modeling.
* Handle **missing values** explicitly (don’t let them silently break training).
* Avoid chained indexing (e.g., `df[df['x']>0]['y'] = 1`) — it can cause bugs.
* Use **vectorized operations** instead of loops (faster).
* Keep track of transformations for **test data consistency**.
* Use `.copy()` when you don’t want changes to affect the original DataFrame.

---

## 6. **Common pitfalls**

* Forgetting to reset index after dropping rows → misalignment.
* Using `.apply()` for something that can be done with direct arithmetic (slower).
* Not matching preprocessing between train/test datasets → data leakage.
* Memory issues with very large datasets (consider **Dask** or **PySpark**).

---

## 7. **pandas in the ML pipeline**

In practice, pandas usually appears **before** scikit-learn:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")
X = pd.get_dummies(df.drop('target', axis=1))
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
```

---




---

# **1. NumPy in Machine Learning**

NumPy (**Numerical Python**) is a core Python library for **fast numerical computation**.
It’s the *foundation* of pandas, scikit-learn, and many ML frameworks.

---

## **Why NumPy is important for ML**

* **Speed** → Uses C under the hood for fast array operations.
* **Multi-dimensional arrays (ndarrays)** for representing vectors, matrices, and tensors.
* **Mathematical operations**: linear algebra, statistics, Fourier transforms, random sampling.
* **Memory efficiency** → Stores data in contiguous memory blocks.
* **Interoperability** → Works seamlessly with pandas, scikit-learn, PyTorch, TensorFlow.

---

## **Core NumPy Features**

```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

# Element-wise operations
c = a * 2        # [2, 4, 6]
d = b + 10       # [[11, 12], [13, 14]]

# Mathematical functions
np.mean(a)       # average
np.std(a)        # standard deviation

# Linear algebra
np.dot(a, a)     # dot product
np.linalg.inv([[1, 2], [3, 4]])  # matrix inverse
```

---

## **In ML, NumPy is used for:**

1. **Raw data representation** → before converting to DataFrames.
2. **Matrix math** → for gradient calculations, dot products.
3. **Weight initialization** in neural networks.
4. **Performance optimization** for batch operations.
5. **Random sampling** for train/test splits or synthetic datasets.

---

# **2. Matplotlib in Machine Learning**

Matplotlib is Python’s most widely used **data visualization** library.
It’s the “plotting engine” behind pandas `.plot()` and is often paired with **Seaborn** for prettier charts.

---

## **Why Matplotlib is important for ML**

* Visualizing **data distributions** before training.
* Tracking **model performance** over epochs (loss curves, accuracy).
* Creating **confusion matrices** and **ROC curves**.
* Debugging → seeing where predictions go wrong.

---

## **Core Matplotlib Workflow**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label="sin(x)")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Sine Wave")
plt.legend()
plt.show()
```

---

## **In ML, Matplotlib is used for:**

1. **Exploratory Data Analysis (EDA)**

   ```python
   plt.hist(df['age'], bins=20)
   plt.show()
   ```

2. **Model training curves**

   ```python
   plt.plot(history.history['loss'], label='Train Loss')
   plt.plot(history.history['val_loss'], label='Val Loss')
   plt.legend()
   ```

3. **Evaluation metrics**

   ```python
   from sklearn.metrics import confusion_matrix
   import seaborn as sns

   cm = confusion_matrix(y_true, y_pred)
   sns.heatmap(cm, annot=True, fmt="d")
   ```

---

# **3. How pandas, NumPy, and Matplotlib fit together in ML**

Think of them as a **data science toolkit**:

* **NumPy** → The math engine (fast arrays, matrix ops).
* **pandas** → The spreadsheet brain (structured, labeled data).
* **Matplotlib** → The artist (plots & visualizations).

**Example ML mini-pipeline:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data.csv")

# Clean data
df.dropna(inplace=True)

# Convert to NumPy for modeling
X = df.drop('target', axis=1).values
y = df['target'].values

# Simple visualization
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```

---


