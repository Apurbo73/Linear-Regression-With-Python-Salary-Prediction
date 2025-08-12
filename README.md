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

