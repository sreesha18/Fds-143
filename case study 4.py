# %% [markdown]
# # Fundamentals of Data Science - Case Study 4: Student Data and Performance
# **Name:** Sreesha CH | **Roll No:** 160124733143
#
# Target: Dropout / Enrolled / Graduate. Workflow: audit -> explore -> hypothesis test -> classification -> per-class evaluation.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, ConfusionMatrixDisplay)

RANDOM_STATE = 42
DATA_PATH = "data.csv"   # change to your student dataset file; it is semicolon-separated

# %% [markdown]
# ## 1. Load data and quality audit

# %%
df = pd.read_csv(DATA_PATH, sep=";")
df.columns = df.columns.str.strip()
print("Records:", df.shape[0], "| Columns:", df.shape[1])
df.head()

# %%
# Find the target column (usually named "Target")
TARGET = "Target"
assert TARGET in df.columns, f"Rename TARGET to match your file. Columns: {list(df.columns)}"

print("Missing cells  :", int(df.isna().sum().sum()))
print("Duplicate rows :", int(df.duplicated().sum()))
print("Target classes :", df[TARGET].nunique(), list(df[TARGET].unique()))
df.dtypes.value_counts()

# %% [markdown]
# ## 2. Target distribution

# %%
dist = df[TARGET].value_counts()
dist_df = pd.DataFrame({"Count": dist, "Share_%": (dist / len(df) * 100).round(1)})
print(dist_df)

order = ["Dropout", "Enrolled", "Graduate"]
colors = ["#d9a58b", "#a9bfae", "#6b8f7a"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
dist.reindex(order).plot(kind="bar", ax=ax[0], color=colors)
ax[0].set_title("Student outcomes: counts"); ax[0].set_ylabel("Records")
ax[1].pie(dist.reindex(order), labels=order, autopct="%1.1f%%", colors=colors)
ax[1].set_title("Student outcomes: percentage")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Descriptive statistics of key variables
# Column names below follow the UCI "Predict students' dropout and academic success" dataset.
# If your file uses different names, edit them in the next cell.

# %%
AGE = "Age at enrollment"
ADM = "Admission grade"
SEM1 = "Curricular units 1st sem (grade)"
SEM2 = "Curricular units 2nd sem (grade)"

key_vars = [AGE, ADM, SEM1]
print(df[key_vars].describe().T[["mean", "std", "min", "50%", "max"]].round(2))

# %%
def group_summary(col):
    return df.groupby(TARGET)[col].agg(["count", "mean", "median", "std"]).round(2).reindex(order)

for col in key_vars:
    print(f"\n{col} by outcome")
    print(group_summary(col))

# %% [markdown]
# ## 4. Box plots and histograms by outcome

# %%
def boxplot_by_outcome(col, title):
    data = [df.loc[df[TARGET] == o, col] for o in order]
    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=order)
    plt.title(title)
    plt.tight_layout(); plt.show()

boxplot_by_outcome(ADM, "Admission grade by outcome")
boxplot_by_outcome(SEM1, "First-semester grade by outcome")

# %%
plt.figure(figsize=(8, 4))
for o, c in zip(order, colors):
    plt.hist(df.loc[df[TARGET] == o, AGE], bins=20, alpha=0.5, label=o, color=c)
plt.legend(); plt.title("Age at enrolment by outcome"); plt.tight_layout(); plt.show()

# Retrospective view only (excluded from the model to avoid temporal leakage)
boxplot_by_outcome(SEM2, "Second-semester grade by outcome (retrospective view)")

# %% [markdown]
# ## 5. Hypothesis test: Kruskal-Wallis
# H0: first-semester grades have the same distribution in all three outcome groups.
# H1: at least one group differs. alpha = 0.05.

# %%
def kruskal_by_outcome(col):
    groups = [df.loc[df[TARGET] == o, col].dropna() for o in order]
    H, p = stats.kruskal(*groups)
    eps2 = H / (len(df) - 1)          # epsilon-squared effect size
    return H, p, eps2

H, p, eps2 = kruskal_by_outcome(SEM1)
print(f"1st-sem grade : H = {H:.3f}, p = {p:.4e}, epsilon^2 = {eps2:.3f}")
print("Decision:", "Reject H0" if p < 0.05 else "Fail to reject H0")

H2, p2, eps2b = kruskal_by_outcome(ADM)
print(f"Admission grade: H = {H2:.3f}, p = {p2:.4e}, epsilon^2 = {eps2b:.3f}")

# %% [markdown]
# Kruskal-Wallis does not say *which* groups differ. Optional post-hoc: pairwise
# Mann-Whitney U tests with Bonferroni correction (a simple alternative to Dunn's test).

# %%
from itertools import combinations
pairs = list(combinations(order, 2))
for a, b in pairs:
    u, pv = stats.mannwhitneyu(df.loc[df[TARGET] == a, SEM1],
                               df.loc[df[TARGET] == b, SEM1])
    print(f"{a} vs {b}: U = {u:.0f}, adjusted p = {min(pv * len(pairs), 1):.3e}")

# %% [markdown]
# ## 6. Preprocessing and classification model
# - Drop every column containing "2nd sem" (not available at an early-warning point).
# - Category-code columns are one-hot encoded; true numeric columns are scaled.
# - 80/20 stratified split, seed 42; the whole pipeline is fit on training data only.

# %%
y = df[TARGET]
X = df.drop(columns=[TARGET])
leak_cols = [c for c in X.columns if "2nd sem" in c.lower()]
X = X.drop(columns=leak_cols)
print("Dropped second-semester columns:", len(leak_cols))

# Category codes stored as numbers (edit if your column names differ)
categorical_cols = [c for c in [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation"] if c in X.columns]

# Binary indicators stay as 0/1 numeric columns
numeric_cols = [c for c in X.columns if c not in categorical_cols]

print("Categorical:", len(categorical_cols), "| Numeric/binary:", len(numeric_cols))

# %%
preprocess = ColumnTransformer([
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), numeric_cols),
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print("Train:", X_train.shape[0], "| Test:", X_test.shape[0])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %% [markdown]
# ## 7. Evaluation (per-class metrics matter more than accuracy alone)

# %%
print(classification_report(y_test, y_pred, labels=order, digits=3))
print("Accuracy   :", round(accuracy_score(y_test, y_pred), 3))
print("Macro F1   :", round(f1_score(y_test, y_pred, average="macro"), 3))
print("Weighted F1:", round(f1_score(y_test, y_pred, average="weighted"), 3))

# %%
cm = confusion_matrix(y_test, y_pred, labels=order)
print(pd.DataFrame(cm, index=[f"Actual {o}" for o in order],
                   columns=[f"Pred {o}" for o in order]))

ConfusionMatrixDisplay(cm, display_labels=order).plot(cmap="Greens")
plt.title("Confusion matrix (held-out test set)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Interpretation and limitations
# - First-semester grade differs strongly across outcomes; admission grade and age show smaller, overlapping differences.
# - Accuracy hides a weak class: the Enrolled class has low recall because it is the smallest and sits between the other two.
# - These are associations, not causes. Use the model only to support human-led, supportive outreach (tutoring, advising), never automatic penalties.
# - Check that every predictor is available at the intended prediction time, and validate on later cohorts.
