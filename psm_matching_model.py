import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import ttest_ind


# =============================================================================
#  Wybór zbioru danych i wstępna eksploracja
# =============================================================================

# Pobranie danych Adult UCI
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data = pd.read_csv(url, header=None, names=columns, skipinitialspace=True)

# Szybka wstępna eksploracja
print("Rozmiar pierwotny zbioru:", data.shape)
print(data.head())
print(data.describe())

# Sprawdzenie i usunięcie ewentualnych braków
data = data.replace('?', np.nan)
print("Braki (NaN) per kolumna:")
print(data.isna().sum())
data.dropna(inplace=True)
print("Rozmiar po usunięciu braków:", data.shape)

# Rozkład zmiennej "income"
sns.countplot(data=data, x='income')
plt.title("Rozkład dochodów (>50K vs. <=50K)")
plt.show()

# Rozkład wieku
sns.histplot(data=data, x='age', kde=True, bins=20)
plt.title('Rozkład wieku')
plt.show()


# =============================================================================
# 2. Implementacja analizy w Pythonie
# =============================================================================

# Definicja Treatment i Outcome
data['Treatment'] = data['education-num'].apply(lambda x: 1 if x >= 13 else 0)
# Treatment=1 (Bachelors i wyżej), 0 (reszta)

data['Outcome'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Przygotowanie cech do modelu Propensity Score
features = [
    "age", "hours-per-week", "capital-gain", "capital-loss",
    "workclass", "marital-status", "occupation", "relationship", 
    "race", "sex", "native-country"
]
X = data[features]
y = data['Treatment']

# Transformacje kolumn
numeric_cols = ["age", "hours-per-week", "capital-gain", "capital-loss"]
cat_cols = ["workclass", "marital-status", "occupation", "relationship", 
            "race", "sex", "native-country"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop='first'), cat_cols)
])

# Pipeline -> preprocessor + LogisticRegression
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("logistic_classifier", LogisticRegression(max_iter=1000))
])

pipe.fit(X, y)

# Ocena modelu
ps = pipe.predict_proba(X)[:, 1]
data["Propensity_Score"] = ps

roc_auc = roc_auc_score(y, ps)
f1 = f1_score(y, pipe.predict(X))
print(f"[Model Propensity Score] ROC AUC = {roc_auc:.4f}, F1-score = {f1:.4f}")

# Podział na treated i control
treated = data[data["Treatment"] == 1].copy()
control = data[data["Treatment"] == 0].copy()
print("Liczba w grupie treated:", len(treated))
print("Liczba w grupie control:", len(control))

# Sprawdzenie overlap w PS
plt.figure(figsize=(10,6))
sns.kdeplot(treated["Propensity_Score"], label="Treated", fill=True)
sns.kdeplot(control["Propensity_Score"], label="Control", fill=True)
plt.title("Rozkład Propensity Score (treated vs. control)")
plt.legend()
plt.show()

# Matching (1 nearest neighbor + caliper)
caliper = 0.2 * np.std(data["Propensity_Score"])
print(f"Caliper = {caliper:.5f}")

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["Propensity_Score"]])

distances, indices = nn.kneighbors(treated[["Propensity_Score"]])
distances = distances.flatten()
indices = indices.flatten()

# filtrowanie pary z distance <= caliper
mask = distances <= caliper
matched_control_indices = indices[mask]

matched_treated = treated.iloc[mask].copy()
matched_control = control.iloc[matched_control_indices].copy()

matched_data = pd.concat([matched_treated, matched_control], axis=0)\
                  .reset_index(drop=True)

print("Matched treated:", len(matched_treated), 
      "| Matched control:", len(matched_control))
print("Łącznie w matched_data:", len(matched_data))


# =============================================================================
# Prezentacja i interpretacja wyników
# =============================================================================

def balance_test(group1, group2, colname):
    return ttest_ind(group1[colname], group2[colname])[1]

matched_treated_part = matched_data[matched_data["Treatment"] == 1]
matched_control_part = matched_data[matched_data["Treatment"] == 0]

print("\nTEST BALANSU PO MATCHOWANIU")
balance_features = ["age", "hours-per-week", "capital-gain", "capital-loss"]
for col in balance_features:
    pval = balance_test(matched_treated_part, matched_control_part, col)
    print(f"Cecha: {col:<15} => p-value = {pval:.4f}")

# Obliczenie ATT
treated_outcome_matched = matched_treated_part["Outcome"]
control_outcome_matched = matched_control_part["Outcome"]
ATT = treated_outcome_matched.mean() - control_outcome_matched.mean()

print(f"\nATT (po dopasowaniu) = {ATT:.4f}")

# Test t-Studenta na różnicę w Outcome
t_stat, p_value = ttest_ind(treated_outcome_matched, control_outcome_matched)
print(f"\nTEST t-Studenta] Różnica w Outcome (matched):")
print(f"  t = {t_stat:.4f}, p-value = {p_value:.4f}")
if p_value < 0.05:
    print("  => Różnice statystycznie istotne (p < 0.05).")
else:
    print("  => Różnice NIE są statystycznie istotne (p >= 0.05).")


# Efekt wielkości - Cohen's d
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.var(x) + (ny-1)*np.var(y)) / dof)

effect_size = cohen_d(treated_outcome_matched, control_outcome_matched)
print(f"Cohen's d (wielkość efektu): {effect_size:.4f}")
