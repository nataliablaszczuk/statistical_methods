import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)
# =====================================
# 1. Wczytanie danych
# =====================================
file_path = "statistical_methods/EU_Health_Indicators_2019.xlsx"
data = pd.read_excel(file_path)

# Ustawiamy kolumnę 'Country' jako indeks
data.set_index("Country", inplace=True)

# =====================================
# 2. Wstępne przetwarzanie danych
# =====================================
print(data.columns)
# Usuwanie duplikatów
data = data.drop_duplicates()
print(data.columns)

# Konwersja kolumn na wartości liczbowe (float), pozostałe -> NaN
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Interpolacja braków i usunięcie wierszy nadal zawierających NaN
data = data.interpolate(method='linear').dropna()
print(data.columns)

print("Podgląd danych po wstępnym przetworzeniu:")
print(data.head())
print("\nRozmiar zbioru:", data.shape)

# =====================================
# Analiza korelacji & usuwanie zmiennych silnie skorelowanych
# =====================================
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Macierz korelacji zmiennych (przed usunięciem >0.9)')
plt.show()

upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
if len(high_corr) > 0:
    print("Usuwamy zmiennie silnie skorelowane:", high_corr)
    data = data.drop(columns=high_corr)

data = data.drop(columns="safe_water_proportion")
print("\nFinalny zestaw zmiennych (kolumn) po eliminacji korelacji > 0.9:")
print(list(data.columns))

for col in data.columns:
    print(f"\n=== Zmienna: {col} ===")
    desc = data[col].describe()
    print("Parametry opisowe:")
    print(desc)  # count, mean, std, min, 25%, 50%, 75%, max
    
    # Który kraj ma wartość minimalną, a który maksymalną?
    country_min = data[col].idxmin()
    country_max = data[col].idxmax()
    print(f"Minimalna wartość: {desc['min']} (kraj: {country_min})")
    print(f"Maksymalna wartość: {desc['max']} (kraj: {country_max})")
    
    # Dodatkowe miary kształtu rozkładu
    skewness = data[col].skew()
    kurtosis = data[col].kurt()
    print(f"Skośność (skewness): {skewness:.3f}")
    print(f"Kurtoza (kurtosis): {kurtosis:.3f}")

# =====================================
# 3. Transformacje i Standaryzacja
# =====================================
countries = data.index
X = data.values

# =====================================
# Logarytmiczne przekształcenie zmiennych odstających
# =====================================
# Lista zmiennych do przekształcenia
variables_to_transform = ['maternal_mortality_ratio', 'suicide_mortality_rate']

# Dodanie stałej do zmiennych, aby uniknąć problemów z wartościami zero
epsilon = 1e-6  # Mała wartość dodawana do zmiennych
for var in variables_to_transform:
    data[var] = np.log(data[var] + epsilon)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================
# 4. Analiza głównych składowych (PCA)
# =====================================

#Pełne PCA na danych skalowanych
pca_full = PCA()
pca_full.fit(X_scaled)

# Wariancja wyjaśniana przez kolejne składowe
var_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(var_ratio)

# Wyświetlanie proporcji wariancji
print("Proporcja wyjaśnianej wariancji przez kolejne składowe PCA:")
for i, (vr, cum) in enumerate(zip(var_ratio, cumulative_variance), start=1):
    print(f"PC{i}: {vr:.3f} (łączna: {cum:.3f})")

# Wykres osypiska (scree plot) z graficzną interpretacją
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(var_ratio)+1), var_ratio, alpha=0.6, label='Pojedyncza składowa')
plt.plot(range(1, len(var_ratio)+1), cumulative_variance, marker='o', color='red', label='Łączna wariancja')
plt.axhline(y=0.85, color='green', linestyle='--', label='85% wariancji')
plt.axvline(x=np.argmax(cumulative_variance >= 0.85) + 1, color='blue', linestyle='--', label='Liczba składowych')
plt.xlabel('Numer składowej')
plt.ylabel('Proporcja wyjaśnionej wariancji')
plt.title('Wariancja wyjaśniana przez składowe PCA (osypisko)')
plt.legend()
plt.show()

# Wybór liczby składowych, by pokryć >=85% wariancji
k = np.argmax(cumulative_variance >= 0.85) + 1
print(f"\nWybrana liczba składowych (pokrycie >=85%): {k}")

# Dopasowanie PCA do wybranej liczby składowych
pca = PCA(n_components=k)
X_pca = pca.fit_transform(X_scaled)
print("Kształt macierzy po PCA:", X_pca.shape)

# Dla wizualizacji w 2D (opcjonalne)
pca_2d = PCA(n_components=2).fit(X_scaled)
X_pca_2d = pca_2d.transform(X_scaled)

# =====================================
# Wektory własne (ładunki) do interpretacji głównych składowych
# =====================================

components = pd.DataFrame(
    data=pca.components_[:6],  # Pierwsze 6 głównych składowych
    columns=data.columns,  # Nazwy zmiennych
    index=[f'PC{i+1}' for i in range(6)]  # Indeksy PC1, PC2, ...
)
print("\nWektory własne (ładunki) dla pierwszych 6 głównych składowych:")
print(components)

# Interpretacja logiczna: Wykres ładunków dla pierwszych 6 głównych składowych
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.bar(components.columns, np.abs(components.iloc[i]), label=f'PC{i+1}')
plt.xlabel('Zmienne')
plt.ylabel('Wartość bezwzględna współczynników')
plt.title('Wektory własne (ładunki) dla 6 pierwszych głównych składowych')
plt.legend()
plt.xticks(rotation=90)
plt.show()

# =====================================
# 5. Metoda taksonomiczna - klasteryzacja k-means
# =====================================

# --- Klasteryzacja K-means na oryginalnym zbiorze X_scaled ---
kmeans_original = KMeans(n_clusters=3, random_state=42)
labels_original = kmeans_original.fit_predict(X_scaled)
sil_original = silhouette_score(X_scaled, labels_original)
print(f"K-means (3 klastry) na oryginalnych zmiennych -> Silhouette = {sil_original:.3f}")

# --- Klasteryzacja K-means na PCA (k składowych) ---
kmeans_pca = KMeans(n_clusters=3, random_state=42)
labels_pca = kmeans_pca.fit_predict(X_pca)  
sil_pca = silhouette_score(X_pca, labels_pca)
print(f"K-means (3 klastry) na {k} składowych PCA -> Silhouette = {sil_pca:.3f}")

# Dla wizualizacji 2D (PC1, PC2) weźmiemy X_pca_2d
kmeans_2d = KMeans(n_clusters=3, random_state=42)
labels_2d = kmeans_2d.fit_predict(X_pca_2d)

# =====================================
# 5. Metoda taksonomiczna - dendrogram
# =====================================

# Z = linkage(X_scaled, method='ward')
# plt.figure(figsize=(10,6))
# dendrogram(Z, labels=countries.values, leaf_rotation=90, leaf_font_size=10)
# plt.title("Dendrogram (Ward) - Oryginalne dane")
# plt.xlabel("Kraje")
# plt.ylabel("Odległość")
# plt.show()

# # Przykład: tniemy dendrogram na 3 skupienia
# labels_hier = fcluster(Z, t=3, criterion='maxclust')

# print("Hierarchiczna (3 skupienia) - przypisanie krajów:")
# for c, country in zip(labels_hier, countries):
#     print(f"{country}: cluster {c}")

# import matplotlib.cm as cm

# pca_df = pd.DataFrame(X_pca_2d, columns=["PC1", "PC2"], index=countries)
# # Dodajemy etykiety klastrów
# pca_df["KMeans_Original"] = labels_original
# pca_df["KMeans_PCA"] = labels_pca  # te na k wymiarach (ale tu wizualizujemy w 2D)
# pca_df["KMeans_2D"] = labels_2d
# pca_df["HierCluster"] = labels_hier

# # Wizualizacja klastrów K-means (na oryginalnych danych) w układzie PC1-PC2
# plt.figure(figsize=(10,6))
# sns.scatterplot(
#     data=pca_df,
#     x="PC1", y="PC2",
#     hue="KMeans_Original",
#     palette="Set1",
#     s=100
# )
# for label in pca_df.index:
#     plt.text(x=pca_df["PC1"][label]+0.03, y=pca_df["PC2"][label]+0.03, s=label, fontsize=9)
# plt.title("K-means (na oryginalnych danych), wizualizacja w przestrzeni PC1-PC2")
# plt.legend(title='Cluster')
# plt.grid(True)
# plt.show()

# # Wizualizacja klastrów K-means (na PCA) w układzie PC1-PC2
# plt.figure(figsize=(10,6))
# sns.scatterplot(
#     data=pca_df,
#     x="PC1", y="PC2",
#     hue="KMeans_PCA",
#     palette="Set2",
#     s=100
# )
# for label in pca_df.index:
#     plt.text(x=pca_df["PC1"][label]+0.03, y=pca_df["PC2"][label]+0.03, s=label, fontsize=9)
# plt.title("K-means (na k składowych PCA), wizualizacja w PC1-PC2")
# plt.legend(title='Cluster')
# plt.grid(True)
# plt.show()

# # Wizualizacja klastrów hierarchicznych w układzie PC1-PC2
# plt.figure(figsize=(10,6))
# sns.scatterplot(
#     data=pca_df,
#     x="PC1", y="PC2",
#     hue="HierCluster",
#     palette="Set3",
#     s=100
# )
# for label in pca_df.index:
#     plt.text(x=pca_df["PC1"][label]+0.03, y=pca_df["PC2"][label]+0.03, s=label, fontsize=9)
# plt.title("Hierarchiczna (3 skupienia), wizualizacja w PC1-PC2")
# plt.legend(title='Cluster')
# plt.grid(True)
# plt.show()