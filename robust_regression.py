import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

class RobustRegression:
    def __init__(self, threshold=None):
        # Inicjalizacja modelu i ustawienie progu odległości Cooka
        self.threshold = threshold
        self.model = LinearRegression()
        self.weights = None

    def fit(self, X, y):
        # Dopasowanie modelu odpornej regresji liniowej do danych
        n, p = X.shape  # Liczba obserwacji (n) i zmiennych objaśniających (p)
        self.weights = np.ones(n)  # Inicjalizacja wag dla wszystkich obserwacji na 1

        # Jeśli próg jest None, ustawiamy dynamicznie na 4/n
        if self.threshold is None:
            self.threshold = 4 / n

        for _ in range(5):  # Iteracyjne dopasowanie modelu, 5 iteracji
            # Skalowanie danych wejściowych i wyjściowych za pomocą pierwiastka z wag
            weighted_X = X * np.sqrt(self.weights).reshape(-1, 1)
            weighted_y = y * np.sqrt(self.weights)

            # Dopasowanie modelu liniowego
            self.model.fit(weighted_X, weighted_y)
            y_pred = self.model.predict(X)  # Predykcja wartości

            # Obliczenie reszt i leverage dla każdej obserwacji
            residuals = y - y_pred
            mse = mean_squared_error(y, y_pred)  # Średni błąd kwadratowy
            leverage = (1 / n) + (np.sum((X - np.mean(X, axis=0)) ** 2, axis=1) / np.sum((X - np.mean(X, axis=0)) ** 2))

            # Obliczenie odległości Cooka
            cooks_distance = (residuals ** 2 / (p * mse)) * (leverage / (1 - leverage) ** 2)

            # Aktualizacja wag: mniejsze wagi dla obserwacji odstających
            self.weights = np.where(cooks_distance > self.threshold, 0.5, 1)

    def predict(self, X):
        # Predykcja na podstawie dopasowanego modelu
        return self.model.predict(X)

    def coefficients(self):
        # Pobranie współczynników regresji i wyrazu wolnego
        return self.model.coef_, self.model.intercept_

def bootstrap_error(X, y, model, n_bootstrap):
    # Szacowanie błędów współczynników metodą bootstrap
    coef_samples = []  # Lista przechowująca współczynniki dla każdej iteracji
    intercept_samples = []  # Lista przechowująca wyrazy wolne dla każdej iteracji

    for _ in range(n_bootstrap):
        # Losowe próbkowanie z powtórzeniami
        X_resampled, y_resampled = resample(X, y)
        model.fit(X_resampled, y_resampled)
        coef, intercept = model.coefficients()
        coef_samples.append(coef)
        intercept_samples.append(intercept)

    # Konwersja wyników
    coef_samples = np.array(coef_samples)
    intercept_samples = np.array(intercept_samples)

    # Obliczenie odchylenia standardowego dla błędów
    coef_errors = coef_samples.std(axis=0)  # Błędy dla współczynników
    intercept_error = intercept_samples.std()  # Błąd dla wyrazu wolnego

    return coef_errors, intercept_error

def main(X, y, threshold=None, n_bootstrap=1000):
    # Dopasowanie modelu odpornej regresji
    model = RobustRegression(threshold=threshold)
    model.fit(X, y)
    coef, intercept = model.coefficients()

    # Oszacowanie błędów metodą bootstrap
    coef_errors, intercept_error = bootstrap_error(X, y, model, n_bootstrap=n_bootstrap)

    return {
        "coefficients": coef,
        "intercept": intercept,
        "coef_errors": coef_errors,
        "intercept_error": intercept_error
    }

if __name__ == "__main__":
    # Przykładowe dane
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 3 * X.squeeze() + np.random.randn(100) * 2

    # Dodanie odstających wartości
    X = np.vstack([X, [[20], [22]]])
    y = np.append(y, [100, -50])

    # Wywołanie funkcji main() z danymi
    results = main(X, y, threshold=None, n_bootstrap=1000)

    # Wyświetlenie wyników
    print("Współczynniki:", results["coefficients"])
    print("Wyraz wolny:", results["intercept"])
    print("Błędy współczynników:", results["coef_errors"])
    print("Błąd wyrazu wolnego:", results["intercept_error"])
