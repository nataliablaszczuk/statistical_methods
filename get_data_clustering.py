### Statystyczne metody analizy wielowymiarowej - porównanie kraj Europy pod kątem wskaźników zdrowotnych - analiza skupień - clustering (n-średnich) i metoda głównych składowych (PCA)

# Cel - zbadanie podobieństw i róznic między krajami Europy pod względem wybranych wskaźników zdrowotnych
# Hipotezy:
# 1. Kraje o podobnym stopniu rozwoju gospodarczego wykazują podobieństwa w wybranych wskaźnikach zdrowotnych
# 2. Analiza głównych składowych pozwoli na redukcję wymiaru danych przy zachowaniu kluczowych informacji

# Zakres merytoryczny: zdrowotność populacji (np. długość zycia, umieralność niemowląt, wydatki na zdrowie, dostępność lekarzy)
# Zakres terytorialny: kraje Europy

# Pobranie danych z WHO API

import requests
import pandas as pd
import numpy as np

# Lista krajów UE
eu_countries = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", 
    "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", 
    "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", 
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"
]

# Lista kodów indykatorów (przykładowe)
indicators = [
    "WHOSIS_000001",  # 1. Life expectancy at birth
    "MDG_0000000001",  # 2. Infant mortality rate (deaths per 1000 live births)
    "NCD_BMI_30A",  # 3. Obesity prevalence (BMI ≥ 30), Age-standardized prevalence of obesity among adults (18+ years)
    "HWF_0001", # 4. Density of physicians (per 10 000 population)
    "NCDMORT3070", # 5. Probability of premature mortality from NCDs; Mortality rate attributed to cardiovascular disease
    "SDGSUICIDE", # 6. Suicide Mortality Rate
    "WHOSIS_000003", # 7. Neonatal mortality rate (per 1000 live births)
    "WHOSIS_000002", # 8. Healthy life expectancy at birth
    "NCD_HYP_PREVALENCE_A", # 9. Prevalence of hypertension among adults aged 30 to 79.
    "MDG_0000000007", # 10. Under-five mortality rate (per 1000 live births)
    "MDG_0000000026", # 11. Maternal Mortality Ratio
    "WSH_WATER_SAFELY_MANAGED", # 12. Proportion of population using safely managed drinking-water services (%)
    "SA_0000001688", # 13. Total alcohol per capita (>= 15 years of age) consumption (litres of pure alcohol)
    "HWF_0006", # 14. Density of nursing and midwifery personnel (per 10 000 population)
    "GHED_GGHE-DGGE_SHA2011" # 15. Domestic general government health expenditure (GGHE-D) as percentage of general government expenditure (GGE) (%)
]

year = 2019

def fetch_data_for_country_and_indicator(country_code, indicator_code, year):
    url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
    filters = f"$filter=TimeDim eq {year} and SpatialDim eq '{country_code}'"
    full_url = f"{url}?{filters}"
    print(f"Fetching data for country: {country_code}, indicator: {indicator_code}")
    response = requests.get(full_url)
    if response.status_code == 200:
        data = response.json()
        if data.get("value"):
            # Pobierz tylko wartość wskaźnika
            for record in data.get("value", []):
                if record["SpatialDim"] == country_code and record["TimeDim"] == year:
                    return record.get("Value")
    else:
        print(f"Error fetching data for country {country_code}, indicator {indicator_code}: {response.status_code}")
        print(f"Response: {response.text}")
    return None

# Tworzenie pustej listy do przechowywania wyników
result_data = []

# Pobieranie danych dla wszystkich krajów i indykatorów
for country in eu_countries:
    country_data = {"Country": country}
    for indicator in indicators:
        value = fetch_data_for_country_and_indicator(country, indicator, year)
        country_data[indicator] = value
    result_data.append(country_data)

# Konwersja wyników do DataFrame
final_data = pd.DataFrame(result_data)

# Oczyszczanie danych: wyodrębnienie wartości przed spacją, jeśli występuje
for column in final_data.columns[1:]:
    final_data[column] = final_data[column].apply(lambda x: str(x).split(" ")[0] if isinstance(x, str) else x)

final_data.to_excel("EU_Health_Indicators_2019.xlsx", index=False, engine='openpyxl')
print("Data saved to 'EU_Health_Indicators_2019.xlsx'")

# Wyświetlenie danych
print(final_data)