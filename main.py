import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_squared_log_error, median_absolute_error, explained_variance_score
)

df = pd.read_csv("data/Exam_Score_Prediction.csv")
print("Početni oblik:", df.shape)

if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)

df["sleep_quality"] = df["sleep_quality"].map({"poor":1,"average":2,"good":3})
df["facility_rating"] = df["facility_rating"].map({"low":1,"medium":2,"high":3})
df["exam_difficulty"] = df["exam_difficulty"].map({"easy":1,"moderate":2,"hard":3})

print("\nNedostajuće vrijednosti nakon mapiranja:")
print(df.isnull().sum())

df = pd.get_dummies(df, columns=["gender","course","internet_access","study_method"], drop_first=True)

X = df.drop("exam_score", axis=1)
y = df["exam_score"]

X_trening, X_test, y_trening, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nOblik trening skupa:", X_trening.shape)
print("Oblik test skupa:", X_test.shape)

linearni_model = LinearRegression()
linearni_model.fit(X_trening, y_trening)
predikcija_lr = linearni_model.predict(X_test)

random_forest_model = RandomForestRegressor(n_estimators=200, random_state=42)
random_forest_model.fit(X_trening, y_trening)
predikcija_rf = random_forest_model.predict(X_test)

def evaluacija_svih_metrika(stvarne, predikcije, naziv):
    r2 = r2_score(stvarne, predikcije)
    mae = mean_absolute_error(stvarne, predikcije)
    rmse = np.sqrt(mean_squared_error(stvarne, predikcije))
    msle = mean_squared_log_error(stvarne, predikcije)
    medae = median_absolute_error(stvarne, predikcije)
    evs = explained_variance_score(stvarne, predikcije)
    
    print(f"\n{naziv}")
    print("-"*50)
    print("R2:                ", round(r2,4))
    print("MAE:               ", round(mae,4))
    print("RMSE:              ", round(rmse,4))
    print("MSLE:              ", round(msle,4))
    print("Medijan AE:        ", round(medae,4))
    print("Objašnjena varijanca:", round(evs,4))
    
    return r2, mae, rmse, msle, medae, evs

metrike_lr = evaluacija_svih_metrika(y_test, predikcija_lr, "Linearna regresija")
metrike_rf = evaluacija_svih_metrika(y_test, predikcija_rf, "Random Forest")

lr_cv = cross_val_score(linearni_model, X, y, cv=5, scoring="r2")
rf_cv = cross_val_score(random_forest_model, X, y, cv=5, scoring="r2")

print("\nRezultati unakrsne validacije (R2 - 5 presjeka)")
print("-"*50)
print("Linearna regresija CV prosjek R2:", round(lr_cv.mean(),4))
print("Random Forest CV prosjek R2:", round(rf_cv.mean(),4))

rezultati = pd.DataFrame({
    "Model": ["Linearna regresija","Random Forest"],
    "R2": [metrike_lr[0], metrike_rf[0]],
    "MAE": [metrike_lr[1], metrike_rf[1]],
    "RMSE": [metrike_lr[2], metrike_rf[2]],
    "MSLE": [metrike_lr[3], metrike_rf[3]],
    "Medijan AE": [metrike_lr[4], metrike_rf[4]],
    "Objašnjena varijanca": [metrike_lr[5], metrike_rf[5]],
    "CV prosjek R2 (5-fold)": [lr_cv.mean(), rf_cv.mean()]
})

rezultati = rezultati.round(4)

print("\nProfesionalna usporedba modela - proširene metrike")
print("-"*60)
print(rezultati)


plt.figure(figsize=(8,6))
plt.scatter(y_test, predikcija_lr, alpha=0.5, label="Linearna regresija")
plt.scatter(y_test, predikcija_rf, alpha=0.5, label="Random Forest")
plt.plot([0,100],[0,100],'r--')
plt.xlabel("Stvarni rezultat ispita")
plt.ylabel("Predviđeni rezultat ispita")
plt.title("Predviđeno vs Stvarno")
plt.legend()
plt.grid(True)

for model, naziv in zip([linearni_model, random_forest_model], ["Linearna regresija","Random Forest"]):
    velicine_treninga, trening_rezultati, test_rezultati = learning_curve(
        model, X, y, cv=5, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1,1.0,5)
    )
    
    prosjek_trening = np.mean(trening_rezultati, axis=1)
    prosjek_test = np.mean(test_rezultati, axis=1)
    
    plt.figure(figsize=(8,6))
    plt.plot(velicine_treninga, prosjek_trening, 'o-', label="Trening R2")
    plt.plot(velicine_treninga, prosjek_test, 'o-', label="CV R2")
    plt.xlabel("Broj trening primjera")
    plt.ylabel("R2 rezultat")
    plt.title(f"Krivulja učenja - {naziv}")
    plt.legend()
    plt.grid(True)

plt.figure(figsize=(8,6))
sns.histplot(y_test - predikcija_lr, kde=True, color='blue', label="Reziduali - Linearna regresija", alpha=0.5)
sns.histplot(y_test - predikcija_rf, kde=True, color='orange', label="Reziduali - Random Forest", alpha=0.5)
plt.xlabel("Greška (Stvarno - Predikcija)")
plt.ylabel("Broj primjera")
plt.title("Distribucija reziduala")
plt.legend()
plt.grid(True)

greske = pd.DataFrame({
    "Linearna regresija": y_test - predikcija_lr,
    "Random Forest": y_test - predikcija_rf
})

plt.figure(figsize=(8,6))
sns.boxplot(data=greske)
plt.title("Boxplot grešaka po modelu")
plt.ylabel("Greška (Stvarno - Predikcija)")
plt.grid(True)

metrike = ["R2", "MAE", "RMSE", "MSLE", "Medijan AE", "Objašnjena varijanca", "CV prosjek R2 (5-fold)"]
modeli = ["Linearna regresija","Random Forest"]

vrijednosti = [rezultati.loc[rezultati["Model"]==m, metrike].values.flatten() for m in modeli]

indeksi_inverzije = [1,2,3,4]
normalizirane_vrijednosti = []

for v in vrijednosti:
    kopija = v.astype(float)
    for i in indeksi_inverzije:
        kopija[i] = kopija[i] * -1
    normalizirane_vrijednosti.append(kopija)

x = np.arange(len(metrike))
sirina = 0.35

plt.figure(figsize=(12,6))
plt.bar(x - sirina/2, normalizirane_vrijednosti[0], sirina, label=modeli[0])
plt.bar(x + sirina/2, normalizirane_vrijednosti[1], sirina, label=modeli[1])
plt.xticks(x, metrike, rotation=30)
plt.ylabel("Performanse (veće = bolje)")
plt.title("Usporedba modela po metrikama")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()