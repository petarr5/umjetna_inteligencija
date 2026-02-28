UPUTE ZA POKRETANJE PROGRAMA

Ovaj program implementira regresijske modele (Linearna regresija i Random Forest) za predikciju rezultata ispita te generira evaluacijske metrike i grafičke prikaze rezultata.

---

1. PREDUVJETI

Potrebno je imati instalirano:

* Python 3.8 ili noviji
* pip (Python package manager)

Preporučena razvojna okruženja:

* Visual Studio Code
* PyCharm
* Jupyter Notebook

Python se može preuzeti sa službene stranice:
[https://www.python.org](https://www.python.org)

---

2. INSTALACIJA POTREBNIH BIBLIOTEKA

U terminalu (Command Prompt, PowerShell ili Terminal) pokrenuti sljedeću naredbu:

pip install pandas numpy matplotlib seaborn scikit-learn

Ako koristite Anacondu:

conda install pandas numpy matplotlib seaborn scikit-learn

---

3. STRUKTURA PROJEKTA

Projekt mora imati sljedeću strukturu direktorija:

projekt/
│
├── data/
│   └── Exam_Score_Prediction.csv
│
└── main.py

Važno:
Datoteka Exam_Score_Prediction.csv mora se nalaziti unutar mape "data" jer se u kodu učitava pomoću:

pd.read_csv("data/Exam_Score_Prediction.csv")

Ako datoteka nije u toj mapi, pojavit će se greška FileNotFoundError.

---

4. POKRETANJE PROGRAMA

Korak 1:
Otvoriti terminal i pozicionirati se u direktorij gdje se nalazi main.py.

Korak 2:
Pokrenuti program naredbom:

python main.py

---

5. ŠTO PROGRAM RADI

Nakon pokretanja program će:

1. Učitati podatke iz CSV datoteke.
2. Ukloniti stupac student_id (ako postoji).
3. Mapirati ordinalne kategorije u numeričke vrijednosti.
4. Primijeniti one-hot encoding na nominalne varijable.
5. Podijeliti podatke na trening i test skup (80% trening, 20% test).
6. Trenirati dva modela:

   * Linearnu regresiju
   * Random Forest (200 stabala)
7. Izračunati regresijske metrike:

   * R2
   * MAE
   * RMSE
   * MSLE
   * Medijan apsolutne pogreške
   * Objašnjenu varijancu
8. Provesti 5-fold cross-validation.
9. Ispisati tabličnu usporedbu modela.
10. Prikazati grafove:

    * Stvarno vs Predviđeno
    * Krivulje učenja
    * Distribucija reziduala
    * Boxplot grešaka
    * Usporedba modela po metrikama

Grafovi će se automatski otvoriti u novim prozorima.

---

6. MOGUĆE GREŠKE I RJEŠENJA

ModuleNotFoundError
Rješenje: Instalirati nedostajuću biblioteku pomoću pip naredbe.

FileNotFoundError
Rješenje: Provjeriti nalazi li se Exam_Score_Prediction.csv unutar mape data.

---

Ako se program pokrene bez grešaka, na ekranu će se prikazati evaluacijski rezultati modela te grafičke vizualizacije njihove usporedbe.
