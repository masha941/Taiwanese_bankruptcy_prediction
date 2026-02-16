import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

print("Učitavanje podataka")
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("GRESKA: Molim vas preuzmite 'data.csv' sa UCI repozitorijuma i stavite ga u isti folder.")
    exit()

df.fillna(0, inplace=True)

target_col = df.columns[0]
X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"Dimenzije skupa: {df.shape}")
print(f"Raspodela klasa:\n{y.value_counts()}")

print("\nGenerisanje 2D vizuelizacije")
scaler_viz = StandardScaler()
X_scaled_viz = scaler_viz.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_viz)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.6)
plt.title('2D PCA Vizuelizacija podataka (Bankrot vs Stabilno)')
plt.xlabel('Komponenta 1')
plt.ylabel('Komponenta 2')
plt.savefig('vizuelizacija_2d.png')
print("Slika 'vizuelizacija_2d.png' je sačuvana.")

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_scaled.to_csv('processed_data.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

selector = SelectKBest(f_classif, k=10)
X_train_red = selector.fit_transform(X_train_res, y_train_res)
X_test_red = selector.transform(X_test)
selected_features = X.columns[selector.get_support()]
print(f"\nOdabrani redukovani atributi: {list(selected_features)}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True)
}

def evaluate_models(models, X_tr, y_tr, X_te, y_te, suffix):
    results = {}
    print(f"\n--- Rezultati za {suffix} skup atributa ---")
    for name, model in models.items():
        print(f"Treniranje: {name}...")
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)

        print(f"Model: {name} | Accuracy: {acc:.4f}")
        print(classification_report(y_te, y_pred))

        results[name] = acc

        filename = f"model_{name.replace(' ', '_')}_{suffix}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    return results

results_full = evaluate_models(models, X_train_res, y_train_res, X_test, y_test, "FULL")

results_red = evaluate_models(models, X_train_red, y_train_res, X_test_red, y_test, "REDUCED")

plt.figure(figsize=(12, 6))
labels = list(models.keys())
x = np.arange(len(labels))
width = 0.35

val_full = [results_full[m] for m in labels]
val_red = [results_red[m] for m in labels]

plt.bar(x - width/2, val_full, width, label='Svi atributi')
plt.bar(x + width/2, val_red, width, label='Redukovani atributi (Top 10)')

plt.ylabel('Tačnost (Accuracy)')
plt.title('Poređenje tačnosti modela: Svi atributi vs Redukovani')
plt.xticks(x, labels)
plt.legend()
plt.savefig('poredjenje_modela.png')
print("\nAnaliza završena. Slike i modeli su sačuvani.")