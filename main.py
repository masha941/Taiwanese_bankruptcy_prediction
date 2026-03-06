import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
import pickle
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

COLOR_STABLE   = '#4878CF'   
COLOR_BANKRUPT = '#FFB6C1'   
COLOR_FULL     = '#4878CF'   
COLOR_REDUCED  = '#FFB6C1'   
PALETTE        = {0: COLOR_STABLE, 1: COLOR_BANKRUPT}

print("=" * 60)
print("FAZA 1: Učitavanje i osnovna analiza podataka")
print("=" * 60)

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
print(f"Broj atributa: {X.shape[1]}")
print(f"\nRaspodela klasa:")
vc = y.value_counts()
for cls, cnt in vc.items():
    label = "Bankrot" if cls == 1 else "Stabilno"
    print(f"  Klasa {cls} ({label}): {cnt} ({cnt/len(y)*100:.2f}%)")

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    vc.values,
    labels=['Stabilno (0)', 'Bankrot (1)'],
    colors=[COLOR_STABLE, COLOR_BANKRUPT],
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax.set_title('Raspodela klasa u skupu podataka', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('raspodela_klasa.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSlika 'raspodela_klasa.png' je sačuvana.")

print("\n" + "=" * 60)
print("FAZA 2: PCA vizuelizacija prostora podataka")
print("=" * 60)

scaler_viz = StandardScaler()
X_scaled_viz = scaler_viz.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_viz)

plt.figure(figsize=(10, 7))
for cls, color, label in [(0, COLOR_STABLE, 'Stabilno'), (1, COLOR_BANKRUPT, 'Bankrot')]:
    mask = y == cls
    plt.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=color, label=label, alpha=0.5, s=15, edgecolors='none'
    )
plt.title('2D PCA Vizuelizacija podataka — Bankrot vs Stabilno', fontsize=14, fontweight='bold')
plt.xlabel(f'Komponenta 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varijanse)')
plt.ylabel(f'Komponenta 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varijanse)')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('vizuelizacija_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("Slika 'vizuelizacija_2d.png' je sačuvana.")
print(f"  PCA objašnjava {sum(pca.explained_variance_ratio_)*100:.1f}% ukupne varijanse sa 2 komponente.")

print("\n" + "=" * 60)
print("FAZA 3: Preprocesiranje — skaliranje, SMOTE, selekcija atributa")
print("=" * 60)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_scaled.to_csv('processed_data.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Trening skup pre SMOTE:  {X_train.shape[0]} primeraka ({y_train.sum()} bankrota)")
print(f"Trening skup posle SMOTE: {X_train_res.shape[0]} primeraka ({y_train_res.sum()} bankrota)")
print(f"Test skup:               {X_test.shape[0]} primeraka ({y_test.sum()} bankrota)")

# Selekcija atributa
selector = SelectKBest(f_classif, k=10)
X_train_red = selector.fit_transform(X_train_res, y_train_res)
X_test_red  = selector.transform(X_test)
selected_features = X.columns[selector.get_support()]
feature_scores    = selector.scores_[selector.get_support()]

print(f"\nTop 10 odabranih atributa (po F-skoru):")
sorted_idx = np.argsort(feature_scores)[::-1]
for i, idx in enumerate(sorted_idx):
    print(f"  {i+1:2}. {selected_features[idx]:<45} F = {feature_scores[idx]:.2f}")

fig, ax = plt.subplots(figsize=(10, 6))
sorted_feats  = [selected_features[i] for i in sorted_idx]
sorted_scores = [feature_scores[i] for i in sorted_idx]
bars = ax.barh(sorted_feats[::-1], sorted_scores[::-1], color=COLOR_STABLE, edgecolor='white')
ax.set_xlabel('ANOVA F-skor', fontsize=12)
ax.set_title('Top 10 atributa po ANOVA F-testu', fontsize=14, fontweight='bold')
ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('top10_atributi.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSlika 'top10_atributi.png' je sačuvana.")

df_top10 = X_scaled[selected_features].copy()
df_top10[target_col] = y.values
corr = df_top10.corr()

plt.figure(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 340, as_cmap=True)  # plava-roze divergentna paleta
sns.heatmap(
    corr, mask=mask, cmap=cmap, center=0,
    square=True, linewidths=.5, annot=True, fmt='.2f', annot_kws={'size': 8},
    cbar_kws={"shrink": .8}
)
plt.title('Korelaciona matrica — Top 10 atributa + ciljna promenljiva', fontsize=13, fontweight='bold')
plt.xticks(rotation=35, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('korelaciona_matrica.png', dpi=150, bbox_inches='tight')
plt.close()
print("Slika 'korelaciona_matrica.png' je sačuvana.")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "SVM":                 SVC(kernel='rbf', probability=True)
}

def print_confusion_matrix(cm, model_name, suffix):
    """Štampa matricu konfuzije u terminalu u čitljivom formatu."""
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Matrica konfuzije — {model_name} ({suffix})")
    print(f"  {'':25} Predviđeno: Stabilno   Predviđeno: Bankrot")
    print(f"  {'Stvarno: Stabilno':<25} {tn:^22} {fp:^22}")
    print(f"  {'Stvarno: Bankrot':<25} {fn:^22} {tp:^22}")


def evaluate_models(models, X_tr, y_tr, X_te, y_te, suffix):
    """
    Trenira i evaluira svaki model, čuva rezultate.
    Vraća rečnik sa accuracy, odziv za klasu 1, precision za klasu 1 i sačuvane modele.
    """
    results = {}
    print(f"\n{'='*60}")
    print(f"REZULTATI: {suffix.upper()} SKUP ATRIBUTA")
    print(f"{'='*60}")

    for name, model in models.items():
        print(f"\n>>> Treniranje: {name}...")
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_te, y_pred)
        report_dict = classification_report(y_te, y_pred, output_dict=True)
        odziv_1     = report_dict['1']['recall']
        precision_1 = report_dict['1']['precision']
        f1_1        = report_dict['1']['f1-score']

        print(f"    Tačnost  : {acc:.4f}")
        print(f"    Odziv    (bankrot): {odziv_1:.4f}")
        print(f"    Preciznost (bankrot): {precision_1:.4f}")
        print(f"    F1-skor  (bankrot): {f1_1:.4f}")
        print(classification_report(y_te, y_pred, target_names=['Stabilno', 'Bankrot']))

        cm = confusion_matrix(y_te, y_pred)
        print_confusion_matrix(cm, name, suffix)

        results[name] = {
            'accuracy':   acc,
            'odziv':      odziv_1,
            'precision':  precision_1,
            'f1':         f1_1,
            'y_prob':     y_prob,
            'model':      model
        }

        # Sačuvaj model
        fname = f"model_{name.replace(' ', '_')}_{suffix}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(model, f)

    return results


results_full = evaluate_models(models, X_train_res,  y_train_res, X_test,     y_test, "FULL")
results_red  = evaluate_models(models, X_train_red,  y_train_res, X_test_red, y_test, "REDUCED")

print("\n" + "=" * 60)
print("FAZA 4: Stratified K-Fold Cross-Validation (5 foldova)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    # Koristimo originalne (neskaliranje) podatke — SMOTE se ne primenjuje unutar CV
    # radi brzine; za produkcijsku upotrebu koristiti Pipeline
    scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='recall', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<25} Odziv CV: {scores.mean():.4f} ± {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
names  = list(cv_results.keys())
means  = [cv_results[n].mean() for n in names]
stds   = [cv_results[n].std()  for n in names]
x = np.arange(len(names))
bars = ax.bar(x, means, yerr=stds, capsize=5, color=COLOR_STABLE, edgecolor='white', error_kw={'ecolor': '#555'})
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right')
ax.set_ylabel('Odziv — klasa Bankrot', fontsize=12)
ax.set_title('5-Fold Crossvalidation — Odziv za klasu Bankrot\n(trening skup sa SMOTE)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], padding=6, fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0.8, color=COLOR_BANKRUPT, linestyle='--', linewidth=1.5, label='Prag 0.80')
ax.legend()
plt.tight_layout()
plt.savefig('crossvalidation_odziv.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSlika 'crossvalidation_odziv.png' je sačuvana.")

print("\n" + "=" * 60)
print("FAZA 5: ROC krive i AUC skorovi")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_roc = ['#4878CF', '#E84393', '#2ca02c', '#d62728', '#9467bd']

for ax, (res_dict, suffix, X_te, y_te) in zip(
    axes,
    [(results_full, "Svi atributi (95)", X_test, y_test),
     (results_red,  "Redukovani atributi (10)", X_test_red, y_test)]
):
    for (name, res), color in zip(res_dict.items(), colors_roc):
        if res['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_te, res['y_prob'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
            print(f"  {suffix} | {name:<25} AUC = {roc_auc:.4f}")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Stopa lažno pozitivnih (FPR)', fontsize=11)
    ax.set_ylabel('Stopa tačno pozitivnih (TPR)', fontsize=11)
    ax.set_title(f'ROC Krive — {suffix}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('roc_krive.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSlika 'roc_krive.png' je sačuvana.")

print("\n" + "=" * 60)
print("FAZA 6: Precision-Recall krive")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (res_dict, suffix, y_te) in zip(
    axes,
    [(results_full, "Svi atributi (95)",       y_test),
     (results_red,  "Redukovani atributi (10)", y_test)]
):
    for (name, res), color in zip(res_dict.items(), colors_roc):
        if res['y_prob'] is not None:
            prec, rec, _ = precision_recall_curve(y_te, res['y_prob'])
            ap = average_precision_score(y_te, res['y_prob'])
            ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AP = {ap:.3f})")

    baseline = y_test.mean()
    ax.axhline(baseline, color='gray', linestyle='--', lw=1, label=f'Baseline ({baseline:.2f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Odziv', fontsize=11)
    ax.set_ylabel('Preciznost (Precision)', fontsize=11)
    ax.set_title(f'Precision-Recall Krive — {suffix}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('precision_recall_krive.png', dpi=150, bbox_inches='tight')
plt.close()
print("Slika 'precision_recall_krive.png' je sačuvana.")

print("\n" + "=" * 60)
print("FAZA 7: Feature Importance — Random Forest (svi atributi)")
print("=" * 60)

rf_model = results_full["Random Forest"]['model']
importances   = rf_model.feature_importances_
feat_names    = X.columns
sorted_idx    = np.argsort(importances)[-20:]  # Top 20

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(feat_names[sorted_idx], importances[sorted_idx], color=COLOR_BANKRUPT, edgecolor='white')
ax.set_xlabel('Važnost atributa (Gini impurity)', fontsize=12)
ax.set_title('Top 20 atributa po važnosti\nRandom Forest — Svi atributi', fontsize=13, fontweight='bold')
ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Slika 'feature_importance_rf.png' je sačuvana.")

# Top 5 ispisi
top5_idx = np.argsort(importances)[::-1][:5]
print("\nTop 5 najvažnijih atributa (Random Forest):")
for i, idx in enumerate(top5_idx):
    print(f"  {i+1}. {feat_names[idx]:<45} = {importances[idx]:.4f}")

plt.figure(figsize=(13, 6))
labels    = list(models.keys())
x         = np.arange(len(labels))
width     = 0.35

val_full = [results_full[m]['accuracy'] for m in labels]
val_red  = [results_red[m]['accuracy']  for m in labels]

bars1 = plt.bar(x - width/2, val_full, width, label='Svi atributi (95)',       color=COLOR_FULL,    edgecolor='white')
bars2 = plt.bar(x + width/2, val_red,  width, label='Redukovani atributi (10)', color=COLOR_REDUCED, edgecolor='white')

plt.bar_label(bars1, fmt='%.4f', padding=3, fontsize=8.5)
plt.bar_label(bars2, fmt='%.4f', padding=3, fontsize=8.5)
plt.ylabel('Tačnost (Accuracy)', fontsize=12)
plt.title('Poređenje tačnosti modela: Svi atributi vs Redukovani', fontsize=13, fontweight='bold')
plt.xticks(x, labels, rotation=10, ha='right')
plt.ylim(0.80, 1.00)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('poredjenje_modela.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSlika 'poredjenje_modela.png' je sačuvana.")

plt.figure(figsize=(13, 6))
val_full_r = [results_full[m]['odziv'] for m in labels]
val_red_r  = [results_red[m]['odziv']  for m in labels]

bars1 = plt.bar(x - width/2, val_full_r, width, label='Svi atributi (95)',       color=COLOR_FULL,    edgecolor='white')
bars2 = plt.bar(x + width/2, val_red_r,  width, label='Redukovani atributi (10)', color=COLOR_REDUCED, edgecolor='white')

plt.bar_label(bars1, fmt='%.2f', padding=3, fontsize=9)
plt.bar_label(bars2, fmt='%.2f', padding=3, fontsize=9)
plt.ylabel('Odziv — klasa Bankrot', fontsize=12)
plt.title('Poređenje Odziva za klasu Bankrot: Svi atributi vs Redukovani', fontsize=13, fontweight='bold')
plt.xticks(x, labels, rotation=10, ha='right')
plt.ylim(0, 1.1)
plt.axhline(0.8, color='gray', linestyle='--', linewidth=1.2, label='Preporučeni prag (0.80)')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('poredjenje_odziv.png', dpi=150, bbox_inches='tight')
plt.close()
print("Slika 'poredjenje_odziv.png' je sačuvana.")

print("\n" + "=" * 60)
print("ANALIZA ZAVRŠENA — Generisane slike:")
print("=" * 60)
files = [
    ('raspodela_klasa.png',         'Pie chart raspodele klasa'),
    ('vizuelizacija_2d.png',        '2D PCA vizuelizacija'),
    ('top10_atributi.png',          'F-skorovi top 10 atributa'),
    ('korelaciona_matrica.png',     'Heatmap korelacije atributa'),
    ('poredjenje_modela.png',       'Bar grafik tačnosti svih modela'),
    ('poredjenje_odziv.png',        'Bar grafik odziva svih modela'),
    ('roc_krive.png',               'ROC krive sa AUC skorovima'),
    ('precision_recall_krive.png',  'Odziv-Preciznost krive'),
    ('feature_importance_rf.png',   'Feature importance — Random Forest'),
    ('crossvalidation_odziv.png',   '5-Fold CV odziv grafik'),
]
for fname, desc in files:
    print(f"  {fname:<35} {desc}")

print("\nSvi .pkl modeli su sačuvani u radnom direktorijumu.")
