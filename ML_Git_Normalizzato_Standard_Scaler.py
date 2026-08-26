# Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import display


# CONFIGURAZIONE
CAT_COLS = ['R', 'W', 'V', 'Z', 'AA']
STAT_COLS = ['SOGGETTO','STAT1','STAT2']
CSV_PATH = 'https://raw.githubusercontent.com/Giangysam/Correlations/main/DATI_RIS_coperti.xlsx'
TARGET_COLUMN = 'TARGET'

# CARICAMENTO DEI DATI
df = pd.read_excel(CSV_PATH)
print("Dataset caricato con successo.")
print (df.head())
print(df.info())
print()
num_righe, num_colonne = df.shape
print(f"Numero di righe: {num_righe}")
print(f"Numero di colonne: {num_colonne}")
print()

# Utilizzo di Random Forest, il quale non necessita di normalizzazione, lavorando con gli alberi
# Utilizzo dello stratified k-fold, per migliorare il modello
# Parte finale con la ricerca delle features più importanti che incidono maggiormente sui difetti

# Creazione di un nuovo df (X) contente le features, dal target (y)
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Split data in training e test set
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Training set: {X_train.shape[0]} features')
print(f'Test set {X_test.shape[0]} features')

# Random Forest (non ha bisogno di dati normalizzati)
model = RandomForestClassifier (n_estimators=500, random_state=42, class_weight='balanced')

# Addestramento del modello
model.fit(X_train, y_train)
print('Random Forest addestrato con successo')

# Cross-validation strategy
skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)

scoring = {'recall_defects': 'recall', 'precision_defects': 'precision', 'f1_defects': 'f1'}

cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)

print('Cross-Validation Results (media e dev standard):')
for metric_name, scores in cv_results.items():
    if metric_name.startswith('test_'):
        print(f'{metric_name.replace('test_', '')}: {scores.mean():.2f} (+/- {scores.std():.2f})')


# Predizione del modello
y_pred = model.predict(X_test)

print('\nClassification Report')
print(classification_report(y_test, y_pred))
print()

# Calcolo della Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print()
print(cm)
print()

# Calcolo della recall e f1-score per la classe 1 (difetti)
recall_positive = recall_score (y_test, y_pred, pos_label=1)
print(f'\nRecall per difetti (Classe 1): {recall_positive:.2f}')
print()
f1_positive = f1_score(y_test, y_pred, pos_label=1)
print(f'F1-score per Difetti (classe 1): {f1_positive:.2f}')

if recall_positive >=0.8:
    print('Great!')
else:
    print('Cerca altri modelli o migliora il tuning')
    
# Analisi delle features più importanti per la predizione dei difetti
# Random Forest fornisce anche punteggi di importanza delle features, indicando quali variabili hanno 
# maggiore influenza nelle decisioni del modello
feature_importances = model.feature_importances_

# Creazione dataframe delle variabili più importanti con i rispettivi nomi
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print('\nTop 30 più Importanti per Predizione Difetto:')
display(feature_importance_df.head(30))
print()

# Grafico delle prime n features più importanti
plt.figure(figsize=(16, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(30))
plt.title('Top 30 Feature più importanti per Predizione Difetti')
plt.xlabel('Importanza')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
