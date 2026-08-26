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
