# Preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIGURAZIONE
CAT_COLS = ['R', 'W', 'V', 'Z', 'AA']
STAT_COLS = ['SOGGETTO','STAT1','STAT2']
CSV_PATH = 'https://raw.githubusercontent.com/Giangysam/Correlations/main/DATI_RIS_coperti.xlsx'
TARGET_COLUMN = 'TARGET'

# CARICAMENTO DEI DATI
df = pd.read_excel(CSV_PATH)
print("Dataset caricato con successo.")
display(df.head())
print(f"Dimensioni del dataset: {df.shape}")
