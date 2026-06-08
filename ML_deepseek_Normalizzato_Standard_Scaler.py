#caricamento dei dati

import pandas as pd
df = pd.read_csv('C:\\Users\\administrator\\.vscode\\DATI_RIS.csv', delimiter=';', decimal=',')

print(df.head())
print(df.shape)

num_righe, num_colonne = df.shape
print(f"Numero di righe: {num_righe}")
print(f"Numero di colonne: {num_colonne}")
#print(df.columns)

#se serve richiamare una colonna ricorda che ti serve una variabile col nome della colonna

colonna= df['ROTOLO']
print(colonna)

#verifica dei duplicati
#in questo caso richiamo la colonna ROTOLO, per verificare che non ci siano duplicati
duplicati=df['ROTOLO'].duplicated()
print("Duplicati nella colonna 'ROTOLO':", duplicati)

#rimozione delle colonne SCARTO_TE, SCARTO_TO, UNGHIA_TE, UNGHIA_TO
df=df.drop(['SCARTO_TE', 'SCARTO_CO', 'UNGHIA_TE', 'UNGHIA_CO', 'WIDTH', 'OUTER_DIAMETER', 'BASEGR'], axis=1)
print(df.head)
#print(df.info)


# Normalizzazione dei dati con Standard Scaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Selezione delle colonne da normalizzare, escludendo quelle con id rotolo, spessore, ciclo_ris
# e dell'unghiatura
df.columns = df.columns.str.strip()

columns_not_normalize = ['ROTOLO', 'SPESSORE_NERO', 'CICLO_RIS', 'UNGHIA', 'POSITION']
remaining_columns = [col for col in df.columns if col not in columns_not_normalize]

#Conversione della virgola da , a .
#for col in remaining_columns:
 #   try:
  #      df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
   #     df[col] = pd.to_numeric(df[col], errors='coerce')
    #except Exception as e:
     #   print(f"Errore colonna {col}: {e}")

#print(type(df[remaining_columns].isnull()))
#print(type(df[remaining_columns].isnull().sum()))
#print(df[remaining_columns].dtypes)

#Controllo dei dtype dopo la conversione
#print("Colonne ancora object dopo conversione:")
#print(df[remaining_columns].dtypes[df[remaining_columns].dtypes == 'object'])

#cols_to_drop = df [remaining_columns].dtypes[df[remaining_columns].dtypes == 'object'].index.tolist()
#if cols_to_drop:
 #   print(f"Colonne rimosse perchè non numeriche: {cols_to_drop}")
  #  df=df.drop(columns=cols_to_drop)
   # remaining_columns=[col for col in remaining_columns if col not in cols_to_drop]

#Controllo di valori Nan
#imputed_values = df[remaining_columns].mean()
#df[remaining_columns] = df[remaining_columns].fillna(imputed_values)

#Verifica di valori NaN
#nan_check = df[remaining_columns].isnull().sum()
#nan_check = nan_check[nan_check > 0]
#if len(nan_check) > 0:
 #   print("Nessun NaN rimasto")

#Df prima della rimozione
#cols_prima=set(df.columns)

# Rimozione colonne costanti (tutti gli stessi valori)
#df = df.loc[:, (df != df.iloc[0]).any(axis=0)]

#Colonne rimosse
#cols_dopo=set(df.columns)
#cols_rimosse= cols_prima - cols_dopo
#print(f"Colonne rimosse: ({len(cols_rimosse)}):")
#for colo in cols_rimosse:
 #   print(f" - {col}")

#Colonne costanti
#for col in df.columns:
 #   if(df[col]==df[col].iloc[0]).all():
  #      print(f" - {col}: valore unico= {df[col].iloc[0]}, dtype={df[col].dtype}")

# Aggiornamento del df
#remaining_columns = [col for col in remaining_columns if col in df.columns]

#print(df.head())
#print(df.shape)

# Ai valori NaN si assegna la media dei dati
#imputed_values = df[remaining_columns].mean()
#df[remaining_columns] = df[remaining_columns].fillna(imputed_values)

#LE RIGHE DALLA 50 ALLA 100 SOLO PER CAPIRE CHE NEL DATASET ORIGINALE ANZICHE' ESSERCI IL PUNTO COME SEPARATORE DECIMALE, C'ERA LA VIRGOLA. E SONO DIVENTATO SCEMO!!!
#LE RIGHE DALLA 42 ALLA 48 INVECE POSSONO ESSERE SOSTITUITE SEMPLICEMENTE AGGIUNGENDO AL CARICAMENTO DEL FILE LA DECITURA "DECIMAL=','"!!!
#Normalizzazione delle colonne
scaler = StandardScaler()
df[remaining_columns] = scaler.fit_transform(df[remaining_columns])
print("Normalizzazione del dataframe")
print(df.head(10))


# Creazione di un nuovo DataFrame con le feature normalizzate e il target originale
   