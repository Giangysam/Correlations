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

colonna= df['col0']
print(colonna)

#verifica dei duplicati
#in questo caso richiamo la colonna col0, per verificare che non ci siano duplicati
duplicati=df['col0'].duplicated()
print("Duplicati nella colonna 'col0':", duplicati)

#rimozione delle colonne col1, col2, col3, col4, col5, col6, col7
df=df.drop(['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7'], axis=1)
print(df.head)
#print(df.info)


# Normalizzazione dei dati con Standard Scaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Selezione delle colonne da normalizzare, escludendo quelle con identificativo col, spessore, ciclo,
# difetto e posizione in forno
df.columns = df.columns.str.strip()

columns_not_normalize = ['col0', 'SPESSORE', 'CICLO', 'DIFETTO', 'POSITION']
remaining_columns = [col for col in df.columns if col not in columns_not_normalize]


#Normalizzazione delle colonne
scaler = StandardScaler()
df[remaining_columns] = scaler.fit_transform(df[remaining_columns])
print("Normalizzazione del dataframe")
print(df.head(10))



   