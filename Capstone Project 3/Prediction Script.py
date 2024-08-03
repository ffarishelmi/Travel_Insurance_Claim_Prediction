# Impor library dasar
import pandas as pd
import numpy as np
import pickle

# Load data
df_insurance = pd.read_csv('clean_data_travel_insurance.csv')

# Load model
filename = 'model_production.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Praproses dan prediksi
y_pred = loaded_model.predict(df_insurance)
df_insurance['klaim_asuransi'] = np.where(y_pred==1, 'Yes', 'No')

df_insurance.to_csv('hasil_prediksi.csv')
