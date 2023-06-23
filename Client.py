import pandas as pd
import os
from pathlib import Path
import datetime
import sqlite3
import requests
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers
print('TensorFlow version: {}'.format(tf.__version__))
from scipy.stats import zscore
from datetime import datetime, timedelta

run = True
# Step 1: Send intial data to API
if run:
    cwd = Path(os.getcwd())
    data_path = cwd / 'data'
    files = data_path.glob('*.csv')
    payload = []
    for file in files:
        payload.append(('file', open(file, 'rb')))
    post_endpoint = "http://192.168.1.99:9020"
    response = requests.post(post_endpoint, files =payload)

# Step 2: Pull data from API into DF
conn = sqlite3.connect('stocks.db')
df = pd.read_sql_query("SELECT * FROM stocks", conn)


# Step 3: Transform, normalize, missing values handle, loads into sqlite
symbols_of_interest = df.Symbol.unique()[0:5]


for symbol in symbols_of_interest:
    #Load into mysql table database
    table_name = symbol
    subset_df = df[df.Symbol == symbol][['Date', 'Open']]
    subset_df['Open'] = zscore(subset_df['Open']) # normalizing by zscore
    subset_df.to_sql(table_name, conn, if_exists='replace', index=False)


# Set up structure for dataframe
col_indices = ["Day_" + str(i) for i in range(20,-1,-1)]
col_dict = {key: [] for key in col_indices}
df = pd.DataFrame.from_dict(col_dict)


# Step 4: Construct input data to ML model
#   1. construct a matrix with columns (Day-20, Day-19, Day-18, etc..)
#   2. Our label will be day '0'
# predict.

symbol_to_predict = 'AAPL'
for i in range(60):
    number_of_days_predict_with = 40
    date_to_predict = datetime(2021, 3, 1) + timedelta(i)
    min_day_predicting = date_to_predict-timedelta(number_of_days_predict_with)
    min_day_str = min_day_predicting.strftime('%Y-%m-%d')
    predict_day_str = date_to_predict.strftime('%Y-%m-%d')

    data = pd.read_sql_query(f"SELECT DISTINCT * FROM {symbol_to_predict} WHERE Date<'{predict_day_str}' AND "
                                 f"Date>'{min_day_str}'", conn)
    i_input_data = data.Open[-21:] # use the last 20 days stock price to predict the final price. The
    df.loc[i] = i_input_data.to_numpy()


# Step 5: Simple ML Model
model = keras.Sequential(
    [
        layers.Dense(30, activation='relu', name='layer1'),
    ]
)
model.compile(optimizer='Adam',
              loss = 'mse',
              metrics=['mse']
)

history = model.fit(x = df.iloc[:, 0:-1], y=df.iloc[:, -1],epochs=10, validation_split=0.1)
