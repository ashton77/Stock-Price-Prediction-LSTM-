from operator import inv
import re
import json
from io import StringIO
from threading import active_count
from bs4 import BeautifulSoup
from keras.layers.core import Activation, RepeatVector
from pandas.core.indexes import period
import requests
import csv
from math import ceil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen, CointRankResults, select_coint_rank
from bokeh.plotting import figure, show
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense, Dropout, TimeDistributed, Flatten, RepeatVector
from LSTM_encoder_decoder.code import lstm_encoder_decoder

scaler = StandardScaler()



def get_data(company):
    stock_url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?'
    params = {
    'range': '10y',
    'interval': '1d',
    'events': 'history'
    #&includeAdjustedClose=true
    }
    response = requests.get(stock_url.format(company.upper()), params=params)
    file = StringIO(response.text)
    reader = csv.reader(file)
    data = tuple(reader)
    data_df = pd.DataFrame(data=data[1:], columns=data[:1][0])
    data_df['Date'] = data_df['Date'].astype('datetime64[ns]')
    data_df['Open'] = data_df['Open'].astype('float')
    data_df['High'] = data_df['High'].astype('float')
    data_df['Low'] = data_df['Low'].astype('float')
    data_df['Close'] = data_df['Close'].astype('float')
    data_df['Adj Close'] = data_df['Adj Close'].astype('float')
    data_df['Volume'] = data_df['Volume'].astype('float')
    data_df.set_index('Date', inplace=True, drop=True)
    return data_df

# def set_index(data):
#     data =  data.set_index('Date', inplace=True, drop=True)
#     return data

def data_scaling(data):
    return scaler.transform(data)

def inv_scaling(data, old_data):
    pred_copies = np.repeat(data, old_data.shape[1], axis=-1)
    pred_y_inv_scaled = scaler.inverse_transform(pred_copies)[:,0]
    return pred_y_inv_scaled

def sliding_window_prep(data, window_size):
    features = []
    target = []
    n_future = 1
    for i in range(window_size, len(data) - n_future + 1):
        features.append(data[i - window_size:i, 0:data.shape[1]])
        target.append(data[i + n_future - 1:i + n_future, 3])

    return np.array(features), np.array(target)

def pred_to_list(prediction):
    temp_list = [i[3] for i in prediction]
    return temp_list

def LSTM_model(train_X, train_y):
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(train_y.shape[1]))
    return model

def LSTM_E_D_model(train_X, train_y):
    model = Sequential()
    model.add(LSTM(150, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(RepeatVector(train_y.shape[1]))
    model.add(LSTM(150, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(TimeDistributed(Dense(train_y.shape[1])))
    return model

def main():
    data_df = get_data('aapl')
    data_df_new = data_df

    # sns.lineplot(data=data_df, x='Date', y='Close')
    # plt.show()
    scaler.fit(data_df)

    data_df_scaled = data_scaling(data_df)
    
    features_df, target_df = sliding_window_prep(data_df_scaled, 14)

    train_X, test_X, train_y, test_y = train_test_split(features_df, target_df, test_size = 0.2, shuffle=False, stratify=None)

    model = LSTM_E_D_model(train_X, train_y)
    # model = load_model('P:/BDBA/SEM_4/Analytics_4/LSTM_E_D_model_1.h5')
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    # LSTM model using pytorch

    # model = lstm_encoder_decoder.lstm_seq2seq(input_size=train_X.shape[2], hidden_size=128)

    # loss = model.train_model(train_X, train_y, n_epochs = 50, target_len = train_y.shape[0], batch_size = 5, 
    #                      training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, 
    #                      learning_rate = 0.01, dynamic_tf = False)

    # pred_y = model.predict(test_X, test_y.shape[0])

    history = model.fit(train_X, train_y
                    , epochs = 10
                    , batch_size = 32
                    , validation_split = 0.1
                    , verbose = 1)

    # plt.plot(history.history['loss'], label='Training Loss')

    # plt.plot(history.history['val_loss'], label ='Validation Loss')
    # plt.legend()
    # plt.show()
    model.save('P:/BDBA/SEM_4/Analytics_4/LSTM_E_D_model_2.h5')

    pred_y = model.predict(test_X)

    pred_y_inv_scaled = inv_scaling(pred_y, data_df)

    prediction_list = pred_to_list(pred_y_inv_scaled)

    test_y_inv_scaled = inv_scaling(test_y, data_df)

    plot_df = pd.DataFrame()
    plot_df['pred_y'] = prediction_list
    plot_df['test_y'] = test_y_inv_scaled

    sns.lineplot(data=plot_df, x=range(0,test_y.shape[0]), y='pred_y')
    sns.lineplot(data=plot_df, x=range(0,test_y.shape[0]), y='test_y')
    plt.legend(labels=['Predicted','Actual'])
    plt.show()

    data_df_date = list(data_df.index.values)

    future_length = 90
    forecast_dates_period = pd.date_range(data_df_date[-1], periods=future_length, freq='1d').tolist()

    forecast = model.predict(test_X[-future_length:])

    forecast_inv_scaled = inv_scaling(forecast, data_df)

    forecast_final = [i[3] for i in forecast_inv_scaled]

    forecast_df = pd.DataFrame({'Date':np.array(forecast_dates_period), 'Close': forecast_final})

    main_data = pd.DataFrame(data=data_df_new['Close'])
    main_data['Date'] = data_df_date
    main_data['Date'] = pd.to_datetime(main_data['Date'])
    main_data = main_data.loc[main_data['Date'] >= '2019-01-01']

    sns.lineplot(data=main_data, x='Date', y='Close')
    sns.lineplot(data=forecast_df, x='Date', y='Close')
    plt.show()

if __name__=='__main__':
    main()