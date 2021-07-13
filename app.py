from io import StringIO, BytesIO
from numpy.lib.npyio import load
from pandas.core.indexes import period
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
# from keras.layers import LSTM
# from keras.layers import Dense, Dropout, TimeDistributed, Flatten, RepeatVector
# from keras.layers.core import Activation, RepeatVector
from flask import Flask, request, jsonify, render_template, Markup, send_file
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

fig, ax = plt.subplots(figsize=(6,6))
ax= sns.set_style(style="darkgrid")

app = Flask(__name__)
scaler = StandardScaler()

# def get_data(company):
#     stock_url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?'
#     params = {
#     'range': '10y',
#     'interval': '1d',
#     'events': 'history'
#     #&includeAdjustedClose=true
#     }
#     response = requests.get(stock_url.format(company.upper()), params=params)
#     file = StringIO(response.text)
#     reader = csv.reader(file)
#     data = tuple(reader)
#     data_df = pd.DataFrame(data=data[1:], columns=data[:1][0])
#     data_df['Date'] = data_df['Date'].astype('datetime64[ns]')
#     data_df['Open'] = data_df['Open'].astype('float')
#     data_df['High'] = data_df['High'].astype('float')
#     data_df['Low'] = data_df['Low'].astype('float')
#     data_df['Close'] = data_df['Close'].astype('float')
#     data_df['Adj Close'] = data_df['Adj Close'].astype('float')
#     data_df['Volume'] = data_df['Volume'].astype('float')
#     data_df.set_index('Date', inplace=True, drop=True)
#     return data_df

def get_data():
    # data = pd.read_csv('P:/BDBA/SEM_4/Analytics_4/Stock_prediction/AAPL.csv')
    data = pd.read_csv('https://github.com/ashton77/Stock-Price-Prediction-LSTM-/blob/69df1b6cf917eb46823371e08322a0bea669e444/AAPL.csv')
    data.set_index('Date', inplace=True, drop=True)
    data.index = pd.to_datetime(data.index)
    data['Open'] = data['Open'].astype('float')
    data['High'] = data['High'].astype('float')
    data['Low'] = data['Low'].astype('float')
    data['Close'] = data['Close'].astype('float')
    data['Adj Close'] = data['Adj Close'].astype('float')
    data['Volume'] = data['Volume'].astype('float')
    return data


def sliding_window_prep(data, window_size):
    features = []
    target = []
    n_future = 1
    for i in range(window_size, len(data) - n_future + 1):
        features.append(data[i - window_size:i, 0:data.shape[1]])
        target.append(data[i + n_future - 1:i + n_future, 3])
    return np.array(features), np.array(target)

def inv_scaling(data, old_data):
    pred_copies = np.repeat(data, old_data.shape[1], axis=-1)
    pred_y_inv_scaled = scaler.inverse_transform(pred_copies)[:,0]
    return pred_y_inv_scaled

# labels = [
# 'JAN', 'FEB', 'MAR', 'APR',
# 'MAY', 'JUN', 'JUL', 'AUG',
# 'SEP', 'OCT', 'NOV', 'DEC'
# ]

# values = [
#     967.67, 1190.89, 1079.75, 1349.19,
#     2328.91, 2504.28, 2873.83, 4764.87,
#     4349.29, 6458.30, 9907, 16297
# ]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pred')
def pred():
    return render_template('line_chart.html')

@app.route('/predict')
def predict():
    data_df = get_data()
    # print(data_df.dtypes)
    # data_df_new = data_df
    scaler.fit(data_df)
    data_df_scaled = scaler.transform(data_df)
    features, target = sliding_window_prep(data_df_scaled, 14)

    train_X, test_X, train_y, test_y = train_test_split(features, target, test_size = 0.2, shuffle=False, stratify=None)
    
    model = load_model('P:/BDBA/SEM_4/Analytics_4/Stock_prediction/LSTM_E_D_model_1.h5')

    data_df_date = list(data_df.index.values)

    future_length = 90
    forecast_dates_period = pd.date_range(data_df_date[-1], periods=future_length, freq='1d').tolist()

    forecast = model.predict(test_X[-future_length:])

    forecast_inv_scaled = inv_scaling(forecast, data_df)

    forecast_final = [i[3] for i in forecast_inv_scaled]

    forecast_df = pd.DataFrame({'Date':np.array(forecast_dates_period), 'Close': forecast_final})

    main_data = pd.DataFrame()
    # main_data['Date'] = data_df_date
    # main_data['Date'] = pd.to_datetime(main_data['Date'])
    # main_data.reset_index(drop=True)
    # print(main_data)
    main_data['Close'] = data_df['Close']
    main_data['Date'] = data_df_date
    # print(main_data)
    main_data = main_data.loc[main_data['Date'] >= '2019-01-01']

    # print(main_data)

    df_to_plot = main_data.append(forecast_df)
    # print(df_to_plot)

    labels = list(df_to_plot['Date'])
    values = list(df_to_plot['Close'])

    # return render_template('index.html', title='Stock Price Prediction in USD', max=500, labels=labels, values=values)

    # line_labels=labels
    # line_values=values
    # return jsonify({'labels' : labels, 'values': values})

    sns.lineplot(labels, values)
    pred_point = main_data.Date[-1]
    plt.axvline(x=pred_point, color='blue', linestyle='--')
    canvas=FigureCanvas(fig)
    img=BytesIO()
    fig.savefig(img)
    img.seek(0)
    # return render_template('line_chart.html')
    return send_file(img, mimetype='img/png')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)