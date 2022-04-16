# // CODE FOR DEPLOYING THE APP ON STREMLIT // 
# // THE LSTM MODEL CODE IS  ON ANOTHER REPO //
# // UNSUPERVISED LEARNING-RNN //
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import datetime as dt
from nsepy import get_history
import io

#Initializing the start and end date for the data
start_date =  dt.datetime(2000, 1,1)
end_date =  dt.datetime(2021, 4, 30)

st.title('Stock Trend Prediction')
st.text('with LSTM Model(multi-variate)')


st.caption('Enter Indian stocks that exists from 2000 - 2021')

# by default the ticker has the value of "AXISBANK"
user_input = st.text_input('Enter Stock Ticker', 'AXISBANK')
raw_data = get_history(symbol = user_input, start = start_date, end = end_date)
df = raw_data.copy()

# drop the variables that are un-neccesary IMO
to_drop = [ 'Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover',  'Trades', 'Deliverable Volume', '%Deliverble']
df = df.drop(to_drop, axis = 1)

# DESCRIBE THE DATE
st.subheader("Data from 2000 - '21")
st.write(df.describe())

# // optional //

#EXTRACT THE INFO
#buffer = io.StringIO()
#df.info(buf=buffer)
#s = buffer.getvalue()
#st.text(s)


#VISUALIZE THE OPENING PRICE
st.subheader('Open Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Open, label = 'Open Price')
plt.legend()
st.pyplot(fig)

#PLOTTING MOVING AVERAGE of 100
st.subheader('Open Price vs Time Chart with 100 MA')
ma100 = df.Open.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, label = 'MA100')
plt.plot(df.Open, label = 'Open Price')
plt.legend()
st.pyplot(fig)

#PLOTTING MOVING AVERAGE OF 200
st.subheader('Open Price vs Time Chart with 100MA and 200MA')
ma100 = df.Open.rolling(100).mean()
ma200 = df.Open.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, label = 'MA100')
plt.plot(ma200, label = 'MA200')
plt.plot(df.Open, label = 'Open Pice')
plt.legend()
st.pyplot(fig)

#ASSIGNING THE DATA
df_for_model = df.copy()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(df_for_model)

scaled_df_for_training = scaler.transform(df_for_model)

x_train = []
y_train = []

n_future = 1  # no. of days to predict
n_past = 14   # no of days we want the prediction to be based on
              # based on above lines, we'll train the model for every 14 days and predict for the 15th day.

for i in range(n_past, len(scaled_df_for_training) - n_future + 1):
    x_train.append(scaled_df_for_training[i - n_past: i, 0: df_for_model.shape[0]])
    y_train.append(scaled_df_for_training[i + n_future - 1: i + n_future, 0])

#CONVERT X_TRAIN AND Y_TRAIN INTO ARRAYS(they're in list form)
x_train, y_train = np.array(x_train), np.array(y_train)

#LOAD THE LSTM MODEL WITH .H5 FORMAT, available in the same repository
model = load_model('RNN_MODEL.h5')

#extract dates from raw_data
#train_dates = pd.to_datetime(raw_data['Date'])
train_dates = pd.date_range(start = start_date, end = end_date)

#prediction for the future 180 days
n_past = 366
n_prediction_days = 365 #predicting the past 365 days for training

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods = n_prediction_days, freq = '1d').tolist()

#PREDICT
predict = model.predict(x_train[-n_prediction_days:])

predict_copies = np.repeat(predict, df_for_model.shape[1], axis = 1)
y_pred_future = scaler.inverse_transform(predict_copies)[:,0]

#Convert time stamps into Dates
predict_dates = []

for i in predict_period_dates:
    predict_dates.append(i.date())

df_predict = pd.DataFrame({'Date': np.array(predict_dates), 'Open': y_pred_future})
df_predict['Date'] = pd.to_datetime(df_predict['Date'])

original = raw_data[['Open']]
# // code snippet from LSTM model that won't work here
#original['Date'] = pd.date_range(start = start_date, end = end_date)
# for visiualization we'll look at graph from a early stage
#original = original.loc[original['Date'] >= '2018-5-1']

#FINAL VISUALIZATION
st.subheader('Predicted vs Actual data')
fig2 = plt.figure(figsize=(12,6))
#plt.plot(original['Date'], original['Open'], color = 'r', label = 'Original Price')
plt.plot(original['Open'], color = 'b', label = 'Original Price')
plt.plot(df_predict['Date'], df_predict['Open'], color = 'k', label = 'Predicted Price')
#plt.plot(df_predict['Open'], color = 'g', label = 'Predicted Price')
plt.title(user_input + ' Stock Trend Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig2)

st.caption('Made by Prakhar Saxena')
