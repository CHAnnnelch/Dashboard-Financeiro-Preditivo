import streamlit as st 
import datetime
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
from statsmodels.tools.eval_measures import rmse 
from prophet import Prophet 

st.title("Dashboard Financeiro Preditivo")
ticker = st.sidebar.text_input("Ticker")
start_date = st.sidebar.date_input("Data Inicial")
end_date = st.sidebar.date_input("Data Final")

data = yf.download(ticker,start=start_date,end=end_date)
data.columns = data.columns.droplevel("Ticker")
data.to_csv('pred_preco.csv')
data = pd.read_csv('pred_preco.csv')
data['Date'] = pd.to_datetime(data['Date'])

data["Date"] = [
    datetime.datetime.strptime(
        str(target_date).split(" ")[0], '%Y-%m-%d').date()
        for target_date in data["Date"]
]

fig1 = px.line(data, x = data['Date'], y = data['Adj Close'], title=ticker)
st.plotly_chart(fig1)

tela_preco, tela_pred_preco, tela_pred_retorno = st.tabs(['Preços', "Predição com preços", "Predição com retorno"])

with tela_preco:
    st.header('Dados da ação')
    data2 = data
    data2['Retornos'] = (data['Adj Close'] /  data['Adj Close'].shift(1)) - 1
    data2.dropna(inplace = True)
    st.write(data2)
    st.header('Gráfico dos Retornos (em %)')
    fig2 = px.line(data2, x = data2['Date'], y = data2['Retornos'], title='Retornos')
    st.plotly_chart(fig2)

with tela_pred_preco:
    st.header('Predição')
    st.write(data)
    data3 = data[['Date', 'Close']]
    data3.columns = ['ds', 'y']
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(data3)
    dias_pred = int(st.sidebar.text_input("Dias para prever - preço"))
    future1 = prophet.make_future_dataframe(periods = dias_pred)
    predict1 = prophet.predict(future1)
    from prophet.plot import plot_plotly
    fig3 = plot_plotly(prophet, predict1)
    st.plotly_chart(fig3)
    st.write(predict1)
    predictions1 = predict1.iloc[-10:]['yhat']
    test = data3.iloc[-10:]
    st.write("Root Mean Square Error entre os valores reais e os previstos pelo modelo: ", rmse(predictions1,test['y']))
    st.write("Valor médio do dataset de testes: ", test['y'].mean())

with tela_pred_retorno:
    st.header('Predição')
    st.write(data)
    data4 = data[['Date', 'Retornos']]
    data4.columns = ['ds', 'y']
    prophet1 = Prophet(daily_seasonality=True)
    prophet1.fit(data4)
    dias_predr = int(st.sidebar.text_input("Dias para prever - retorno"))
    future2 = prophet1.make_future_dataframe(periods = dias_predr)
    predict2 = prophet1.predict(future2)
    from prophet.plot import plot_plotly
    fig4 = plot_plotly(prophet1, predict2)
    st.plotly_chart(fig4)
    st.write(predict2)
    predictions2 = predict2.iloc[-10:]['yhat']
    test1 = data4.iloc[-10:]
    st.write("Root Mean Square Error entre os valores reais e os previstos pelo modelo: ", rmse(predictions2,test['y']))
    st.write("Valor médio do dataset de testes: ", test1['y'].mean())
