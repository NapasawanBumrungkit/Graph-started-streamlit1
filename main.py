import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

page = st.sidebar.selectbox("เลือกหน้า", 
    ["หน้าแนะนำ (Machine Learning)", "หน้าแนะนำ (Neural Network)", 
     "Demo: Machine Learning", "Demo: Neural Network"])

if page == "หน้าแนะนำ (Machine Learning)":
    st.markdown(
    "<p style='text-align: right;'>นางสาวนภัสวรรณ บำรุงกิจ 6604062620140</p>",
    unsafe_allow_html=True
)
    st.title("การพัฒนาโมเดล Machine Learning")
    st.markdown(" 🔹ขั้นตอนพัฒนาโมเดล ML ")
    st.markdown("""🔹1. การเตรียมข้อมูล ที่มาของข้อมูล TSLA_2010-06-29_2025-02-13 (1).csv 
                คือ https://www.kaggle.com/datasets/umerhaddii/tesla-stock-data-2025 """)                

    
    code = """df = pd.read_csv("TSLA_2010-06-29_2025-02-13 (1).csv")

print(df.head())
print(df.info())

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True) """
    st.code(code, language="python")

    st.markdown(" - มีการโหลด dataset TSLA_2010-06-29_2025-02-13 (1).csv ")
    st.markdown(" - มีการตรวจสอบและดู DataFrame")
    st.markdown(" - มีการแปลงวันจากdatasetให้ใช่งานได้")
    
    code = """df = df.dropna(subset=['Close'])  
X = df[['Open', 'High', 'Low', 'Volume']]  
y = df['Close'] """
    st.code(code, language="python")
    
    st.markdown(" - มีการลบแถวที่ไม่มีข้อมูล")
    st.markdown(" - มีการกำหนด Features และ Target")

    st.markdown("🔹2. ทฤษฎีของอัลกอริทึมที่พัฒนา")
    st.markdown(" - เนื่องจาก dataset เป็นข้อมูลหุ้นสิ่งมีระยะเวลาการเก็บข้อมูลที่ยาวนาน จึงเลือกเป็นโมเดล 3 อันนี้ คือ")
    st.markdown("1. LinearRegression เพราะเป็นโมเดลที่สามารถวิเคราะห์ข้อมูลที่มีแนวโน้มระยะยาว")
    st.markdown("2. RandomForestRegressor เพราะโมเดลนี้สามารถเทรนโมเดลที่ไม่ใช่เส้นตรงได้ สามารถรับค่าได้หลายตัวแปร และแก้ปัญหา  Outliers และ Noise ได้ดี ")
    st.markdown("3. XGBoost เพราะโมเดลที่ใช้งานกับ dataที่ซับซ้อนได้แก้ปัญหา  Outliers และ Noise ได้ดี")
  
    st.markdown("🔹3. ขั้นตอนการพัฒนาโมเดล")
    st.markdown("- การแบ่งข้อมูลเพื่อไป train โมเดล")
    code = """X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) """
    st.code(code, language="python")
    
    st.markdown("- การ train โมเดลLinearRegression")
    code = """lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test) """
    st.code(code, language="python")

    st.markdown("- การ train โมเดลRandomForestRegres")
    code = """rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test) """
    st.code(code, language="python")

    st.markdown("- การ train โมเดลXGBoost")
    code = """xg_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xg_model.fit(X_train, y_train)
y_pred_xg = xg_model.predict(X_test)"""
    st.code(code, language="python")

    st.markdown("- การคำนวณค่าความคลาดเคลื่อน ถ้ามีค่าที่น้อยยิ่งเป็นโมเดลที่มีความแม่นยำมากใช้ โดยใช้ "
    "Mean Squared Error (MSE) และ Root Mean Squared Error (RMSE) และแสดงผล")
    code = """def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse

mse_lr, rmse_lr = evaluate_model(y_test, y_pred_lr)
mse_rf, rmse_rf = evaluate_model(y_test, y_pred_rf)
mse_xg, rmse_xg = evaluate_model(y_test, y_pred_xg)

print(f"Linear Regression - MSE: {mse_lr}, RMSE: {rmse_lr}")
print(f"Random Forest - MSE: {mse_rf}, RMSE: {rmse_rf}")
print(f"XGBoost - MSE: {mse_xg}, RMSE: {rmse_xg}")"""
    st.code(code, language="python")

    st.markdown("- การสร้างกราฟ")
    code = """plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='True Prices', color='blue')
plt.plot(y_test.index, y_pred_lr, label='Linear Regression', color='red', linestyle='--')
plt.plot(y_test.index, y_pred_rf, label='Random Forest', color='green', linestyle='--')
plt.plot(y_test.index, y_pred_xg, label='XGBoost', color='orange', linestyle='--')
plt.legend()
plt.title('Stock Price Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()"""
    st.code(code, language="python")



elif page == "หน้าแนะนำ (Neural Network)":
    st.markdown(
    "<p style='text-align: right;'>นางสาวนภัสวรรณ บำรุงกิจ 6604062620140</p>",
    unsafe_allow_html=True)
    
    st.title("การพัฒนาโมเดล Neural Network")
    st.markdown("""
    ### 🔹 ขั้นตอนพัฒนา Neural Network """)

    st.markdown("""🔹1. การเตรียมข้อมูล ที่มาของข้อมูล Ali_Baba_Stock_Data.csv
                คือ https://www.kaggle.com/datasets/mhassansaboor/alibaba-stock-dataset-2025 """) 
    
    st.markdown("- มีการโหลด dataset Ali_Baba_Stock_Data.csv")
    code = """df = pd.read_csv('Ali_Baba_Stock_Data.csv') """
    st.code(code, language="python")
    
    st.markdown("- มีการแปลงวันจากdatasetให้ใช่งานได้")
    st.markdown("- มีการ sortข้อมูลของ date")
    code = """df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)"""
    st.code(code, language="python")

    st.markdown("- เป็นการเลือกข้อมูลที่จะใช้ train model")
    code = """data = df['Close'].values.reshape(-1, 1)"""
    st.code(code, language="python")

    st.markdown("- การกำหนดขนาดข้อมูลให้อยู่ในช่วง ค่า MinMax ของข้อมูลช่วยลดความผิดพลาดของการ train model")
    code = """scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)"""
    st.code(code, language="python")
    
    st.markdown("- มีการกำหนดข้อมุลย้อนหลังเพื่อนำมา train model")
    st.markdown("- มีการวน loop เพื่อคำนวณค่าที่เก็บย้อนหลัง 60 วัน และค่าที่จะ train ปัจจุบัน")
    code = """look_back = 60
def create_sequences(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0]) 
        y.append(data[i, 0])  
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, look_back)"""
    st.code(code, language="python")

    st.markdown("🔹2.ทฤษฎีของอัลกอริทึมที่พัฒนา")
    st.markdown("อัลกอริทึมที่เลือกใช้คือ LSTM ")
    st.markdown("- เหตุผลที่เลือก LSTM Model  เพราะเป็นโมเดลที่มีการทำงานที่ข้อมูลมีการเก็บมาหลายๆปี, โมเดลนี้มี่การแก้ Vanishing Gradient ของ RNN และ สามารถใช้กับ Time Series Data ได้")

    st.markdown("🔹3.ขั้นตอนการพัฒนาโมเดล")

    st.markdown("- การแบ่งข้อมูลเพื่อไป train โมเดล")
    code = """train_size = int(len(X) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y[:train_size], y[train_size:]"""
    st.code(code, language="python")

    st.markdown("""- การสร้างโมเดล 2 ชั้น
    - การใช้ dropout ลด Overfitting
    - ใช้ Adam Optimizer และวัดค่า Mean Squared Error (MSE)""")
    code = """lstm_model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mean_squared_error")"""
    st.code(code, language="python")

    st.markdown("- การแสดงผลเป็นกราฟ และ ค่า Root Mean Squared Error (RMSE)")
    code = """def evaluate_and_plot(model, X_test, y_test, title):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"{title} Test RMSE: {rmse}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

evaluate_and_plot(lstm_model, X_test_lstm, y_test, "Alibaba Stock Price Prediction using LSTM")"""
    st.code(code, language="python")



elif page == "Demo: Machine Learning":
    st.title("Demo: โมเดล Machine Learning")

    if st.button("Run Model"):
        try:
          
            df = pd.read_csv("TSLA_2010-06-29_2025-02-13 (1).csv")
            df.dropna(inplace=True)

           
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

         
            X = df[['Open', 'High', 'Low', 'Volume']]
            y = df['Close']

          
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

         
            from sklearn.linear_model import LinearRegression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X_test)

           
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)

            import xgboost as xgb
            xg_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            xg_model.fit(X_train, y_train)
            y_pred_xg = xg_model.predict(X_test)

    
            def evaluate_model(y_true, y_pred):
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                return mse, rmse

            mse_lr, rmse_lr = evaluate_model(y_test, y_pred_lr)
            mse_rf, rmse_rf = evaluate_model(y_test, y_pred_rf)
            mse_xg, rmse_xg = evaluate_model(y_test, y_pred_xg)

            st.write(f"**Linear Regression** - MSE: {mse_lr}, RMSE: {rmse_lr}")
            st.write(f"**Random Forest** - MSE: {mse_rf}, RMSE: {rmse_rf}")
            st.write(f"**XGBoost** - MSE: {mse_xg}, RMSE: {rmse_xg}")

          
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_test.index, y_test, label='True Prices', color='blue')
            ax.plot(y_test.index, y_pred_lr, label='Linear Regression', color='red', linestyle='--')
            ax.plot(y_test.index, y_pred_rf, label='Random Forest', color='green', linestyle='--')
            ax.plot(y_test.index, y_pred_xg, label='XGBoost', color='orange', linestyle='--')
            ax.legend()
            ax.set_title('Stock Price Prediction Comparison')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            st.pyplot(fig)

        except FileNotFoundError:
            st.error("ไม่พบไฟล์ 'TSLA_2010-06-29_2025-02-13 (1).csv'")


elif page == "Demo: Neural Network":
    st.title("Demo: โมเดล Neural Network (การทำนายราคาหุ้น)")

    if st.button("Run Model"):
        try:
            
            import os
            if not os.path.exists("Ali_Baba_Stock_Data.csv"):
               st.error("ไม่พบไฟล์ 'Ali_Baba_Stock_Data.csv' กรุณาอัปโหลดไฟล์ก่อน")
            else:
               df = pd.read_csv("Ali_Baba_Stock_Data.csv")
         
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True) 

            data = df['Close'].values.reshape(-1, 1)

        
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            
            look_back = 60

            def create_sequences(data, look_back=60):
                X, y = [], []
                for i in range(look_back, len(data)):
                    X.append(data[i-look_back:i, 0])  
                    y.append(data[i, 0])  
                return np.array(X), np.array(y)

            X, y = create_sequences(scaled_data, look_back)

           
            X_lstm = np.reshape(X, (X.shape[0], X.shape[1], 1))

            train_size = int(len(X) * 0.8)
            X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            lstm_model = Sequential()
            lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(LSTM(units=50, return_sequences=False))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(Dense(units=25))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')

            
            lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

            
            def evaluate_and_plot(model, X_test, y_test, title):
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
                st.write(f"{title} Test RMSE: {rmse}")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(y_test_actual, label='Actual Price')
                ax.plot(predictions, label='Predicted Price')
                ax.set_title(title)
                ax.set_xlabel("Time (Days)")
                ax.set_ylabel("Close Price (USD)")
                ax.legend()
                st.pyplot(fig)

           
            evaluate_and_plot(lstm_model, X_test_lstm, y_test, "Alibaba Stock Price Prediction using LSTM")

        except FileNotFoundError:
            st.error("ไม่พบไฟล์ 'Ali_Baba_Stock_Data.csv'")


