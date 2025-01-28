import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 初始化MT5连接
def init_mt5():
    if not mt5.initialize():
        print("初始化失败")
        mt5.shutdown()
        return False
    return True

# 获取历史数据
def get_historical_data(symbol, timeframe, n_points):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_points)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['close']]

# 数据预处理
def preprocess_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# 构建LSTM模型
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 主程序
def main():
    # 参数设置
    symbol = "XAGUSD"
    timeframe = mt5.TIMEFRAME_H1
    n_points = 5000
    look_back = 60
    epochs = 50
    batch_size = 32
    
    # 初始化MT5
    if not init_mt5():
        return
    
    try:
        # 获取数据
        data = get_historical_data(symbol, timeframe, n_points)
        
        # 数据预处理
        X, y, scaler = preprocess_data(data.values, look_back)
        
        # 构建模型
        model = build_model((X.shape[1], 1))
        
        # 训练模型
        model.fit(X, y, batch_size=batch_size, epochs=epochs)
        
        # 预测
        test_data = data[-look_back:].values
        test_data_scaled = scaler.transform(test_data)
        X_test = np.array([test_data_scaled[:, 0]])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)
        
        print(f"预测价格: {predicted_price[0][0]}")
        
        # 可视化
        plt.figure(figsize=(16,8))
        plt.plot(data.index, data['close'], label='实际价格')
        plt.axvline(x=data.index[-1], color='r', linestyle='--', label='预测点')
        plt.title(f'{symbol} 价格预测')
        plt.xlabel('时间')
        plt.ylabel('价格')
        plt.legend()
        plt.show()
        
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
