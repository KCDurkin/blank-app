import streamlit as st


import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class StockPredictorYF:
    def __init__(self, window_size=30, epochs=50, batch_size=32):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.last_sequence = None
        self.base_models = {
            'LSTM': None,
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        self.meta_model = LinearRegression()

    def get_stock_data(self, ticker, start_date, end_date):
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df

    def calculate_rsi(self, data, periods=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_sma(self, data, window=20):
        return data['Close'].rolling(window=window).mean()

    def calculate_ema(self, data, window=20):
        return data['Close'].ewm(span=window, adjust=False).mean()

    def calculate_bollinger_bands(self, data, window=20):
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band

    def calculate_momentum(self, data, window=10):
        return data['Close'].diff(window)

    def calculate_atr(self, data, window=14):
        high = data['High']
        low = data['Low']
        close = data['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    def prepare_features(self, ticker, start_date, end_date):
        df = self.get_stock_data(ticker, start_date, end_date)
        df['RSI'] = self.calculate_rsi(df)
        df['SMA'] = self.calculate_sma(df)
        df['EMA'] = self.calculate_ema(df)
        upper_band, lower_band = self.calculate_bollinger_bands(df)
        df['Upper Band'] = upper_band
        df['Lower Band'] = lower_band
        df['Momentum'] = self.calculate_momentum(df)
        df['ATR'] = self.calculate_atr(df)

        feature_columns = ['Close', 'RSI', 'SMA', 'EMA', 'Upper Band',
                         'Lower Band', 'Momentum', 'ATR']
        df = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        return df

    def prepare_data(self, df):
        scaled_data = self.scaler.fit_transform(df)
        self.last_sequence = scaled_data[-self.window_size:]

        X, y = [], []
        for i in range(len(scaled_data) - self.window_size):
            X.append(scaled_data[i:(i + self.window_size)])
            y.append(scaled_data[i + self.window_size, 0])

        return np.array(X), np.array(y), df.index[self.window_size:]

    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_base_models(self, X_train, y_train, X_test):
        predictions = {}

        # Train LSTM
        self.base_models['LSTM'] = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        self.base_models['LSTM'].fit(X_train, y_train,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   verbose=0)
        lstm_predictions = self.base_models['LSTM'].predict(X_test).flatten()

        dummy_array = np.zeros((len(lstm_predictions), self.scaler.n_features_in_))
        dummy_array[:, 0] = lstm_predictions
        predictions['LSTM'] = self.scaler.inverse_transform(dummy_array)[:, 0]

        X_train_2d = X_train.reshape((X_train.shape[0], -1))
        X_test_2d = X_test.reshape((X_test.shape[0], -1))

        for name, model in self.base_models.items():
            if name != 'LSTM':
                model.fit(X_train_2d, y_train)
                pred = model.predict(X_test_2d)
                dummy_array = np.zeros((len(pred), self.scaler.n_features_in_))
                dummy_array[:, 0] = pred
                predictions[name] = self.scaler.inverse_transform(dummy_array)[:, 0]

        return predictions

    def train_and_predict(self, X, y):
        split = int(len(X) * 0.9)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        base_predictions = self.train_base_models(X_train, y_train, X_test)
        meta_features = np.column_stack([base_predictions[name] for name in base_predictions])

        dummy_array = np.zeros((len(y_test), self.scaler.n_features_in_))
        dummy_array[:, 0] = y_test
        actual = self.scaler.inverse_transform(dummy_array)[:, 0]

        self.meta_model.fit(meta_features, actual)
        meta_predictions = self.meta_model.predict(meta_features)

        return meta_predictions.reshape(-1, 1), actual.reshape(-1, 1), split

    def predict_future(self, days=30):
        if self.meta_model is None:
            raise ValueError("Model must be trained before making predictions")

        future_predictions = []
        current_sequence = self.last_sequence.copy()

        for _ in range(days):
            base_preds = []
            for name, model in self.base_models.items():
                if name == 'LSTM':
                    pred = model.predict(current_sequence.reshape(1, self.window_size, -1), verbose=0)[0, 0]
                else:
                    pred = model.predict(current_sequence.reshape(1, -1))[0]

                dummy_array = np.zeros((1, self.scaler.n_features_in_))
                dummy_array[0, 0] = pred
                scaled_pred = self.scaler.inverse_transform(dummy_array)[0, 0]
                base_preds.append(scaled_pred)

            next_pred = self.meta_model.predict([base_preds])[0]
            future_predictions.append(next_pred)

            dummy_array = np.zeros((1, self.scaler.n_features_in_))
            dummy_array[0, 0] = next_pred
            scaled_next = self.scaler.transform(dummy_array)[0, 0]
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = scaled_next

        return np.array(future_predictions)

    def evaluate_model(self, actual, predictions):
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        return {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        st.set_page_config(page_title="Stock Price Prediction", layout="wide")
def main():
   
    
    st.title("Advanced Stock Price Prediction")
    st.write("""
    This app predicts stock prices using an ensemble of machine learning models including LSTM, 
    Gradient Boosting, Random Forest, and SVR.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Enter a stock ticker (e.g., AAPL)").upper()
        prediction_days = st.slider("Number of days to predict", 7, 30, 30)
    
    with col2:
        start_date = st.date_input(
            "Start date",
            value=datetime.now() - timedelta(days=365)
        )
        end_date = st.date_input(
            "End date",
            value=datetime.now()
        )
    
    if st.button("Predict"):
        if not ticker:
            st.warning("Please enter a stock ticker.")
            return
            
        try:
            with st.spinner(f"Processing {ticker}..."):
                predictor = StockPredictorYF()
                
                # Prepare data
                progress_bar = st.progress(0)
                st.write("Fetching data and calculating indicators...")
                df = predictor.prepare_features(ticker, start_date, end_date)
                progress_bar.progress(25)
                
                if len(df) == 0:
                    st.warning(f"No data found for {ticker}")
                    return
                    
                X, y, dates = predictor.prepare_data(df)
                progress_bar.progress(50)
                
                st.write("Training models...")
                predictions, actual, split = predictor.train_and_predict(X, y)
                progress_bar.progress(75)
                
                st.write("Generating future predictions...")
                future_predictions = predictor.predict_future(days=prediction_days)
                progress_bar.progress(100)
                
                # Display results
                test_dates = dates[split:]
                
                # Metrics
                metrics = predictor.evaluate_model(actual, predictions)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MSE", f"{metrics['MSE']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                with col3:
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                with col4:
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                
                # Plot predictions
                fig, ax = plt.subplots(figsize=(15, 7))
                ax.plot(test_dates, actual, label='Actual', color='blue', linewidth=2)
                ax.plot(test_dates, predictions, label='Predictions', color='red', linewidth=2)
                
                # Plot future predictions
                last_date = test_dates[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(len(future_predictions))]
                ax.plot(future_dates, future_predictions, label='Future Predictions',
                       color='green', linestyle='--', linewidth=2)
                
                plt.title(f"{ticker} Stock Price Prediction", fontsize=14, pad=20)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Price', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Display future predictions
                st.write(f"\nPredicted prices for next {prediction_days} days:")
                future_df = pd.DataFrame({
                    'Date': [date.strftime('%Y-%m-%d') for date in future_dates],
                    'Predicted Price': [f"${price:.2f}" for price in future_predictions]
                })
                st.dataframe(future_df)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check if the ticker and dates are valid.")

if __name__ == "__main__":
    main()
