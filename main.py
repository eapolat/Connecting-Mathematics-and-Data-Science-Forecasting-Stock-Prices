import pandas as pd
import numpy as np
from math import log

import matplotlib.pyplot as plt


from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# loading data:

def load_data(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df.set_index('Date')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.set_index('date')
    else:
        raise ValueError("No Date column found. Please check CSV columns.")


    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Adj_Close' in df.columns:
        price_col = 'Adj_Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        raise ValueError("No price column ('Adj Close' or 'Close') found. Please check CSV.")

    df = df[[price_col]].rename(columns={price_col: 'Price'})
    return df


# calculating daily log returns:

def compute_log_returns(df: pd.DataFrame) -> pd.Series:

    prices = df['Price']
    log_prices = np.log(prices)
    returns = log_prices.diff().dropna()
    returns.name = 'LogReturn'
    return returns



def train_test_split_series(series: pd.Series, train_ratio: float = 0.8):

    n = len(series)
    split_idx = int(n * train_ratio)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test



def evaluate_predictions(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    correct_dir = np.sum(np.sign(y_true) == np.sign(y_pred))
    directional_accuracy = correct_dir / len(y_true)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "DirectionalAccuracy": directional_accuracy,
    }

def create_sequences(data: np.ndarray, window_size: int):

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def model_gbm(train_returns, test_returns):

    mu_hat = train_returns.mean()
    sigma_hat = train_returns.std(ddof=1)

    print(f"GBM estimated mu (mean daily return): {mu_hat:.6f}")
    print(f"GBM estimated sigma (std of daily return): {sigma_hat:.6f}")

    preds = np.full_like(test_returns, fill_value=mu_hat, dtype=float)

    return preds

def model_arima(train_returns, test_returns, order=(1, 0, 1)):

    print(f"\nFitting ARIMA{order} on training returns...")
    model = ARIMA(train_returns, order=order)
    fitted = model.fit()
    print(f"ARIMA summary (short):")
    print(fitted.summary().tables[1]) 


    arima_forecast = fitted.forecast(steps=len(test_returns))

    arima_forecast.index = test_returns.index

    return arima_forecast


def model_garch(train_returns, test_returns):

    print("\nFitting GARCH(1,1) with constant mean on training returns...")

    train_ret_pct = train_returns * 100

    am = arch_model(train_ret_pct, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')

    print("GARCH(1,1) parameter estimates:")
    print(res.summary().tables[1])

    mu_hat_pct = res.params['mu']
    mu_hat = mu_hat_pct / 100.0

    print(f"\nGARCH estimated mean daily return (in decimal): {mu_hat:.6f}")

    garch_preds = np.full_like(test_returns, fill_value=mu_hat, dtype=float)

    cond_vol = res.conditional_volatility
    print("\nLast 5 conditional volatilities on training set (in %):")
    print(cond_vol.tail())

    return garch_preds


def model_lstm(train_returns, test_returns,
               window_size: int = 60,
               epochs: int = 20,
               batch_size: int = 32):


    print(f"\nTraining LSTM on returns with window_size={window_size}, epochs={epochs}, batch_size={batch_size}...")

    scaler = StandardScaler()
    train_values = train_returns.values.reshape(-1, 1)
    test_values = test_returns.values.reshape(-1, 1)

    scaler.fit(train_values)

    train_scaled = scaler.transform(train_values)

    X_train, y_train = create_sequences(train_scaled.flatten(), window_size)

    print("LSTM training data shape:", X_train.shape, y_train.shape)

    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size, 1)))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    full_series = np.concatenate([train_values, test_values], axis=0)
    full_scaled = scaler.transform(full_series)
    full_scaled_flat = full_scaled.flatten()

    n_train = len(train_returns)
    n_total = len(full_scaled_flat)


    X_test = []
    for i in range(n_train, n_total):
        window = full_scaled_flat[i - window_size:i]
        X_test.append(window)

    X_test = np.array(X_test).reshape(-1, window_size, 1)
    print("LSTM test data shape:", X_test.shape)

    y_pred_scaled = model.predict(X_test)

    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    lstm_preds = pd.Series(y_pred, index=test_returns.index, name="LSTM_Pred")

    return lstm_preds



def main():

    csv_path = "SPY_prices.csv"
    df = load_data(csv_path)
    returns = compute_log_returns(df)
    train_ret, test_ret = train_test_split_series(returns, train_ratio=0.8)

    print("Data loaded.")
    print("Total observations:", len(returns))
    print("Train size:", len(train_ret))
    print("Test size:", len(test_ret))

    gbm_preds = model_gbm(train_ret, test_ret)


    gbm_metrics = evaluate_predictions(test_ret, gbm_preds)

    print("\n=== GBM Performance on Test Set ===")
    for k, v in gbm_metrics.items():
        if k == "DirectionalAccuracy":
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.6f}")

    arima_preds = model_arima(train_ret, test_ret, order=(1, 0, 1))
    arima_metrics = evaluate_predictions(test_ret, arima_preds)

    print("\n=== ARIMA(1,0,1) Performance on Test Set ===")
    for k, v in arima_metrics.items():
        if k == "DirectionalAccuracy":
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.6f}")

    garch_preds = model_garch(train_ret, test_ret)
    garch_metrics = evaluate_predictions(test_ret, garch_preds)

    print("\n=== GARCH(1,1) Performance on Test Set ===")
    for k, v in garch_metrics.items():
        if k == "DirectionalAccuracy":
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.6f}")


    lstm_preds = model_lstm(train_ret, test_ret,
                            window_size=60,
                            epochs=20,
                            batch_size=32)

    lstm_metrics = evaluate_predictions(test_ret, lstm_preds)

    print("\n=== LSTM Performance on Test Set ===")
    for k, v in lstm_metrics.items():
        if k == "DirectionalAccuracy":
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.6f}")


    plt.figure(figsize=(12, 4))
    plt.plot(df['Price'])
    plt.title("SPY Price Series")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(returns)
    plt.title("Daily Log-Returns of SPY")
    plt.xlabel("Date")
    plt.ylabel("Log-Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    models = ["GBM", "ARIMA", "GARCH", "LSTM"]
    accuracies = [
        gbm_metrics["DirectionalAccuracy"],
        arima_metrics["DirectionalAccuracy"],
        garch_metrics["DirectionalAccuracy"],
        lstm_metrics["DirectionalAccuracy"],
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(models, accuracies)
    plt.title("Directional Accuracy of All Models")
    plt.ylabel("Accuracy")
    plt.ylim(0.45, 0.60)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
