import yfinance as yf
import numpy as np

correct = 0
total = 0

def normalize(x):
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std

def dtw(a, b):
    a = normalize(a)
    b = normalize(b)
    n = len(a)
    m = len(b)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(a[i-1] - b[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )

    return dtw_matrix[n, m]

ticker = "AAPL"
data = yf.download(ticker, start="2024-01-01", end="2025-01-01")
close_data = data["Close"]

for t in range(60, len(close_data) - 10):
    recent_data = close_data[t-30:t]
    results = []

    for i in range(t - 60):
        window = close_data[i:i+30]
        d = dtw(recent_data.values, window.values)
        results.append((d, i))

    results.sort()
    top_5 = results[:5]

    predictions = []

    for d, idx in top_5:
        future = close_data[idx+30:idx+40]

        if len(future) < 10:
            continue

        change_rate = (future.values[-1] - future.values[0]) / future.values[0]

        if change_rate > 0:
            predictions.append(1)
        else:
            predictions.append(0)

    if len(predictions) == 0:
        continue

    prediction = 1 if sum(predictions) >= 3 else 0

    actual_future = close_data[t:t+10]
    actual_change_rate = (actual_future.values[-1] - actual_future.values[0]) / actual_future.values[0]
    actual = 1 if actual_change_rate > 0 else 0

    if prediction == actual:
        correct += 1

    total += 1

print("正解数:", correct)
print("試行回数:", total)

if total > 0:
    accuracy = correct / total
    print("正解率:", accuracy)
else:
    print("評価できません")