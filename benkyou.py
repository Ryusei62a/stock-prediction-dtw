import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


TICKER = "AAPL"
START_DATE = "2022-01-01"
END_DATE = "2025-01-01"

WINDOW_SIZE = 30
FUTURE_DAYS = 10
EXCLUSION_DAYS = 60
TOP_K = 5
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_np(x):
    x = np.asarray(x, dtype=np.float32).flatten()
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def normalize_torch(x, dim=-1, eps=1e-8):
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, unbiased=False, keepdim=True)
    return (x - mean) / torch.clamp(std, min=eps)


def batched_dtw(query, candidates):
    query = normalize_torch(query.flatten())
    candidates = normalize_torch(candidates, dim=1)

    batch_size, candidate_len = candidates.shape
    query_len = query.shape[0]

    prev = torch.full((batch_size, candidate_len + 1), float("inf"), device=DEVICE)
    prev[:, 0] = 0.0

    for i in range(1, query_len + 1):
        curr = torch.full((batch_size, candidate_len + 1), float("inf"), device=DEVICE)
        for j in range(1, candidate_len + 1):
            cost = torch.abs(query[i - 1] - candidates[:, j - 1])
            curr[:, j] = cost + torch.minimum(
                torch.minimum(prev[:, j], curr[:, j - 1]),
                prev[:, j - 1]
            )
        prev = curr

    return prev[:, candidate_len]


def batched_ddtw(query, candidates):
    if query.shape[0] < 2 or candidates.shape[1] < 2:
        return torch.full((candidates.shape[0],), float("inf"), device=DEVICE)

    return batched_dtw(torch.diff(query), torch.diff(candidates, dim=1))


def make_window_tensor(values, start, end, window_size):
    windows = [values[i:i + window_size] for i in range(start, end)]
    return torch.tensor(np.asarray(windows, dtype=np.float32), device=DEVICE)


def find_similar_patterns(close_data, t, window_size=WINDOW_SIZE):
    candidate_end = t - EXCLUSION_DAYS
    if candidate_end < TOP_K:
        return None

    recent_close = close_data[t - window_size:t]
    recent_tensor = torch.tensor(recent_close, dtype=torch.float32, device=DEVICE)
    close_windows = make_window_tensor(close_data, 0, candidate_end, window_size)

    with torch.no_grad():
        dtw_distances = batched_dtw(recent_tensor, close_windows)
        ddtw_distances = batched_ddtw(recent_tensor, close_windows)
        distance_scores = dtw_distances + 0.2 * ddtw_distances
        top_scores, top_indices = torch.topk(distance_scores, TOP_K, largest=False)

    return {
        "recent_close": recent_close,
        "top_scores": top_scores.cpu().numpy(),
        "top_indices": top_indices.cpu().numpy(),
        "dtw_distances": dtw_distances[top_indices].cpu().numpy(),
        "ddtw_distances": ddtw_distances[top_indices].cpu().numpy(),
    }


def make_one_sample(close_data, t, window_size=WINDOW_SIZE, future_days=FUTURE_DAYS):
    similar = find_similar_patterns(close_data, t, window_size)
    if similar is None:
        return None, None

    dtw_features = []
    similar_cases = []

    for score, idx, dtw_distance, ddtw_distance in zip(
        similar["top_scores"],
        similar["top_indices"],
        similar["dtw_distances"],
        similar["ddtw_distances"],
    ):
        future = close_data[idx + window_size:idx + window_size + future_days]
        if len(future) < future_days:
            return None, None

        future_return = (future[-1] - future[0]) / future[0]
        dtw_features.extend([score, dtw_distance, ddtw_distance, future_return])
        similar_cases.append(
            {
                "idx": int(idx),
                "score": float(score),
                "dtw": float(dtw_distance),
                "ddtw": float(ddtw_distance),
                "future_return": float(future_return),
            }
        )

    price_norm = normalize_np(similar["recent_close"])
    seq_features = price_norm.reshape(-1, 1)

    dtw_features = np.array(dtw_features, dtype=np.float32)
    dtw_repeated = np.tile(dtw_features, (window_size, 1))
    x_one = np.concatenate([seq_features, dtw_repeated], axis=1)

    return x_one.astype(np.float32), similar_cases


def make_dataset(close_data, window_size=WINDOW_SIZE, future_days=FUTURE_DAYS):
    X = []
    y = []

    for t in range(EXCLUSION_DAYS, len(close_data) - future_days):
        x_one, _ = make_one_sample(close_data, t, window_size, future_days)
        if x_one is None:
            continue

        future = close_data[t:t + future_days]
        future_change = (future[-1] - future[0]) / future[0]
        label = 1 if future_change > 0 else 0

        X.append(x_one)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


def calculate_metrics(labels, preds, probs=None):
    metrics = {
        "acc": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": np.nan,
    }

    if probs is not None and len(np.unique(labels)) == 2:
        metrics["auc"] = roc_auc_score(labels, probs)

    return metrics


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * X_batch.size(0)
            total += y_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return total_loss / total, calculate_metrics(all_labels, all_preds, all_probs)


def print_baselines(y_test):
    always_up_preds = np.ones_like(y_test)
    majority_label = np.bincount(y_test).argmax()
    majority_preds = np.full_like(y_test, majority_label)

    always_up = calculate_metrics(y_test, always_up_preds)
    majority = calculate_metrics(y_test, majority_preds)

    print(
        "Baseline always up | "
        f"Acc: {always_up['acc']:.4f}, "
        f"Precision: {always_up['precision']:.4f}, "
        f"Recall: {always_up['recall']:.4f}, "
        f"F1: {always_up['f1']:.4f}"
    )
    print(
        "Baseline majority  | "
        f"Acc: {majority['acc']:.4f}, "
        f"Precision: {majority['precision']:.4f}, "
        f"Recall: {majority['recall']:.4f}, "
        f"F1: {majority['f1']:.4f}"
    )


def predict_latest(model, close_data):
    x_latest, similar_cases = make_one_sample(close_data, len(close_data), WINDOW_SIZE, FUTURE_DAYS)
    if x_latest is None:
        print("Not enough data for latest prediction.")
        return

    model.eval()
    with torch.no_grad():
        X_latest = torch.tensor(x_latest, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        outputs = model(X_latest)
        up_probability = torch.softmax(outputs, dim=1)[0, 1].item()

    print("\nLatest prediction")
    print(f"Ticker: {TICKER}")
    print(f"Up probability over next {FUTURE_DAYS} days: {up_probability:.4f}")
    print("Prediction:", "UP" if up_probability >= 0.5 else "DOWN")
    print("Similar past patterns:")

    for rank, case in enumerate(similar_cases, start=1):
        print(
            f"{rank}. index={case['idx']}, "
            f"score={case['score']:.4f}, "
            f"dtw={case['dtw']:.4f}, "
            f"ddtw={case['ddtw']:.4f}, "
            f"future_return={case['future_return']:.4f}"
        )


def main():
    print("DEVICE:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    data = yf.download(TICKER, start=START_DATE, end=END_DATE)
    close_data = np.asarray(data["Close"]).flatten().astype(np.float32)

    print("close_data shape:", close_data.shape)

    X, y = make_dataset(close_data, WINDOW_SIZE, FUTURE_DAYS)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    label_values, label_counts = np.unique(y, return_counts=True)
    label_distribution = dict(zip(label_values.tolist(), label_counts.tolist()))
    print("label distribution:", label_distribution)
    print(f"positive ratio: {np.mean(y == 1):.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    train_loader = DataLoader(
        StockDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        StockDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = StockTransformer(input_dim=X.shape[2]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print_baselines(y_test)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_metrics = evaluate(model, test_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Acc: {test_metrics['acc']:.4f}, "
            f"Precision: {test_metrics['precision']:.4f}, "
            f"Recall: {test_metrics['recall']:.4f}, "
            f"F1: {test_metrics['f1']:.4f}, "
            f"AUC: {test_metrics['auc']:.4f}"
        )

    predict_latest(model, close_data)


if __name__ == "__main__":
    main()
