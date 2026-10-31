"""STEP 4 - every detector has the same interface:  Detector(seed).fit_score(X) -> scores.
Higher score = more anomalous. Labels are never passed in."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class IForest:
    def __init__(self, seed=0, n_estimators=200):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)

    def fit_score(self, X):
        self.model.fit(X)
        return -self.model.score_samples(X)


class OCSVM:
    """One-Class SVM is slow on huge inputs, so fit on a random subsample and score every pixel."""
    def __init__(self, seed=0, n_fit=8000, nu=0.01):
        self.seed, self.n_fit = seed, n_fit
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)

    def fit_score(self, X):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=min(self.n_fit, len(X)), replace=False)
        self.model.fit(X[idx])
        return -self.model.decision_function(X)


class RX:
    """Global RX: Mahalanobis distance from the scene's mean spectrum. Deterministic. The classic baseline."""
    def __init__(self, seed=0):
        pass

    def fit_score(self, X):
        X = X.astype(np.float64)
        d = X - X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        return np.sum((d @ np.linalg.inv(cov)) * d, axis=1)


class Autoencoder:
    """Small MLP autoencoder; the anomaly score is per-pixel reconstruction error."""
    def __init__(self, seed=0, epochs=30, batch_size=512, lr=1e-3):
        self.seed, self.epochs, self.batch_size, self.lr = seed, epochs, batch_size, lr

    def fit_score(self, X):
        import torch                                     # imported here so the other detectors work without torch
        from torch import nn
        torch.manual_seed(self.seed)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        d = X.shape[1]
        hidden, latent = max(d // 2, 8), max(d // 8, 4)
        net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, latent), nn.ReLU(),
                            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, d)).to(dev)
        data = torch.from_numpy(X.astype(np.float32))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)
        n = len(data)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch_size):
                b = data[perm[i:i + self.batch_size]].to(dev)
                loss = ((net(b) - b) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, n, 8192):
                b = data[i:i + 8192].to(dev)
                out.append(((net(b) - b) ** 2).mean(dim=1).cpu())
        return torch.cat(out).numpy()


DETECTORS = {"rx": RX, "iforest": IForest, "ocsvm": OCSVM, "autoencoder": Autoencoder}
DETERMINISTIC = {"rx"}                                   # no need to repeat over seeds