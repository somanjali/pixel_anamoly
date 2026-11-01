"""STEP 4 - every detector has the same interface:  Detector(seed).fit_score(X) -> scores.
Higher score = more anomalous. Labels are never passed in."""
from functools import partial

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


class LocalRX:
    """Local RX: each pixel is compared with its OWN surroundings (an annulus between an inner guard
    window and an outer window) instead of the whole scene. A panel on grass is judged against grass.
    Works on the first `n_components` principal components so the local covariance is well conditioned.
    Needs the pixel grid, so it is called as fit_score(X, valid, shape)."""
    needs_grid = True

    def __init__(self, seed=0, inner=5, outer=15, n_components=10, ridge=1e-3, chunk=20000):
        assert inner % 2 == 1 and outer % 2 == 1 and outer > inner
        self.inner, self.outer, self.nc, self.ridge, self.chunk = inner, outer, n_components, ridge, chunk

    def fit_score(self, X, valid, shape):
        from scipy.ndimage import uniform_filter
        from sklearn.decomposition import PCA
        Z = X.astype(np.float64)
        if Z.shape[1] > self.nc:
            Z = PCA(self.nc, random_state=0).fit_transform(Z)
        d = Z.shape[1]
        H, W = shape
        vm = valid.reshape(H, W)
        grid = np.zeros((H, W, d))
        grid[vm] = Z                                       # invalid pixels stay 0, so they add nothing to sums
        ao, ai = self.outer ** 2, self.inner ** 2

        def annulus_sum(a):                                # sum over outer window minus inner window
            return (ao * uniform_filter(a, self.outer, mode="constant")
                    - ai * uniform_filter(a, self.inner, mode="constant"))

        n = annulus_sum(vm.astype(np.float64))             # valid samples in each annulus
        iu = np.triu_indices(d)
        s1 = np.stack([annulus_sum(grid[..., i]) for i in range(d)], -1)
        s2 = np.stack([annulus_sum(grid[..., i] * grid[..., j]) for i, j in zip(*iu)], -1)

        ok = vm & (n >= max(3 * d, 30))                    # enough local samples to estimate a covariance
        rows, cols = np.nonzero(ok)
        scores = np.full((H, W), np.nan)
        eye = np.eye(d)
        for a in range(0, len(rows), self.chunk):
            r, c = rows[a:a + self.chunk], cols[a:a + self.chunk]
            nn = n[r, c]
            mu = s1[r, c] / nn[:, None]
            cov = np.empty((len(r), d, d))
            sec = s2[r, c] / nn[:, None]
            cov[:, iu[0], iu[1]] = sec
            cov[:, iu[1], iu[0]] = sec
            cov -= mu[:, :, None] * mu[:, None, :]
            cov *= (nn / (nn - 1))[:, None, None]
            tr = np.trace(cov, axis1=1, axis2=2)
            cov += (self.ridge * tr / d + 1e-9)[:, None, None] * eye
            diff = grid[r, c] - mu
            sol = np.linalg.solve(cov, diff[:, :, None])[:, :, 0]
            scores[r, c] = (diff * sol).sum(axis=1)
        out = scores[vm]
        return np.where(np.isnan(out), np.nanmin(out), out)    # pixels without enough context: least anomalous


DETECTORS = {"rx": RX, "localrx": LocalRX, "localrx_wide": partial(LocalRX, inner=9, outer=25),
             "iforest": IForest, "ocsvm": OCSVM, "autoencoder": Autoencoder}
DETERMINISTIC = {"rx", "localrx", "localrx_wide"}        # no need to repeat over seeds