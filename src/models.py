from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class OutcomeNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class OutcomeModels:
    model_0: OutcomeNet
    model_1: OutcomeNet
    device: torch.device

    def predict(self, features: np.ndarray, arm: int, batch_size: int = 4096) -> np.ndarray:
        model = self.model_1 if arm == 1 else self.model_0
        model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=self.device)
                predictions.append(model(batch).detach().cpu().numpy())
        return np.concatenate(predictions)


def fit_outcome_models(
    features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    seed: int,
    epochs: int = 90,
    batch_size: int = 512,
) -> OutcomeModels:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        0: OutcomeNet(features.shape[1]).to(device),
        1: OutcomeNet(features.shape[1]).to(device),
    }
    for arm, model in models.items():
        mask = treatment == arm
        arm_features = torch.as_tensor(features[mask], dtype=torch.float32)
        arm_outcome = torch.as_tensor(outcome[mask], dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(arm_features, arm_outcome),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        model.train()
        for _ in range(epochs):
            for batch_features, batch_outcome in loader:
                batch_features = batch_features.to(device)
                batch_outcome = batch_outcome.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch_features), batch_outcome)
                loss.backward()
                optimizer.step()
    return OutcomeModels(model_0=models[0], model_1=models[1], device=device)

