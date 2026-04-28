"""Reusable TimeMCL utilities for taxi demand benchmarking.

This module keeps notebook cells clean and exposes:
- train_time_mcl_on_panel: train multi-hypothesis model with AWTA schedule
- build_timemcl_outputs: point + probabilistic outputs on test split
- probabilistic_scores: calibration/sharpness style summary metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class TimeMCLConfig:
    window: int = 24
    horizon: int = 1
    num_hyps: int = 8
    hidden: int = 128
    epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42
    temp_start: float = 3.0
    temp_end: float = 0.2
    diversity_weight: float = 0.08
    max_train_windows: int | None = 4000
    verbose: bool = True
    progress_every: int = 1
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    dropout: float = 0.1
    early_stopping_patience: int = 8
    balance_weight: float = 0.03


class _WinDS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y[:, 0, :], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


class TimeMCLNet(nn.Module):
    def __init__(self, window: int, n_feat: int, hidden: int, num_hyps: int, dropout: float = 0.1):
        super().__init__()
        self.window = window
        self.n_feat = n_feat
        self.backbone = nn.Sequential(
            nn.Linear(window * n_feat, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n_feat) for _ in range(num_hyps)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h = self.backbone(x.reshape(b, -1))
        return torch.stack([head(h) for head in self.heads], dim=1)  # [B, H, F]


def _awta_loss(preds: torch.Tensor, target: torch.Tensor, temp: float) -> torch.Tensor:
    dist = ((preds - target.unsqueeze(1)) ** 2).mean(dim=2)  # [B, H]
    w = torch.softmax(-dist / max(temp, 1e-6), dim=1)
    return (w * dist).sum(dim=1).mean()


def _head_balance_loss(preds: torch.Tensor, target: torch.Tensor, temp: float) -> torch.Tensor:
    """
    Encourage all heads to be used across the batch.
    We minimize KL(avg_assignment || uniform), which is 0 when balanced.
    """
    dist = ((preds - target.unsqueeze(1)) ** 2).mean(dim=2)  # [B, H]
    w = torch.softmax(-dist / max(temp, 1e-6), dim=1)  # [B, H]
    avg_w = w.mean(dim=0)  # [H]
    h = avg_w.numel()
    uniform = torch.full_like(avg_w, 1.0 / h)
    eps = 1e-8
    kl = torch.sum(avg_w * (torch.log(avg_w + eps) - torch.log(uniform + eps)))
    return kl


def _diversity_repulsion_loss(preds: torch.Tensor) -> torch.Tensor:
    """
    Encourage heads to spread out by penalizing similarity.
    Lower is better; minimizing this pushes pairwise distances up.
    """
    b, h, _ = preds.shape
    if h < 2:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    d = torch.cdist(preds, preds, p=2)  # [B, H, H]
    sim = torch.exp(-d)
    eye = torch.eye(h, device=preds.device, dtype=preds.dtype).unsqueeze(0)
    off_diag = sim * (1.0 - eye)
    return off_diag.sum() / (b * h * (h - 1))


def _split_code(split: str) -> int:
    return {"train": 0, "val": 1, "test": 2}[split]


def _make_windows(values: np.ndarray, split_codes: np.ndarray, window: int, horizon: int):
    X, Y, S, H = [], [], [], []
    n = len(values)
    for t in range(window, n - horizon + 1):
        X.append(values[t - window : t])
        Y.append(values[t : t + horizon])
        S.append(split_codes[t])
        H.append(t)
    return np.array(X), np.array(Y), np.array(S), np.array(H)


def train_time_mcl_on_panel(panel_df: pd.DataFrame, top_zones: List[int], config: TimeMCLConfig):
    if config.verbose:
        print(
            f"[TimeMCL] entering train_time_mcl_on_panel | rows={len(panel_df):,} | "
            f"zones={len(top_zones)} | window={config.window} | horizon={config.horizon}"
        )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if config.verbose:
        print(f"[TimeMCL] device resolved: {device}")

    # Keep only required columns and top zones before building the matrix
    panel = panel_df.loc[panel_df["PULocationID"].isin(top_zones), ["pickup_hour", "PULocationID", "pickup_count", "split"]].copy()
    panel = panel.sort_values(["pickup_hour", "PULocationID"])
    panel = panel.drop_duplicates(subset=["pickup_hour", "PULocationID"], keep="last")

    if config.verbose:
        print(f"[TimeMCL] panel reduced for pivot | rows={len(panel):,}")

    # Faster than pivot_table(..., aggfunc='first') for this usage
    wide = panel.set_index(["pickup_hour", "PULocationID"])["pickup_count"].unstack("PULocationID").sort_index()
    wide = wide.reindex(columns=top_zones).fillna(0.0)

    if config.verbose:
        print(
            f"[TimeMCL] wide matrix built | hours={len(wide):,} | "
            f"time_span=({wide.index.min()} -> {wide.index.max()})"
        )

    hour_split = panel[["pickup_hour", "split"]].drop_duplicates().sort_values("pickup_hour")
    split_map = hour_split.set_index("pickup_hour")["split"]

    train_idx = split_map[split_map == "train"].index
    mu = wide.loc[train_idx].mean(axis=0)
    sigma = wide.loc[train_idx].std(axis=0).replace(0, 1.0)
    wide_z = (wide - mu) / sigma

    arr = wide_z.values.astype(np.float32)
    hours = wide_z.index.to_numpy()
    split_codes = np.array([_split_code(split_map[h]) for h in hours])

    if config.verbose:
        print(f"[TimeMCL] standardized matrix ready | arr_shape={arr.shape}")

    Xw, Yw, Sw, Hw = _make_windows(arr, split_codes, config.window, config.horizon)

    train_mask = Sw == 0
    val_mask = Sw == 1
    test_mask = Sw == 2

    Xw_train = Xw[train_mask]
    Yw_train = Yw[train_mask]
    if config.max_train_windows is not None and len(Xw_train) > config.max_train_windows:
        keep_idx = np.linspace(0, len(Xw_train) - 1, num=config.max_train_windows, dtype=int)
        Xw_train = Xw_train[keep_idx]
        Yw_train = Yw_train[keep_idx]

    train_dl = DataLoader(_WinDS(Xw_train, Yw_train), batch_size=config.batch_size, shuffle=True)
    val_dl = DataLoader(_WinDS(Xw[val_mask], Yw[val_mask]), batch_size=config.batch_size, shuffle=False)

    n_feat = Xw.shape[-1]
    model = TimeMCLNet(config.window, n_feat, config.hidden, config.num_hyps, dropout=config.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, config.epochs))

    best_val = np.inf
    best_state = None
    train_trace, val_trace = [], []
    no_improve = 0

    if config.verbose:
        print(
            f"[TimeMCL] device={device} | n_feat={n_feat} | "
            f"windows(train/val/test)={len(Xw_train)}/{int(val_mask.sum())}/{int(test_mask.sum())}"
        )

    for epoch in range(1, config.epochs + 1):
        model.train()
        progress = (epoch - 1) / max(1, config.epochs - 1)
        temp = config.temp_start * ((config.temp_end / max(config.temp_start, 1e-8)) ** progress)
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss_fit = _awta_loss(pred, yb, temp=temp)
            loss_div = _diversity_repulsion_loss(pred)
            loss_bal = _head_balance_loss(pred, yb, temp=temp)
            loss = loss_fit + config.diversity_weight * loss_div + config.balance_weight * loss_bal
            loss.backward()
            if config.grad_clip_norm is not None and config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            opt.step()
            tr_loss += float(loss.item()) * len(xb)
        tr_loss /= max(1, len(train_dl.dataset))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = _awta_loss(pred, yb, temp=0.4)
                va_loss += float(loss.item()) * len(xb)
        va_loss /= max(1, len(val_dl.dataset))

        train_trace.append(tr_loss)
        val_trace.append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step()

        if config.verbose and (epoch % max(1, config.progress_every) == 0 or epoch == 1 or epoch == config.epochs):
            print(
                f"[TimeMCL] epoch {epoch:02d}/{config.epochs} | "
                f"temp={temp:.4f} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | lr={opt.param_groups[0]['lr']:.2e}"
            )

        if no_improve >= config.early_stopping_patience:
            if config.verbose:
                print(f"[TimeMCL] early stop at epoch {epoch} (patience={config.early_stopping_patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.to(device).eval()
    return {
        "model": model,
        "device": device,
        "Xw": Xw,
        "Yw": Yw,
        "Sw": Sw,
        "Hw": Hw,
        "hours": hours,
        "mu": mu,
        "sigma": sigma,
        "top_zones": top_zones,
        "test_mask": test_mask,
        "train_trace": train_trace,
        "val_trace": val_trace,
    }


def build_timemcl_outputs(state: Dict, point_policy: str = "mean") -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = state["model"]
    device = state["device"]
    Xw, Yw = state["Xw"], state["Yw"]
    test_mask, Hw, hours = state["test_mask"], state["Hw"], state["hours"]
    Sw = state["Sw"]
    mu, sigma = state["mu"], state["sigma"]
    top_zones = state["top_zones"]

    Xt = torch.tensor(Xw[test_mask], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_h = model(Xt).cpu().numpy()  # [N, H, F]

    y_true_z = Yw[test_mask][:, 0, :]
    y_true = y_true_z * sigma.values + mu.values
    pred_h_denorm = pred_h * sigma.values[None, None, :] + mu.values[None, None, :]
    pred_h_denorm = np.clip(pred_h_denorm, 0, None)

    if point_policy == "median":
        point = np.median(pred_h_denorm, axis=1)
    elif point_policy == "val_best_head":
        val_mask = Sw == 1
        if val_mask.sum() == 0:
            point = np.median(pred_h_denorm, axis=1)
        else:
            Xv = torch.tensor(Xw[val_mask], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_val_h = model(Xv).cpu().numpy()  # [Nv, H, F]

            y_val_z = Yw[val_mask][:, 0, :]
            y_val = y_val_z * sigma.values + mu.values
            pred_val_denorm = np.clip(pred_val_h * sigma.values[None, None, :] + mu.values[None, None, :], 0, None)

            # Choose best head per zone by validation MAE
            # pred_val_denorm: [Nv, H, F]
            abs_err = np.abs(pred_val_denorm - y_val[:, None, :])  # [Nv, H, F]
            mae_by_head_zone = abs_err.mean(axis=0)  # [H, F]
            best_head_by_zone = mae_by_head_zone.argmin(axis=0)  # [F]

            # Apply zone-specific best head on test
            n_test, _, n_feat = pred_h_denorm.shape
            point = np.empty((n_test, n_feat), dtype=pred_h_denorm.dtype)
            for j in range(n_feat):
                point[:, j] = pred_h_denorm[:, best_head_by_zone[j], j]
    else:
        point = np.mean(pred_h_denorm, axis=1)

    test_hours = hours[Hw[test_mask]]

    point_rows, prob_rows = [], []
    for j, zone in enumerate(top_zones):
        point_rows.append(
            pd.DataFrame(
                {
                    "model": "TimeMCL",
                    "PULocationID": zone,
                    "pickup_hour": test_hours,
                    "y_true": y_true[:, j],
                    "y_pred": point[:, j],
                }
            )
        )

        for h in range(pred_h_denorm.shape[1]):
            prob_rows.append(
                pd.DataFrame(
                    {
                        "model": "TimeMCL",
                        "PULocationID": zone,
                        "pickup_hour": test_hours,
                        "hypothesis": h,
                        "y_true": y_true[:, j],
                        "y_pred_h": pred_h_denorm[:, h, j],
                    }
                )
            )

    point_df = pd.concat(point_rows, ignore_index=True)
    prob_df = pd.concat(prob_rows, ignore_index=True)
    return point_df, prob_df


def _pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    e = y - q_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def probabilistic_scores(prob_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zone, g in prob_df.groupby("PULocationID", observed=True):
        pivot = g.pivot_table(index="pickup_hour", columns="hypothesis", values="y_pred_h")
        truth = g.drop_duplicates("pickup_hour").set_index("pickup_hour")["y_true"].reindex(pivot.index).values

        q10 = np.quantile(pivot.values, 0.10, axis=1)
        q50 = np.quantile(pivot.values, 0.50, axis=1)
        q90 = np.quantile(pivot.values, 0.90, axis=1)

        coverage_80 = float(np.mean((truth >= q10) & (truth <= q90)))
        width_80 = float(np.mean(q90 - q10))

        rows.append(
            {
                "model": "TimeMCL",
                "PULocationID": int(zone),
                "pinball_q10": _pinball(truth, q10, 0.10),
                "pinball_q50": _pinball(truth, q50, 0.50),
                "pinball_q90": _pinball(truth, q90, 0.90),
                "coverage_10_90": coverage_80,
                "mean_width_10_90": width_80,
            }
        )

    out = pd.DataFrame(rows)
    macro = out[["pinball_q10", "pinball_q50", "pinball_q90", "coverage_10_90", "mean_width_10_90"]].mean().to_dict()
    macro.update({"model": "TimeMCL", "PULocationID": "macro_avg"})
    return pd.concat([out, pd.DataFrame([macro])], ignore_index=True)
