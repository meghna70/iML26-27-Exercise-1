"""
IML WS 26/27 - Exercise 1, Task for 1 bonus point
How does model fit (underfit -> good fit -> overfit) change feature effect explanations?

Datasets : Pima Indians Diabetes (clf), German Credit (clf), California Housing (reg)
Models   : DecisionTree + GradientBoosting (classifier / regressor)
Outputs  : figures/*.png, results/summary_metrics.csv, printed tables

Requirements: scikit-learn, pandas, numpy, matplotlib
    pip install scikit-learn pandas numpy matplotlib
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]       # each seed = model trained on a different 80% subsample
N_BACKGROUND = 300            # instances used for PDP / ICE
N_ICE_PLOT = 30               # ICE curves drawn
N_GRID = 20                   # grid points per feature
N_BINS_ALE = 20               # ALE bins
CALIFORNIA_MAX_ROWS = 8000    # subsample California for speed (None = all 20640)

# Features whose effect we explain (numeric ones, easy to interpret)
FEATURES = {
    "Pima Diabetes":      ["plas", "mass"],                 # glucose, BMI
    "German Credit":      ["age", "duration"],
    "California Housing": ["MedInc", "AveOccup"],
}

# Complexity ladders, ordered from simple (underfit) to complex (overfit)
DT_GRID = [dict(max_depth=d) for d in [1, 2, 3, 5, 8, 12, None]]
GB_GRID = [
    dict(n_estimators=5,   max_depth=1, learning_rate=0.05),
    dict(n_estimators=20,  max_depth=1, learning_rate=0.1),
    dict(n_estimators=100, max_depth=2, learning_rate=0.1),
    dict(n_estimators=200, max_depth=3, learning_rate=0.1),
    dict(n_estimators=300, max_depth=5, learning_rate=0.2),
    dict(n_estimators=300, max_depth=8, learning_rate=0.3),
]
MODELS = {
    "clf": {"Decision Tree": (DecisionTreeClassifier, DT_GRID),
            "Gradient Boosting": (GradientBoostingClassifier, GB_GRID)},
    "reg": {"Decision Tree": (DecisionTreeRegressor, DT_GRID),
            "Gradient Boosting": (GradientBoostingRegressor, GB_GRID)},
}


# ----------------------------------------------------------------------------
# Data loading  (returns X: DataFrame, y: array, task: "clf"/"reg")
# ----------------------------------------------------------------------------
def load_pima():
    X, y = fetch_openml(name="diabetes", version=1, as_frame=True, return_X_y=True)
    y = (y == "tested_positive").astype(int).values
    X = X.astype(float).copy()
    # In Pima, 0 in these columns means "missing" -> median imputation
    for c in ["plas", "pres", "skin", "insu", "mass"]:
        X[c] = X[c].replace(0, np.nan)
    X = X.fillna(X.median())
    return X, y, "clf"


def load_german():
    X, y = fetch_openml(name="credit-g", version=1, as_frame=True, return_X_y=True)
    y = (y == "bad").astype(int).values          # target = P(bad credit)
    X = pd.get_dummies(X, dtype=float)           # one-hot encode categoricals
    for c in ["age", "duration", "credit_amount"]:
        X[c] = X[c].astype(float)
    return X, y, "clf"


def load_california():
    d = fetch_california_housing(as_frame=True)
    X, y = d.data, d.target.values               # median house value (100k USD)
    if CALIFORNIA_MAX_ROWS and len(X) > CALIFORNIA_MAX_ROWS:
        idx = np.random.RandomState(0).choice(len(X), CALIFORNIA_MAX_ROWS, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    return X, y, "reg"


LOADERS = {
    "Pima Diabetes": load_pima,
    "German Credit": load_german,
    "California Housing": load_california,
}


# ----------------------------------------------------------------------------
# Helpers: prediction function, scoring, feature-effect methods
# ----------------------------------------------------------------------------
def make_predict(model, task):
    if task == "clf":
        return lambda X: model.predict_proba(X)[:, 1]
    return lambda X: model.predict(X)


def score(model, X, y, task):
    if task == "clf":
        return roc_auc_score(y, model.predict_proba(X)[:, 1])
    return r2_score(y, model.predict(X))


def ice_pdp(predict, X_bg, feat, grid):
    """ICE matrix (n_instances x n_grid); PDP = mean over instances."""
    ice = np.empty((len(X_bg), len(grid)))
    for j, g in enumerate(grid):
        Xg = X_bg.copy()
        Xg[feat] = g
        ice[:, j] = predict(Xg)
    return ice, ice.mean(axis=0)


def ale_1d(predict, X, feat, grid, n_bins=N_BINS_ALE):
    """First-order ALE (quantile bins), interpolated onto `grid`, centred."""
    x = X[feat].values
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.zeros_like(grid, dtype=float)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    lo, hi = X.copy(), X.copy()
    lo[feat], hi[feat] = edges[idx], edges[idx + 1]
    diff = predict(hi) - predict(lo)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    local = np.array([diff[idx == k].mean() if counts[k] else 0.0
                      for k in range(len(edges) - 1)])
    ale = np.concatenate([[0.0], np.cumsum(local)])
    mid = (ale[:-1] + ale[1:]) / 2
    ale -= (mid * counts).sum() / counts.sum()
    return np.interp(grid, edges, ale)


def center(curves):
    """Subtract each curve's mean over the grid (so only the *shape* is compared)."""
    curves = np.atleast_2d(curves)
    return curves - curves.mean(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Step 1+2: fit the complexity ladder, record train/test performance
# ----------------------------------------------------------------------------
def cfg_label(cfg):
    if "n_estimators" in cfg:
        return f"n={cfg['n_estimators']}\nd={cfg['max_depth']}\nlr={cfg['learning_rate']}"
    return f"depth={cfg['max_depth']}"


def fit_ladder(model_cls, grid, Xtr, ytr, Xte, yte, task):
    rows = []
    for cfg in grid:
        m = model_cls(random_state=0, **cfg).fit(Xtr, ytr)
        tr, te = score(m, Xtr, ytr, task), score(m, Xte, yte, task)
        rows.append(dict(config=cfg, label=cfg_label(cfg), train=tr, test=te, gap=tr - te))
    return pd.DataFrame(rows)


def pick_levels(ladder):
    """underfit = simplest config, good = best test score, overfit = most complex config."""
    return {"Underfit": 0, "Good fit": int(ladder["test"].values.argmax()),
            "Overfit": len(ladder) - 1}


def plot_ladder(name, task, ladders, levels_per_model):
    metric = "ROC-AUC" if task == "clf" else "R²"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, (mname, lad) in zip(axes, ladders.items()):
        xs = np.arange(len(lad))
        ax.plot(xs, lad["train"], "o-", label="train", color="tab:blue")
        ax.plot(xs, lad["test"], "s-", label="test", color="tab:red")
        for lvl, i in levels_per_model[mname].items():
            ax.axvline(i, color="gray", ls=":", alpha=.6)
            ax.text(i, ax.get_ylim()[0], lvl, rotation=90, va="bottom", ha="right",
                    fontsize=8, color="gray")
        ax.set_xticks(xs)
        ax.set_xticklabels(lad["label"], fontsize=7)
        ax.set_xlabel("model complexity  →")
        ax.set_ylabel(metric)
        ax.set_title(f"{mname}")
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle(f"{name}: train vs. test performance along the complexity ladder")
    fig.tight_layout()
    fig.savefig(f"figures/{name.replace(' ', '_')}_complexity.png", dpi=150)
    plt.show()


# ----------------------------------------------------------------------------
# Step 3: feature effects for underfit / good / overfit models over several seeds
# ----------------------------------------------------------------------------
def explain_level(model_cls, cfg, Xtr, ytr, Xte, yte, Xbg, feat, grid, task):
    """Train on SEEDS different 80% subsamples; return curves + metrics."""
    pdps, ales, tests = [], [], []
    ice_first, hetero, rough = None, [], []
    for s in SEEDS:
        rng = np.random.RandomState(s)
        sub = rng.choice(len(Xtr), int(0.8 * len(Xtr)), replace=False)
        m = model_cls(random_state=s, **cfg).fit(Xtr.iloc[sub], ytr[sub])
        pred = make_predict(m, task)
        ice, pdp = ice_pdp(pred, Xbg, feat, grid)
        ale = ale_1d(pred, Xtr.iloc[sub], feat, grid)
        if ice_first is None:
            ice_first = center(ice)
        hetero.append(center(ice).std(axis=0).mean())      # ICE spread
        pdps.append(center(pdp)[0])
        ales.append(center(ale)[0])
        tests.append(score(m, Xte, yte, task))
    pdps, ales = np.array(pdps), np.array(ales)
    pdp_m, ale_m = pdps.mean(0), ales.mean(0)
    slope_sign = np.sign(np.diff(pdp_m)[np.abs(np.diff(pdp_m)) > 1e-9])
    metrics = dict(
        test_score=np.mean(tests),
        ice_heterogeneity=np.mean(hetero),                   # interactions / local variation
        pdp_ale_gap=np.mean(np.abs(pdp_m - ale_m)),          # PDP vs ALE disagreement
        seed_std=pdps.std(axis=0).mean(),                    # consistency across seeds
        roughness=np.abs(np.diff(pdp_m)).sum(),              # total variation of PDP
        slope_sign_changes=int((np.diff(slope_sign) != 0).sum()),  # continuity / monotonicity
    )
    return dict(ice=ice_first, pdp_m=pdp_m, pdp_s=pdps.std(0), ale_m=ale_m,
                ale_s=ales.std(0), metrics=metrics)


def plot_effects(dname, mname, feat, grid, results, metric_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (lvl, r) in zip(axes, results.items()):
        for c in r["ice"][:N_ICE_PLOT]:
            ax.plot(grid, c, color="gray", alpha=.35, lw=.8)
        ax.plot(grid, r["pdp_m"], color="tab:blue", lw=2.5, label="PDP ± seed std")
        ax.fill_between(grid, r["pdp_m"] - r["pdp_s"], r["pdp_m"] + r["pdp_s"],
                        color="tab:blue", alpha=.25)
        ax.plot(grid, r["ale_m"], color="tab:orange", lw=2.5, ls="--", label="ALE ± seed std")
        ax.fill_between(grid, r["ale_m"] - r["ale_s"], r["ale_m"] + r["ale_s"],
                        color="tab:orange", alpha=.25)
        ax.set_title(f"{lvl} ({metric_name} {r['metrics']['test_score']:.2f})")
        ax.set_xlabel(feat)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("centred effect on prediction")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{dname} - {mname} - feature: {feat}   (grey = ICE)")
    fig.tight_layout()
    fname = f"figures/{dname}_{mname}_{feat}_effects.png".replace(" ", "_")
    fig.savefig(fname, dpi=150)
    plt.show()


def plot_summary(df):
    mets = ["ice_heterogeneity", "pdp_ale_gap", "seed_std", "roughness"]
    titles = ["ICE heterogeneity", "|PDP − ALE| gap", "Seed std of PDP (inconsistency)",
              "PDP roughness (total variation)"]
    order = ["Underfit", "Good fit", "Overfit"]
    g = df.groupby(["dataset", "model", "level"])[mets].mean().reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, m, t in zip(axes, mets, titles):
        for (d, mo), sub in g.groupby(["dataset", "model"]):
            sub = sub.set_index("level").loc[order]
            ax.plot(order, sub[m], "o-", label=f"{d} / {mo}")
        ax.set_title(t)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig("figures/summary_metrics.png", dpi=150)
    plt.show()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(loaders=LOADERS):
    all_rows, ladder_rows = [], []
    for dname, loader in loaders.items():
        X, y, task = loader()
        metric_name = "AUC" if task == "clf" else "R²"
        strat = y if task == "clf" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42,
                                              stratify=strat)
        Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
        Xbg = Xtr.sample(min(N_BACKGROUND, len(Xtr)), random_state=0).reset_index(drop=True)
        print(f"\n{'='*80}\n{dname}: {len(X)} rows, {X.shape[1]} features, task={task}\n{'='*80}")

        ladders, levels_per_model = {}, {}
        for mname, (cls, grid) in MODELS[task].items():
            lad = fit_ladder(cls, grid, Xtr, ytr, Xte, yte, task)
            lvl = pick_levels(lad)
            ladders[mname], levels_per_model[mname] = lad, lvl
            print(f"\n--- {mname}: train vs test ({metric_name}) ---")
            print(lad[["label", "train", "test", "gap"]]
                  .assign(label=lad["label"].str.replace("\n", " "))
                  .round(3).to_string(index=False))
            for l, i in lvl.items():
                ladder_rows.append(dict(dataset=dname, model=mname, level=l,
                                        config=str(lad.loc[i, "config"]),
                                        train=lad.loc[i, "train"], test=lad.loc[i, "test"],
                                        gap=lad.loc[i, "gap"]))
        plot_ladder(dname, task, ladders, levels_per_model)

        for mname, (cls, grid) in MODELS[task].items():
            lad, lvl = ladders[mname], levels_per_model[mname]
            for feat in FEATURES[dname]:
                if feat not in X.columns:
                    print(f"  (skip: feature {feat} not in data)")
                    continue
                lo, hi = np.quantile(Xtr[feat], [0.05, 0.95])
                grid_x = np.unique(np.linspace(lo, hi, N_GRID))
                res = {}
                for l, i in lvl.items():
                    res[l] = explain_level(cls, lad.loc[i, "config"], Xtr, ytr, Xte, yte,
                                           Xbg, feat, grid_x, task)
                    all_rows.append(dict(dataset=dname, model=mname, feature=feat,
                                         level=l, **res[l]["metrics"]))
                plot_effects(dname, mname, feat, grid_x, res, metric_name)

    summary = pd.DataFrame(all_rows)
    summary.to_csv("results/summary_metrics.csv", index=False)
    pd.DataFrame(ladder_rows).to_csv("results/train_test_selected_models.csv", index=False)
    print("\n" + "=" * 80 + "\nSUMMARY: explanation metrics per fit level\n" + "=" * 80)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print("\nTrain/test of the selected models (use this to argue under-/overfit):")
    print(pd.DataFrame(ladder_rows).round(3).to_string(index=False))
    plot_summary(summary)
    return summary


if __name__ == "__main__":
    run()