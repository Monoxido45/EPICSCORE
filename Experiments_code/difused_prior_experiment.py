import numpy as np
import pandas as pd
import seaborn as sns

import torch
from Epistemic_CP.scores import RegressionScore
from Epistemic_CP.epistemic_cp import (
    ECP_split,
)

from sklearn.neighbors import KNeighborsRegressor
from Epistemic_CP.utils import average_interval_score_loss
import matplotlib.pyplot as plt

# Set the theme for plots
sns.set_theme(style="whitegrid", rc={"axes.labelsize": 16})


def generate_data(n, rng):
    # Proportional number of points for dense and sparse regions
    num_dense = round(0.425 * n)  # half of the data points go to dense regions
    num_middle = round(
        0.15 * n
    )  # a small fraction of data points for the middle region

    # Generate x values for dense and sparse regions
    x_dense1 = rng.uniform(0, 1.5, num_dense)
    x_dense2 = rng.uniform(8, 10, num_dense)
    # using beta
    x_middle = (rng.beta(8, 8, num_middle) * (8 - 1.5)) + 1.5
    x_sparse = np.concatenate([x_dense1, x_dense2, x_middle])

    # True function to generate y based on x
    def true_function(x):
        y = 2 * np.sin(x) + rng.normal(0, 0.1, len(x))
        mask = (2 < x) & (x < 7.5)
        y[mask] += rng.normal(0, 2, np.sum(mask))
        return y

    # Generate y values
    y = true_function(x_sparse)

    # Return as a data frame
    return pd.DataFrame({"x": x_sparse, "y": y})


# fitting base model
torch.manual_seed(45)
rng = np.random.default_rng(45)
alpha = 0.1
# considering 500 samples first
# Simulating samples
data_train = generate_data(1000, rng)
data_calibration = generate_data(1000, rng)
data_test = generate_data(1000, rng)

X_test = data_test["x"].to_numpy().reshape(-1, 1)
y_test = data_test["y"].to_numpy()

X_calib = data_calibration["x"].to_numpy().reshape(-1, 1)
y_calib = data_calibration["y"].to_numpy()

# gridding
x_grid = np.linspace(data_train["x"].min(), data_train["x"].max(), 300).reshape(-1, 1)

# fitting base model
X_train = data_train["x"].to_numpy().reshape(-1, 1)
y_train = data_train["y"].to_numpy()

model = KNeighborsRegressor(n_neighbors=10)
model.fit(X_train, y_train)

n_sample = 500

# Plotting the fitted model
plt.figure(figsize=(10, 6))
plt.scatter(data_train["x"], data_train["y"], color="blue", label="Training Data")
plt.plot(x_grid, model.predict(x_grid), color="red", label="Fitted Model")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fitted Model on Training Data")
plt.legend()
plt.show()

# standard defused prior on BART
# fitting ECP, weighted, reg-split and mondrian
ecp_bart_dif = ECP_split(
    RegressionScore,
    base_model=model,
    alpha=alpha,
    is_fitted=True,
)

ecp_bart_dif.fit(X_train, y_train)
t_cutoff_dif = ecp_bart_dif.calib(
    X_calib,
    y_calib,
    epistemic_model="BART",
    random_seed_fit=rng,
    random_seed=42,
    m=30,
    var="heteroscedastic",
    N_samples_MC=500,
    type="gamma",
    normalize_y=False,
    n_cores=8,
    progress=True,
    alpha=0.95,
    beta=2,
)
pred_ecp_bart_dif = ecp_bart_dif.predict(x_grid, random_seed=rng)


# concentrated prior
ecp_bart_con = ECP_split(
    RegressionScore,
    base_model=model,
    alpha=alpha,
    is_fitted=True,
)

ecp_bart_con.fit(X_train, y_train)
t_cutoff_con = ecp_bart_con.calib(
    X_calib,
    y_calib,
    epistemic_model="BART",
    random_seed_fit=rng,
    random_seed=42,
    m=30,
    var="heteroscedastic",
    N_samples_MC=500,
    type="gamma",
    normalize_y=False,
    n_cores=8,
    progress=True,
    alpha=0.15,
    beta=2,
)

pred_ecp_bart_con = ecp_bart_con.predict(x_grid, random_seed=rng)


# Plotting prediction intervals
plt.rcParams.update({"font.size": 32})
fig, axs = plt.subplots(1, 2, figsize=(15, 10))

# Plot for Regression Split
axs[0].scatter(data_test["x"], data_test["y"], color="blue", label="Training Data")
axs[0].plot(x_grid, model.predict(x_grid), color="red")
axs[0].fill_between(
    x_grid.ravel(),
    pred_ecp_bart_dif[:, 0],
    pred_ecp_bart_dif[:, 1],
    color="darkred",
    alpha=0.5,
    label="Difused Prior",
)
axs[0].set_title("Difused BART prior")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

# Plot for Weighted Regression Split
axs[1].scatter(data_test["x"], data_test["y"], color="blue")
axs[1].plot(x_grid, model.predict(x_grid), color="red")
axs[1].fill_between(
    x_grid.ravel(),
    pred_ecp_bart_con[:, 0],
    pred_ecp_bart_con[:, 1],
    color="red",
    alpha=0.5,
    label="Concentrated prior",
)
axs[1].set_title("Concentrated BART prior")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

# Use the same y-axis scale for all plots
ylim = [
    min(ax.get_ylim()[0] for ax in axs.flat),
    max(ax.get_ylim()[1] for ax in axs.flat),
]
for ax in axs.flat:
    ax.set_ylim(ylim)
# Increase font size
for ax in axs.flat:
    ax.title.set_fontsize(30)
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
plt.tight_layout()
plt.show()

# Compute SMIS (Set Membership Indices) for each method on the testing set using AISL
pred_ecp_bart_dif_test = ecp_bart_dif.predict(X_test, random_seed=rng)
pred_ecp_bart_con_test = ecp_bart_con.predict(X_test, random_seed=rng)

# Average Interval Score for Difused Prior
ais_dif = average_interval_score_loss(
    pred_ecp_bart_dif_test[:, 0], pred_ecp_bart_dif_test[:, 1], y_test, alpha=alpha
)
print(f"AIS for Difused Prior: {ais_dif:.4f}")

# Average Interval Score for Concentrated Prior
ais_con = average_interval_score_loss(
    pred_ecp_bart_con_test[:, 0], pred_ecp_bart_con_test[:, 1], y_test, alpha=alpha
)
print(f"AIS for Concentrated Prior: {ais_con:.4f}")
