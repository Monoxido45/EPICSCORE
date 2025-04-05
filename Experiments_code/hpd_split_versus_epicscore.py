# importing mdn model
from Epistemic_CP.epistemic_models import MDN_model
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set the theme for plots
sns.set_theme(style="whitegrid", rc={"axes.labelsize": 16})
torch.manual_seed(45)
torch.cuda.manual_seed(45)
rng = np.random.default_rng(45)


# function to generate synthetic data
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


# implementing the model
model = MDN_model(
    input_shape=1,
    num_components=1,
    hidden_layers=[120, 64, 32],
    dropout_rate=0.5,
    normalize_y=True,
    base_model_type="density",
)

alpha = 0.1
# considering 500 samples first
# Simulating samples
data_train = generate_data(500, rng)
data_calibration = generate_data(500, rng)
data_test = generate_data(500, rng)

X_train = data_train["x"].to_numpy().reshape(-1, 1)
y_train = data_train["y"].to_numpy()

X_test = data_test["x"].to_numpy().reshape(-1, 1)
y_test = data_test["y"].to_numpy()

X_calib = data_calibration["x"].to_numpy().reshape(-1, 1)
y_calib = data_calibration["y"].to_numpy()

x_grid = np.linspace(data_train["x"].min(), data_train["x"].max(), 750).reshape(-1, 1)

# fitting model to training data
model.fit(
    X_train,
    y_train,
    epochs=1000,
    lr=0.001,
    patience=25,
)

# EPICSCORE part
# computing the density score in calibration set
dens_score = -model.predict(X_calib, y_calib)
with torch.no_grad():
    pi_prime, mu_prime, sigma_prime = model.predict_mcdropout(
        X_calib,
        num_samples=1000,
    )

# Computing new cumulative score s' or s_prime
sample_s = model.mdn_generate_densities(
    pi_prime,
    mu_prime,
    sigma_prime,
)

# computing the predictive CDF cutoff
s_prime_calibration = model.mixture_cdf_no_scale(sample_s, dens_score)

# converting to numpy
s_prime_calibration_np = s_prime_calibration.flatten()
n = s_prime_calibration_np.shape[0]

t_cutoff = np.quantile(s_prime_calibration_np, np.ceil((n + 1) * (1 - alpha)) / n)


# HPD split part
# computing the HPD score
hpd_score = model.mixture_cdf_density(y_calib, X_calib)
n = hpd_score.shape[0]
# computing quantile for HPD split
hpd_quantile = np.quantile(
    hpd_score,
    np.ceil((n + 1) * (1 - alpha)) / n,
)


# computing prediction regions cutoffs
# first for mcdropout
with torch.no_grad():
    pi_test, mu_test, sigma_test = model.predict_mcdropout(x_grid, num_samples=1000)

# computing t_inverse for obtaining region in
# the original non conf score
sample_test = model.mdn_generate_densities(
    pi_test,
    mu_test,
    sigma_test,
)
t_inverse_test = model.mixture_ppf(sample_test, [t_cutoff]).numpy().flatten()


# now for HPD split
t_cutoff_hpd = model.predict_cdf_cutoff(
    x_grid,
    cutoff=hpd_quantile,
    num_samples=1000,
)

# y grid between -6 and 6
y_grid = np.linspace(-6, 6, 750)
# using gridding to compute density for each x_grid
densities = -model.predict_mixture_density(
    x_grid,
    torch.tensor(y_grid),
)

# Creating dictionaries for HPD and EPICSCORE
hpd_dict = {}
epicscore_dict = {}
for i, x_val in enumerate(x_grid.flatten()):
    # Select densities lower than t_cutoff_hpd for HPD
    hpd_dict[x_val] = y_grid[densities[i] <= t_cutoff_hpd[i]]

    # Select densities lower than t_inverse_test for EPICSCORE
    epicscore_dict[x_val] = y_grid[densities[i] <= t_inverse_test[i]]

# Plotting the prediction regions using subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Plotting the prediction regions for HPD
for x_val, y_vals in hpd_dict.items():
    if len(y_vals) > 0:
        axes[0].fill_betweenx(
            y_vals,
            x_val - 0.01,
            x_val + 0.01,
            color="blue",
            alpha=0.3,
        )

# Adding scatter points for the test data
axes[0].scatter(X_test, y_test, color="black", alpha=0.5, s=10)

axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Prediction Regions: HPD")

# Plotting the prediction regions for EPICSCORE
for x_val, y_vals in epicscore_dict.items():
    if len(y_vals) > 0:
        axes[1].fill_betweenx(
            y_vals,
            x_val - 0.01,
            x_val + 0.01,
            color="red",
            alpha=0.3,
        )
axes[1].scatter(X_test, y_test, color="black", alpha=0.5, s=10)
axes[1].set_xlabel("x")
axes[1].set_title("Prediction Regions: EPICSCORE")

plt.tight_layout()
plt.show()
