from Epistemic_CP.epistemic_models import MDN_model, BART_model
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set the theme for plots
sns.set_theme(style="whitegrid", rc={"axes.labelsize": 16})
torch.manual_seed(25)
torch.cuda.manual_seed(25)
rng = np.random.default_rng(25)
alpha = 0.1


# last experiment, comparing now with BART
def generate_data_new(
    n,
    cond_exp,
    noise_sd_fn,
    rng,
):
    num_dense = round(0.85 * n)  # half of the data points go to dense regions
    num_low = round(0.15 * n)
    x_1 = rng.uniform(size=num_dense, low=-1, high=0)
    x_2 = rng.beta(6, 6, size=num_low)
    x = np.concatenate([x_1, x_2])
    noise_sd = noise_sd_fn(x)
    noise = rng.normal(scale=noise_sd, size=n)
    y = cond_exp(x) + noise
    return pd.DataFrame({"x": x, "y": y})


def cond_exp(x):
    return (x > -0) * 1  # Generally we only make function of first covariate


def noise_sd_fn(x):
    return 0.05 + np.sin(15 * x) ** 2 * (x > 0)


# Simulating samples
data_train = generate_data_new(2000, cond_exp, noise_sd_fn, rng)
data_calibration = generate_data_new(2000, cond_exp, noise_sd_fn, rng)
data_test = generate_data_new(2000, cond_exp, noise_sd_fn, rng)

X_test = data_test["x"].to_numpy().reshape(-1, 1)
y_test = data_test["y"].to_numpy()

X_calib = data_calibration["x"].to_numpy().reshape(-1, 1)
y_calib = data_calibration["y"].to_numpy()

# gridding
x_grid = np.linspace(-1, 1, 300).reshape(-1, 1)

# fitting base model
X_train = data_train["x"].to_numpy().reshape(-1, 1)
y_train = data_train["y"].to_numpy()

# fitting base model for density
model = MDN_model(
    input_shape=1,
    num_components=5,
    hidden_layers=[64, 128],
    dropout_rate=0.15,
    base_model_type="density",
)

model.fit(
    X_train,
    y_train,
    epochs=1000,
    lr=0.001,
    patience=30,
    batch_size=35,
    scale=True,
)

# HPD band
hpd_score = model.mixture_cdf_density(y_calib, X_calib)
n = hpd_score.shape[0]
# computing quantile for HPD split
hpd_quantile = np.quantile(
    hpd_score,
    np.ceil((n + 1) * (alpha)) / n,
)

# cutoffs for grid
t_cutoff_hpd = model.predict_cdf_cutoff(
    x_grid,
    cutoff=hpd_quantile,
    num_samples=1000,
)

# fitting BART to density by hand
# computing scores
dens_score = model.predict(X_calib, y_calib)

# splitting data
(
    X_calib_train,
    X_calib_test,
    scores_calib_train,
    scores_calib_test,
) = train_test_split(
    X_calib,
    dens_score.numpy(),
    test_size=0.3,
    random_state=45,
)

# fitting BART model
bart_epistemic = BART_model(
    m=50,
    type="gamma",
    var="heteroscedastic",
    n_cores=6,
    progressbar=True,
    alpha=0.9,
)

bart_epistemic.fit(
    X_calib_train,
    scores_calib_train,
    n_sample=1000,
    random_seed=750,
)

s_prime_calibration = bart_epistemic.predict_cdf(
    X_calib_test,
    y_test=scores_calib_test,
    random_seed=750,
)

# computing cutoff
t_cutoff = np.quantile(s_prime_calibration, alpha)

# computing cutoff across grid
t_inverse_bart = bart_epistemic.predict_cutoff(
    x_grid,
    t=t_cutoff,
    random_seed=125,
)

# predicting conditional densities
y_grid = np.linspace(-6, 6, 3000)
# using gridding to compute density for each x_grid
densities = model.predict_mixture_density(
    x_grid,
    torch.tensor(y_grid),
)

# Creating dictionaries for HPD and EPICSCORE
hpd_dict = {}
epicscore_dict = {}
for i, x_val in enumerate(x_grid.flatten()):
    # Select densities lower than t_cutoff_hpd for HPD
    hpd_dict[x_val] = y_grid[densities[i] >= t_cutoff_hpd[i]]
    # Select densities lower than t_inverse_test for EPICSCORE
    if len(y_grid[densities[i] >= t_inverse_bart[i]]) == 0:
        print(f"Empty array for x = {x_val}")
    epicscore_dict[x_val] = y_grid[densities[i] >= t_inverse_bart[i]]


# Plotting the prediction regions using subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Plotting the prediction regions for HPD
y_inf, y_sup = np.zeros(x_grid.shape[0]), np.zeros(x_grid.shape[0])

i = 0
for x_val, y_vals in hpd_dict.items():
    y_inf[i] = np.min(y_vals)
    y_sup[i] = np.max(y_vals)
    i += 1

# Adding scatter points for the test data
axes[0].fill_between(
    x_grid.flatten(),
    y_inf,
    y_sup,
    color="blue",
    alpha=0.3,
)

axes[0].scatter(X_test, y_test, color="black", alpha=0.5, s=10)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Prediction Regions: HPD")

# Plotting the prediction regions for EPICSCORE
y_inf, y_sup = np.zeros(x_grid.shape[0]), np.zeros(x_grid.shape[0])

i = 0
for x_val, y_vals in epicscore_dict.items():
    if len(y_vals) == 0:
        print(x_val)
    y_inf[i] = np.min(y_vals)
    y_sup[i] = np.max(y_vals)
    i += 1

axes[1].fill_between(
    x_grid.flatten(),
    y_inf,
    y_sup,
    color="red",
    alpha=0.3,
)
axes[1].scatter(X_test, y_test, color="black", alpha=0.5, s=10)
axes[1].set_xlabel("x")
axes[1].set_title("Prediction Regions: EPICSCORE")

plt.tight_layout()
plt.show()

# computing quantile for HPD split
hpd_quantile = np.quantile(
    hpd_score,
    np.ceil((n + 1) * (alpha)) / n,
)

# cutoffs for grid
t_cutoff_hpd = model.predict_cdf_cutoff(
    x_grid,
    cutoff=hpd_quantile,
    num_samples=1000,
)

# fitting BART to density by hand
# computing scores
dens_score = model.predict(X_calib, y_calib)

# splitting data
(
    X_calib_train,
    X_calib_test,
    scores_calib_train,
    scores_calib_test,
) = train_test_split(
    X_calib,
    dens_score.numpy(),
    test_size=0.45,
    random_state=45,
)

# fitting BART model
bart_epistemic = BART_model(
    m=50,
    type="gamma",
    var="heteroscedastic",
    n_cores=6,
    progressbar=True,
    alpha=0.9,
)

bart_epistemic.fit(
    X_calib_train,
    scores_calib_train,
    n_sample=1000,
    random_seed=750,
)

s_prime_calibration = bart_epistemic.predict_cdf(
    X_calib_test,
    y_test=scores_calib_test,
    random_seed=750,
)

# computing cutoff
t_cutoff = np.quantile(s_prime_calibration, alpha)

# computing cutoff across grid
t_inverse_bart = bart_epistemic.predict_cutoff(
    x_grid,
    t=t_cutoff,
    random_seed=125,
)

# predicting conditional densities
y_grid = np.linspace(-6, 6, 3000)
# using gridding to compute density for each x_grid
densities = model.predict_mixture_density(
    x_grid,
    torch.tensor(y_grid),
)

# Creating dictionaries for HPD and EPICSCORE
hpd_dict = {}
epicscore_dict = {}
for i, x_val in enumerate(x_grid.flatten()):
    # Select densities lower than t_cutoff_hpd for HPD
    hpd_dict[x_val] = y_grid[densities[i] >= t_cutoff_hpd[i]]

    # Select densities lower than t_inverse_test for EPICSCORE
    if len(y_grid[densities[i] >= t_inverse_bart[i]]) == 0:
        print(f"Empty array for x = {x_val}")
    epicscore_dict[x_val] = y_grid[densities[i] >= t_inverse_bart[i]]


# Plotting the prediction regions using subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Plotting the prediction regions for HPD
y_inf, y_sup = np.zeros(x_grid.shape[0]), np.zeros(x_grid.shape[0])

i = 0
for x_val, y_vals in hpd_dict.items():
    y_inf[i] = np.min(y_vals)
    y_sup[i] = np.max(y_vals)
    i += 1

# Adding scatter points for the test data
axes[0].fill_between(
    x_grid.flatten(),
    y_inf,
    y_sup,
    color="blue",
    alpha=0.3,
)

axes[0].scatter(X_test, y_test, color="black", alpha=0.5, s=10)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Prediction Regions: HPD")

# Plotting the prediction regions for EPICSCORE
y_inf, y_sup = np.zeros(x_grid.shape[0]), np.zeros(x_grid.shape[0])

i = 0
for x_val, y_vals in epicscore_dict.items():
    if len(y_vals) == 0:
        print(x_val)
    y_inf[i] = np.min(y_vals)
    y_sup[i] = np.max(y_vals)
    i += 1

axes[1].fill_between(
    x_grid.flatten(),
    y_inf,
    y_sup,
    color="red",
    alpha=0.3,
)
axes[1].scatter(X_test, y_test, color="black", alpha=0.5, s=10)
axes[1].set_xlabel("x")
axes[1].set_title("Prediction Regions: EPICSCORE")

plt.tight_layout()
plt.show()
