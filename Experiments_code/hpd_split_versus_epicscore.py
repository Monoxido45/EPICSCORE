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


def generate_data_2(n, rng):
    x = np.cos(rng.beta(1.2, 0.8, n))

    # True function to generate y with higher variability at the center of x
    def true_function(x):
        mean_y = np.sin(x ** (-3))
        std_y = x**4
        y = rng.normal(mean_y, std_y)  # Generate y from a normal distribution
        return y

    y = true_function(x)
    return pd.DataFrame({"x": x, "y": y})


# implementing the model
model = MDN_model(
    input_shape=1,
    num_components=5,
    hidden_layers=[120],
    dropout_rate=0.5,
    normalize_y=True,
    base_model_type="density",
)

alpha = 0.1
# considering 500 samples first
# Simulating samples
data_train = generate_data_2(100, rng)
data_calibration = generate_data_2(100, rng)
data_test = generate_data_2(100, rng)

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

###############################################################################
# repeating for third simulation
torch.manual_seed(45)
torch.cuda.manual_seed(45)
rng = np.random.default_rng(25)


def generate_data_new(
    n,
    cond_exp,
    noise_sd_fn,
    rng,
):
    num_dense = round(0.3 * n)  # half of the data points go to dense regions
    num_low = round(0.7 * n)

    x_1 = rng.uniform(size=num_dense, low=-1, high=0)
    x_2 = rng.uniform(0, 1, num_low)

    x = np.concatenate([x_1, x_2])

    noise_sd = noise_sd_fn(x)
    noise = rng.normal(scale=noise_sd, size=n)
    y = cond_exp(x) + noise
    return pd.DataFrame({"x": x, "y": y})


def cond_exp(x):
    return (x > -0) * 1  # Generally we only make function of first covariate


def noise_sd_fn(x):
    return 0.01 + np.sin(15 * x) ** 2 * (x > 0)


model = MDN_model(
    input_shape=1,
    num_components=2,
    hidden_layers=[128, 64, 32],
    dropout_rate=0.5,
    normalize_y=True,
    base_model_type="density",
)

alpha = 0.1
# considering 500 samples first
# Simulating samples
data_train = generate_data_new(150, cond_exp, noise_sd_fn, rng)
data_calibration = generate_data_new(150, cond_exp, noise_sd_fn, rng)
data_test = generate_data_new(150, cond_exp, noise_sd_fn, rng)

X_train = data_train["x"].to_numpy().reshape(-1, 1)
y_train = data_train["y"].to_numpy()

X_test = data_test["x"].to_numpy().reshape(-1, 1)
y_test = data_test["y"].to_numpy()

X_calib = data_calibration["x"].to_numpy().reshape(-1, 1)
y_calib = data_calibration["y"].to_numpy()

x_grid = np.linspace(
    data_train["x"].min() - 0.05,
    data_train["x"].max() + 0.05,
    750,
).reshape(-1, 1)

# fitting model to training data
model.fit(
    X_train,
    y_train,
    epochs=1000,
    lr=0.001,
    patience=25,
    batch_size=15,
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

###############################################################################
# repeating for fourth simulation
torch.manual_seed(0)
torch.cuda.manual_seed(0)
rng = np.random.default_rng(0)


# function to generate synthetic data
def generate_data_4(
    n,
    rng,
    loc_1=2,
    loc_2=6.5,
    scale_1=0.5,
    scale_2=1,
):
    X1 = rng.normal(loc=loc_1, scale=scale_1, size=n // 2)
    X2 = rng.normal(loc=loc_2, scale=scale_2, size=n // 2)
    X = np.concatenate([X1, X2])
    rng.shuffle(X)
    # Define the true relationship between X and Y
    Y_true = np.sin(X) + 0.1 * X

    # Add heteroscedastic noise (aleatoric uncertainty)
    noise = rng.normal(0, 0.2 * np.abs(np.sin(X)), size=n)
    Y = Y_true + noise

    return pd.DataFrame({"x": X, "y": Y})


model = MDN_model(
    input_shape=1,
    num_components=5,
    hidden_layers=[128, 64, 32],
    dropout_rate=0.5,
    normalize_y=True,
    base_model_type="density",
)

alpha = 0.1
# considering 500 samples first
# Simulating samples
data_train = generate_data_4(200, rng)
data_calibration = generate_data_4(200, rng)
data_test = generate_data_4(200, rng)

X_train = data_train["x"].to_numpy().reshape(-1, 1)
y_train = data_train["y"].to_numpy()

X_test = data_test["x"].to_numpy().reshape(-1, 1)
y_test = data_test["y"].to_numpy()

X_calib = data_calibration["x"].to_numpy().reshape(-1, 1)
y_calib = data_calibration["y"].to_numpy()

x_grid = np.linspace(0, 10, 750).reshape(-1, 1)

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
