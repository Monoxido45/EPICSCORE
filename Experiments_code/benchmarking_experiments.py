import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from Epistemic_CP.epistemic_models import MDN_model, BART_model, GPApprox_model

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import time


# different seeds generated for splitting data
def generate_seeds(seed_initial, n_rep):
    np.random.seed(seed_initial)
    seeds = np.random.randint(0, 2**31 - 1, size=n_rep)
    return seeds


def benchmark_function(
    n_it=np.array([1000, 2000, 5000, 10000, 20000, 50000]),
    n_rep=10,
    n_features=30,
    central_seed=45,
    noise=1,
    bias=0.5,
    n_informative=5,
    n_targets=1,
):
    seeds = generate_seeds(
        seed_initial=central_seed,
        n_rep=n_rep,
    )

    print(seeds)

    mdn_time = np.zeros((n_it.shape[0], 2))
    gp_time = np.zeros((n_it.shape[0], 2))
    bart_time = np.zeros((n_it.shape[0], 2))

    j = 0
    for n_sample in tqdm(n_it, desc="Computing running times for each experiment:"):
        bart_times, mdn_times, gp_times = [], [], []
        for i, seed in enumerate(seeds):
            X_train, y_train = make_regression(
                n_samples=n_sample,
                n_features=n_features,
                noise=noise,
                random_state=seed,
                bias=bias,
                n_informative=n_informative,
                n_targets=n_targets,
            )

            X_calib, y_calib = make_regression(
                n_samples=n_sample,
                n_features=n_features,
                noise=noise,
                random_state=seed + 15,
                bias=bias,
                n_informative=n_informative,
                n_targets=n_targets,
            )

            # Fit a KNN model
            knn_model = KNeighborsRegressor(n_neighbors=30)
            knn_model.fit(X_train, y_train)

            # Predict on calibration data
            y_calib_pred = knn_model.predict(X_calib)
            scores_calib = np.abs(y_calib - y_calib_pred)

            # fitting models
            if n_sample <= 5000:
                batch_size = 35
                n_inducing_points = 15
            elif 5000 < n_sample <= 20000:
                batch_size = 126
                n_inducing_points = 30
            else:
                batch_size = 250
                n_inducing_points = 65

            # fitting mdn and saving time
            start_time = time.time()
            mdn_model = MDN_model(
                input_shape=n_features,
                num_components=3,
                hidden_layers=[64, 64],
                dropout_rate=0.5,
                base_model_type="density",
                normalize_y=True,
            )
            mdn_model.fit(
                X_calib,
                scores_calib,
                epochs=2000,
                lr=0.001,
                patience=50,
                scale=True,
                batch_size=batch_size,
                verbose=0,
            )
            end_time = time.time()
            mdn_times.append(end_time - start_time)

            # fitting bart and saving time
            start_time = time.time()
            bart_epistemic = BART_model(
                m=50,
                type="normal",
                var="heteroscedastic",
                n_cores=6,
                progressbar=False,
                normalize_y=True,
            )

            bart_epistemic.fit(
                X_calib,
                scores_calib,
                n_sample=500,
                random_seed=750,
            )
            end_time = time.time()
            bart_times.append(end_time - start_time)

            # fitting gp and saving time
            start_time = time.time()
            gp_epistemic = GPApprox_model(
                num_inducing_points=n_inducing_points,
                lr_variational=0.1,
                lr_hyperparams=0.01,
                n_epochs=2000,
            )

            # Training the model with calibration data
            gp_epistemic.fit(
                X_calib,
                scores_calib,
                batch_size=batch_size,
                random_seed_fit=750,
                random_seed_split=150,
                patience=50,
                proportion_train=0.7,
                verbose=0,
            )
            end_time = time.time()
            gp_times.append(end_time - start_time)

        # Compute average and standard error for each model
        mdn_time_mean = np.mean(mdn_times)
        mdn_time_stderr = 0  # np.std(mdn_times) / np.sqrt(len(mdn_times))

        bart_time_mean = np.mean(bart_times)
        bart_time_stderr = 0  # np.std(bart_times) / np.sqrt(len(bart_times))

        gp_time_mean = np.mean(gp_times)
        gp_time_stderr = 0  # np.std(gp_times) / np.sqrt(len(gp_times))

        # Store results
        mdn_time[j] = np.array([mdn_time_mean, mdn_time_stderr])
        bart_time[j] = np.array([bart_time_mean, bart_time_stderr])
        gp_time[j] = np.array([gp_time_mean, gp_time_stderr])

        j += 1

    # returning the results
    return mdn_time, bart_time, gp_time


mdn_time, bart_time, gp_time = benchmark_function(n_rep=1)
