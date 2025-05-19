# testing coverage and interval length given outlier detection
from Epistemic_CP.epistemic_cp import ECP_split
from Epistemic_CP.scores import QuantileScore
from Epistemic_CP.utils import (
    average_coverage,
    average_interval_score_loss,
    compute_interval_length,
)

# base packages
import pandas as pd
import numpy as np
import os
import gc

# importing torch functions
import torch
import pickle

# importing scipy stats
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split

# importing preprocessing functions
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from tqdm import tqdm


original_path = os.getcwd()

# folder path
folder_path = "/Experiments_code"
os.chdir(original_path + folder_path)

from uacqr import uacqr

# returning to original path
os.chdir(original_path)

# fixing random generator and torch seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
CHECKPOINT_INTERVAL = 10

data_path = original_path + folder_path + "/pickle_files/{}_data".format("airfoil")


# different seeds generated for splitting data
def generate_seeds(seed_initial, n_rep):
    np.random.seed(seed_initial)
    seeds = np.random.randint(0, 2**31 - 1, size=n_rep)
    return seeds


# function to fit ECP objects
def adjust_ecp_obj_with_methods(
    ecp_obj,
    X_train,
    y_train,
    X_calib,
    y_calib,
    X_test,
    mdn_params,
    gp_params,
    bart_params,
    ensemble,
):
    # fitting ecp_obj - MC dropout
    ecp_obj.fit(X_train, y_train)
    ecp_obj.calib(
        X_calib,
        y_calib,
        num_components=mdn_params["num_components"],
        dropout_rate=mdn_params["dropout_rate"],
        epistemic_model="MC_dropout",
        hidden_layers=mdn_params["hidden_layers"],
        patience=mdn_params["patience"],
        epochs=mdn_params["epochs"],
        normalize_y=mdn_params["normalize_y"],
        scale=mdn_params["scale"],
        batch_size=mdn_params["batch_size"],
        verbose=mdn_params["verbose"],
        type=mdn_params["type"],
        ensemble=ensemble,
    )
    pred_ecp_mdn_test = ecp_obj.predict(X_test)
    # fitting ecp_obj - GP
    ecp_obj.calib(
        X_calib,
        y_calib,
        epistemic_model="GP_variational",
        scale=gp_params["scale"],
        normalize_y=gp_params["normalize_y"],
        num_inducing_points=gp_params["num_inducing_points"],
        n_epoch=gp_params["n_epoch"],
        batch_size=gp_params["batch_size"],
        verbose=gp_params["verbose"],
        patience=gp_params["patience"],
        ensemble=ensemble,
    )
    pred_ecp_gp_test = ecp_obj.predict(X_test)
    # fitting ecp_obj - BART
    ecp_obj.fit(X_train, y_train)
    ecp_obj.calib(
        X_calib,
        y_calib,
        epistemic_model="BART",
        m=bart_params["m"],
        var=bart_params["var"],
        normalize_y=bart_params["normalize_y"],
        type=bart_params["type"],
        ensemble=ensemble,
    )
    pred_ecp_bart_test = ecp_obj.predict(X_test)
    # deletting objects and removing from memory
    del ecp_obj
    gc.collect()
    return pred_ecp_mdn_test, pred_ecp_gp_test, pred_ecp_bart_test


def fit_and_return_pred_intervals(
    data,
    base_params,
    uacqr_params,
    mdn_params,
    gp_params,
    bart_params,
    alpha=0.1,
    random_seed=45,
    prop_test=0.2,
    prop_train=0.5,
):
    X = data.drop(columns=["target"])
    y = data["target"]
    X_train_calib, X_test, y_train_calib, y_test = train_test_split(
        X,
        y,
        test_size=prop_test,
        random_state=random_seed,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_calib,
        y_train_calib,
        test_size=prop_train,
        random_state=random_seed,
    )
    # fitting base estimator and UACQR
    uacqr_results = uacqr(
        base_params,
        q_lower=alpha / 2 * 100,
        q_upper=(1 - alpha / 2) * 100,
        model_type=uacqr_params["model_type"],
        B=uacqr_params["B"],
        random_state=random_seed,
        uacqrs_agg=uacqr_params["uacqrs_agg"],
    )
    uacqr_results.fit(X_train, y_train)
    uacqr_results.calibrate(X_calib, y_calib)
    uacqr_pred_test = uacqr_results.predict_uacqr(X_test)
    # Preparing Calibration and Test Data for ECP
    X_calib = X_calib.to_numpy()
    y_calib = y_calib.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    # fitting the different ECP methods
    ecp_obj = ECP_split(
        QuantileScore,
        uacqr_results,
        alpha=alpha,
        is_fitted=True,
        base_model_type=uacqr_params["base_model_type"],
    )
    pred_ecp_mdn_test, pred_ecp_gp_test, pred_ecp_bart_test = (
        adjust_ecp_obj_with_methods(
            ecp_obj,
            X_train,
            y_train,
            X_calib,
            y_calib,
            X_test,
            mdn_params,
            gp_params,
            bart_params,
            ensemble=False,
        )
    )
    return (
        pred_ecp_mdn_test,
        pred_ecp_gp_test,
        pred_ecp_bart_test,
        uacqr_pred_test,
        X_test,
        y_test,
    )


def compare_outlier_inlier(
    model_str,
    data,
    base_params,
    uacqr_params,
    mdn_params,
    gp_params,
    bart_params,
    n_rep=50,
    alpha=0.1,
    prop_test=0.2,
    prop_train=0.5,
    inlier_size=0.2,
    contamination=0.05,
    n_neighbors=15,
    n_components=2,
    tsne_random_state=120,
    seed_initial=145,
    data_name="default",
):
    # setting several random seeds
    all_results = []
    seeds = generate_seeds(seed_initial, n_rep)
    start_iteration = 0

    # creating a folder for the data
    data_path = original_path + folder_path + "/pickle_files/{}_data".format(data_name)

    # creating directories to each file
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    # Check if a checkpoint exists
    checkpoint_filename = f"checkpoint_{data_name}_{model_str}"
    checkpoints = [
        f
        for f in os.listdir(data_path)
        if f.startswith(checkpoint_filename) and f.endswith(".pkl")
    ]

    # setting dir as current directory
    os.chdir(data_path)

    if checkpoints:
        # Get the latest checkpoint based on the modification timestamp
        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(x))

        with open(latest_checkpoint, "rb") as f:
            checkpoint_data = pickle.load(f)

        all_results = checkpoint_data["all_results"]
        # seeds = checkpoint_data['seeds']
        start_iteration = checkpoint_data["iteration"]
        print(f"Loaded checkpoint from iteration {start_iteration}")
    else:
        print("No checkpoint found.")

    for i, seed in enumerate(seeds[start_iteration:], start=start_iteration):
        (
            pred_ecp_mdn_test,
            pred_ecp_gp_test,
            pred_ecp_bart_test,
            uacqr_pred_test,
            X_test,
            y_test,
        ) = fit_and_return_pred_intervals(
            data=data,
            base_params=base_params,
            uacqr_params=uacqr_params,
            mdn_params=mdn_params,
            gp_params=gp_params,
            bart_params=bart_params,
            alpha=alpha,
            random_seed=seed,
            prop_test=prop_test,
            prop_train=prop_train,
        )
        # detection outliers in X_test
        # using TSNE for dimensionality reduction
        tsne = TSNE(n_components=n_components, random_state=tsne_random_state)
        X_tsne_test = tsne.fit_transform(X_test)

        # Standardize the features
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_tsne_test)
        # Use Local Outlier Factor for anomaly detection on scaled data
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
        )
        out_pred = lof.fit_predict(X_test_scaled)

        # selecting outliers
        outlier_obs = y_test[out_pred == -1]
        outlier_indexes = np.where(out_pred == -1)[0]
        # selecting 15% top inliers
        inlier_indexes = np.setdiff1d(np.arange(len(y_test)), outlier_indexes)
        inlier_scores = lof.negative_outlier_factor_[inlier_indexes]
        # computing inlier scores
        size = int((y_test.shape[0] - outlier_obs.shape[0]) * inlier_size)
        most_inlier_idxs = np.argsort(inlier_scores)[::-1][:size]
        # outlier labels
        most_cont_labels = y_test[outlier_indexes]

        # Select intervals corresponding to outlier indexes
        outlier_intervals_mdn = pred_ecp_mdn_test[outlier_indexes]
        outlier_intervals_gp = pred_ecp_gp_test[outlier_indexes]
        outlier_intervals_bart = pred_ecp_bart_test[outlier_indexes]
        outlier_intervals_uacqrp = np.column_stack(
            (
                uacqr_pred_test["UACQR-P"]["lower"][outlier_indexes],
                uacqr_pred_test["UACQR-P"]["upper"][outlier_indexes],
            )
        )
        outlier_intervals_uacqrs = np.column_stack(
            (
                uacqr_pred_test["UACQR-S"]["lower"][outlier_indexes],
                uacqr_pred_test["UACQR-S"]["upper"][outlier_indexes],
            )
        )
        outlier_intervals_cqr = np.column_stack(
            (
                uacqr_pred_test["CQR"]["lower"][outlier_indexes],
                uacqr_pred_test["CQR"]["upper"][outlier_indexes],
            )
        )
        outlier_intervals_cqrr = np.column_stack(
            (
                uacqr_pred_test["CQR-r"]["lower"][outlier_indexes],
                uacqr_pred_test["CQR-r"]["upper"][outlier_indexes],
            )
        )
        inlier_intervals_mdn = pred_ecp_mdn_test[most_inlier_idxs]
        inlier_intervals_gp = pred_ecp_gp_test[most_inlier_idxs]
        inlier_intervals_bart = pred_ecp_bart_test[most_inlier_idxs]
        inlier_intervals_uacqrp = np.column_stack(
            (
                uacqr_pred_test["UACQR-P"]["lower"][most_inlier_idxs],
                uacqr_pred_test["UACQR-P"]["upper"][most_inlier_idxs],
            )
        )
        inlier_intervals_uacqrs = np.column_stack(
            (
                uacqr_pred_test["UACQR-S"]["lower"][most_inlier_idxs],
                uacqr_pred_test["UACQR-S"]["upper"][most_inlier_idxs],
            )
        )
        inlier_intervals_cqr = np.column_stack(
            (
                uacqr_pred_test["CQR"]["lower"][most_inlier_idxs],
                uacqr_pred_test["CQR"]["upper"][most_inlier_idxs],
            )
        )
        inlier_intervals_cqrr = np.column_stack(
            (
                uacqr_pred_test["CQR-r"]["lower"][most_inlier_idxs],
                uacqr_pred_test["CQR-r"]["upper"][most_inlier_idxs],
            )
        )

        # Calculating ratio between lengths for each method
        mdn_ratio = np.mean(
            compute_interval_length(
                outlier_intervals_mdn[:, 1], outlier_intervals_mdn[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                inlier_intervals_mdn[:, 1], inlier_intervals_mdn[:, 0]
            )
        )
        gp_ratio = np.mean(
            compute_interval_length(
                outlier_intervals_gp[:, 1], outlier_intervals_gp[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                inlier_intervals_gp[:, 1], inlier_intervals_gp[:, 0]
            )
        )
        bart_ratio = np.mean(
            compute_interval_length(
                outlier_intervals_bart[:, 1], outlier_intervals_bart[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                inlier_intervals_bart[:, 1], inlier_intervals_bart[:, 0]
            )
        )
        uacqrp_ratio = np.mean(
            compute_interval_length(
                outlier_intervals_uacqrp[:, 1], outlier_intervals_uacqrp[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                inlier_intervals_uacqrp[:, 1], inlier_intervals_uacqrp[:, 0]
            )
        )
        uacqrs_ratio = np.mean(
            compute_interval_length(
                outlier_intervals_uacqrs[:, 1], outlier_intervals_uacqrs[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                inlier_intervals_uacqrs[:, 1], inlier_intervals_uacqrs[:, 0]
            )
        )
        cqr_ratio = np.mean(
            compute_interval_length(
                outlier_intervals_cqr[:, 1], outlier_intervals_cqr[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                inlier_intervals_cqr[:, 1], inlier_intervals_cqr[:, 0]
            )
        )
        cqrr_ratio = np.mean(
            compute_interval_length(
                outlier_intervals_cqrr[:, 1], outlier_intervals_cqrr[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                inlier_intervals_cqrr[:, 1], inlier_intervals_cqrr[:, 0]
            )
        )

        # Calculating average coverage for outliers
        avg_coverage_mdn = average_coverage(
            outlier_intervals_mdn[:, 1], outlier_intervals_mdn[:, 0], outlier_obs
        )
        avg_coverage_gp = average_coverage(
            outlier_intervals_gp[:, 1], outlier_intervals_gp[:, 0], outlier_obs
        )
        avg_coverage_bart = average_coverage(
            outlier_intervals_bart[:, 1], outlier_intervals_bart[:, 0], outlier_obs
        )
        avg_coverage_uacqrp = average_coverage(
            outlier_intervals_uacqrp[:, 1], outlier_intervals_uacqrp[:, 0], outlier_obs
        )
        avg_coverage_uacqrs = average_coverage(
            outlier_intervals_uacqrs[:, 1], outlier_intervals_uacqrs[:, 0], outlier_obs
        )
        avg_coverage_cqr = average_coverage(
            outlier_intervals_cqr[:, 1], outlier_intervals_cqr[:, 0], outlier_obs
        )
        avg_coverage_cqrr = average_coverage(
            outlier_intervals_cqrr[:, 1], outlier_intervals_cqrr[:, 0], outlier_obs
        )

        # Calculating SMIS for outliers
        smis_mdn = average_interval_score_loss(
            outlier_intervals_mdn[:, 1], outlier_intervals_mdn[:, 0], outlier_obs, alpha
        )
        smis_gp = average_interval_score_loss(
            outlier_intervals_gp[:, 1], outlier_intervals_gp[:, 0], outlier_obs, alpha
        )
        smis_bart = average_interval_score_loss(
            outlier_intervals_bart[:, 1],
            outlier_intervals_bart[:, 0],
            outlier_obs,
            alpha,
        )
        smis_uacqrp = average_interval_score_loss(
            outlier_intervals_uacqrp[:, 1],
            outlier_intervals_uacqrp[:, 0],
            outlier_obs,
            alpha,
        )
        smis_uacqrs = average_interval_score_loss(
            outlier_intervals_uacqrs[:, 1],
            outlier_intervals_uacqrs[:, 0],
            outlier_obs,
            alpha,
        )
        smis_cqr = average_interval_score_loss(
            outlier_intervals_cqr[:, 1], outlier_intervals_cqr[:, 0], outlier_obs, alpha
        )
        smis_cqrr = average_interval_score_loss(
            outlier_intervals_cqrr[:, 1],
            outlier_intervals_cqrr[:, 0],
            outlier_obs,
            alpha,
        )

        # creating metric dataframe
        metric_result = pd.DataFrame(
            {
                "Method": [
                    "ECP-MDN",
                    "ECP-GP",
                    "ECP-BART",
                    "UACQR-P",
                    "UACQR-S",
                    "CQR-r",
                    "CQR",
                ],
                "Interval Length Ratio": [
                    mdn_ratio,
                    gp_ratio,
                    bart_ratio,
                    uacqrp_ratio,
                    uacqrs_ratio,
                    cqrr_ratio,
                    cqr_ratio,
                ],
                "Coverage outlier": [
                    avg_coverage_mdn,
                    avg_coverage_gp,
                    avg_coverage_bart,
                    avg_coverage_uacqrp,
                    avg_coverage_uacqrs,
                    avg_coverage_cqrr,
                    avg_coverage_cqr,
                ],
                "SMIS outlier": [
                    smis_mdn,
                    smis_gp,
                    smis_bart,
                    smis_uacqrp,
                    smis_uacqrs,
                    smis_cqrr,
                    smis_cqr,
                ],
            }
        )
        # appending the results
        all_results.append(metric_result)

        # saving checkpoint each CHECKPOINT_INTERVAL iterations
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                "iteration": i + 1,
                "all_results": all_results,
                #'seeds': seeds
            }
            with open(f"checkpoint_{data_name}_{model_str}.pkl", "wb") as f:
                pickle.dump(checkpoint, f)
            print(f"Checkpoint saved in iteration {i+1}")

    # computing mean and standard deviation for each methods
    final_results = pd.concat(all_results)
    summary = (
        final_results.groupby("Method")
        .agg(
            {
                "Interval Length Ratio": ["mean", "std"],
                "Coverage outlier": ["mean", "std"],
                "SMIS outlier": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # removing all checkpoints
    for f in os.listdir("."):
        if f.startswith("checkpoint_") and f.endswith(".pkl"):
            os.remove(f)

    return all_results, summary


catboost_params = {
    "iterations": 1000,
    "learning_rate": 1e-3,
    "depth": 6,  # default value
    "l2_leaf_reg": 3,  # default value
    "random_strength": 1,  # default value
    "bagging_temperature": 1,  # default value
    "od_type": "Iter",
    "od_wait": 50,
    "use_best_model": False,
}

# adding quantile neural network parameters
nnet_params = {
    "dropout": 0.05,
    "epochs": 200,
    "hidden_size": 200,
    "lr": 1e-3,
    "batch_size": 32,
    "normalize": True,
    "weight_decay": 1e-7,
    "epoch_model_tracking": True,
    "drop_last": True,
    "undo_quantile_crossing": True,
}

# uacqr_params for catboost
uacqr_params = {
    "model_type": "catboost",
    "B": 999,
    "uacqrs_agg": "std",
    "base_model_type": "Quantile",
}

# uacqr_params for neural network
uacqr_params_net = {
    "model_type": "neural_net",
    "B": nnet_params["epochs"] - 1,
    "uacqrs_agg": "std",
    "base_model_type": "Quantile",
}

mdn_params = {
    "num_components": 3,
    "dropout_rate": 0.5,
    "epistemic_model": "MC_dropout",
    "hidden_layers": [64, 64],
    "patience": 50,
    "epochs": 2000,
    "scale": True,
    "batch_size": 40,
    "normalize_y": True,
    "verbose": 0,
    "type": "gaussian",
}

gp_params = {
    "epistemic_model": "GP_variational",
    "scale": True,
    "normalize_y": True,
    "num_inducing_points": 15,
    "n_epoch": 2000,
    "batch_size": 40,
    "verbose": 0,
    "patience": 50,
}

bart_params = {
    "epistemic_model": "BART_heteroscedastic",
    "m": 100,
    "var": "heteroscedastic",
    "normalize_y": True,
    "type": "normal",
}

alpha = 0.1
if __name__ == "__main__":
    print("We will now compute all conformal statistics for real data")
    model = input("Which model would you like to fit as base model? ")

    data_name = input(
        "Which dataset would you like to use (e.g., 'bike' or 'winewhite')? "
    )
    metrics_filename = (
        input("Enter the filename to save metrics (e.g., 'metrics_bike'): ")
        + "_"
        + model
        + "_outliers.csv"
    )
    it = int(input("How many iterations? "))

    if model == "catboost":
        print("Starting experiment with CatBoost as base model")
        base_params = catboost_params
        uacqr_params = uacqr_params
    elif model == "nnet":
        print("Starting experiment with Neural Network as base model")
        base_params = nnet_params
        uacqr_params = uacqr_params_net
    else:
        print("Invalid model. Please choose either 'catboost' or 'nnet'.")

    # Function to check if the user wants to stop
    def check_for_termination():
        response = input("Do you want to stop the process? (yes/no): ").strip().lower()
        if response == "yes":
            print("Process terminated by user.")
            return True
        return False

    # Load data for the specified dataset
    data = pd.read_csv(original_path + f"/data/processed/{data_name}.csv")

    if data.shape[0] > 10000:
        mdn_params["batch_size"] = 125
        gp_params["batch_size"] = 125
        gp_params["num_inducing_points"] = 50

    if data_name == "WEC":
        mdn_params["batch_size"] = 250
        gp_params["batch_size"] = 250

    # Compute metrics for the specified dataset
    all_results, metrics = compare_outlier_inlier(
        model_str=model,
        data=data,
        base_params=base_params,
        mdn_params=mdn_params,
        gp_params=gp_params,
        bart_params=bart_params,
        uacqr_params=uacqr_params,
        alpha=alpha,
        seed_initial=45,
        n_rep=it,
        prop_train=0.5,
        prop_test=0.2,
        data_name=data_name,
    )

    # Save metrics
    print(metrics)
    metrics.to_csv(metrics_filename, index=False)

    # save all results
    with open(f"all_metrics_{data_name}_{model}_outliers.pkl", "wb") as f:
        pickle.dump(all_results, f)
