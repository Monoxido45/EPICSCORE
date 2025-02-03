# importing the functions from our package
from Epistemic_CP.epistemic_cp import ECP_split
from Epistemic_CP.scores import QuantileScore
from Epistemic_CP.utils import (
    average_coverage,
    average_interval_score_loss,
    compute_interval_length,
    corr_coverage_widths,
)

# base packages
import pandas as pd
import numpy as np
import os
import pickle
import gc

# importing torch functions
import torch

# importing scipy stats
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split

# UACQR function used from https://github.com/rrross/UACQR.git
from uacqr import uacqr

# from experiment import experiment
# original path
original_path = os.getcwd()

# folder path
folder_path = "/Experiments_code"

# fixing random generator and torch seeds
rng = np.random.default_rng(15)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# different seeds generated for splitting data
def generate_seeds(seed_initial, n_rep):
    np.random.seed(seed_initial)
    seeds = np.random.randint(0, 2**31 - 1, size=n_rep)
    return seeds


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
    )
    pred_ecp_bart_test = ecp_obj.predict(X_test)
    # deletting objects and removing from memory
    del ecp_obj
    gc.collect()
    return pred_ecp_mdn_test, pred_ecp_gp_test, pred_ecp_bart_test


# computing metrics all methods
def obtain_metrics_all_methods(
    data,
    target_column,
    base_params,
    mdn_params,
    gp_params,
    bart_params,
    uacqr_params,
    rng,
    data_name="default",
    alpha=0.05,
    seed_initial=45,
    n_it=50,
    prop_train=0.5,
    prop_test=0.2,
):

    all_results = []
    seeds = generate_seeds(seed_initial, n_it)
    start_iteration = 0

    # creating a folder for the data
    data_path = original_path + folder_path + "/pickle_files/{}_data".format(data_name)

    # creating directories to each file
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    # Check if a checkpoint exists
    checkpoint_filename = f"checkpoint_{data_name}"
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
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train_calib, X_test, y_train_calib, y_test = train_test_split(
            X, y, test_size=prop_test, random_state=seed
        )
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train_calib, y_train_calib, test_size=prop_train, random_state=seed
        )

        # fitting base estimator and UACQR
        uacqr_results = uacqr(
            base_params,
            q_lower=alpha / 2 * 100,
            q_upper=(1 - alpha / 2) * 100,
            model_type=uacqr_params["model_type"],
            B=uacqr_params["B"],
            random_state=seed,
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
                rng=rng,
            )
        )

        # computing Average Coverage
        Average_coverage_mdn = average_coverage(
            pred_ecp_mdn_test[:, 1], pred_ecp_mdn_test[:, 0], y_test
        )
        Average_coverage_gp = average_coverage(
            pred_ecp_gp_test[:, 1], pred_ecp_gp_test[:, 0], y_test
        )
        Average_coverage_bart = average_coverage(
            pred_ecp_bart_test[:, 1], pred_ecp_bart_test[:, 0], y_test
        )
        Average_coverage_uacqrp = average_coverage(
            uacqr_pred_test["UACQR-P"]["upper"],
            uacqr_pred_test["UACQR-P"]["lower"],
            y_test,
        )
        Average_coverage_uacqrs = average_coverage(
            uacqr_pred_test["UACQR-S"]["upper"],
            uacqr_pred_test["UACQR-S"]["lower"],
            y_test,
        )
        Average_coverage_cqrr = average_coverage(
            uacqr_pred_test["CQR-r"]["upper"], uacqr_pred_test["CQR-r"]["lower"], y_test
        )
        Average_coverage_cqr = average_coverage(
            uacqr_pred_test["CQR"]["upper"], uacqr_pred_test["CQR"]["lower"], y_test
        )

        # computing ISL
        # AISL: Average Interval Score Loss
        ISL_mdn = average_interval_score_loss(
            pred_ecp_mdn_test[:, 1], pred_ecp_mdn_test[:, 0], y_test, alpha
        )
        ISL_gp = average_interval_score_loss(
            pred_ecp_gp_test[:, 1], pred_ecp_gp_test[:, 0], y_test, alpha
        )
        ISL_bart = average_interval_score_loss(
            pred_ecp_bart_test[:, 1], pred_ecp_bart_test[:, 0], y_test, alpha
        )
        ISL_uacqrp = average_interval_score_loss(
            uacqr_pred_test["UACQR-P"]["upper"],
            uacqr_pred_test["UACQR-P"]["lower"],
            y_test,
            alpha,
        )
        ISL_uacqrs = average_interval_score_loss(
            uacqr_pred_test["UACQR-S"]["upper"],
            uacqr_pred_test["UACQR-S"]["lower"],
            y_test,
            alpha,
        )
        ISL_cqrr = average_interval_score_loss(
            uacqr_pred_test["CQR-r"]["upper"],
            uacqr_pred_test["CQR-r"]["lower"],
            y_test,
            alpha,
        )
        ISL_cqr = average_interval_score_loss(
            uacqr_pred_test["CQR"]["upper"],
            uacqr_pred_test["CQR"]["lower"],
            y_test,
            alpha,
        )

        # for computing interval length and pearson correlation, selecting indexes where UACQR-P is not inf and -inf
        idxs = np.where(
            np.logical_or(
                uacqr_pred_test["UACQR-P"]["upper"] != np.inf,
                uacqr_pred_test["UACQR-P"]["lower"] != -np.inf,
            )
        )[0]

        # checking whether all are
        if idxs.size > 0:
            # Computing interval length (IL)
            IL_mdn = np.mean(
                compute_interval_length(
                    pred_ecp_mdn_test[idxs, 1], pred_ecp_mdn_test[idxs, 0]
                )
            )
            IL_gp = np.mean(
                compute_interval_length(
                    pred_ecp_gp_test[idxs, 1], pred_ecp_gp_test[idxs, 0]
                )
            )
            IL_bart = np.mean(
                compute_interval_length(
                    pred_ecp_bart_test[idxs, 1], pred_ecp_bart_test[idxs, 0]
                )
            )
            IL_uacqrp = np.mean(
                compute_interval_length(
                    uacqr_pred_test["UACQR-P"]["upper"][idxs],
                    uacqr_pred_test["UACQR-P"]["lower"][idxs],
                )
            )
            IL_uacqrs = np.mean(
                compute_interval_length(
                    uacqr_pred_test["UACQR-S"]["upper"][idxs],
                    uacqr_pred_test["UACQR-S"]["lower"][idxs],
                )
            )
            IL_cqrr = np.mean(
                compute_interval_length(
                    uacqr_pred_test["CQR-r"]["upper"][idxs],
                    uacqr_pred_test["CQR-r"]["lower"][idxs],
                )
            )
            IL_cqr = np.mean(
                compute_interval_length(
                    uacqr_pred_test["CQR"]["upper"][idxs],
                    uacqr_pred_test["CQR"]["lower"][idxs],
                )
            )

            pcorr_mdn = corr_coverage_widths(
                pred_ecp_mdn_test[idxs, 1], pred_ecp_mdn_test[idxs, 0], y_test[idxs]
            )
            pcorr_gp = corr_coverage_widths(
                pred_ecp_gp_test[idxs, 1], pred_ecp_gp_test[idxs, 0], y_test[idxs]
            )
            pcorr_bart = corr_coverage_widths(
                pred_ecp_bart_test[idxs, 1], pred_ecp_bart_test[idxs, 0], y_test[idxs]
            )
            pcorr_uacqrp = corr_coverage_widths(
                uacqr_pred_test["UACQR-P"]["upper"][idxs],
                uacqr_pred_test["UACQR-P"]["lower"][idxs],
                y_test[idxs],
            )
            pcorr_uacqrs = corr_coverage_widths(
                uacqr_pred_test["UACQR-S"]["upper"][idxs],
                uacqr_pred_test["UACQR-S"]["lower"][idxs],
                y_test[idxs],
            )
            pcorr_cqr = corr_coverage_widths(
                uacqr_pred_test["CQR"]["upper"][idxs],
                uacqr_pred_test["CQR"]["lower"][idxs],
                y_test[idxs],
            )
            pcorr_cqrr = corr_coverage_widths(
                uacqr_pred_test["CQR-r"]["upper"][idxs],
                uacqr_pred_test["CQR-r"]["lower"][idxs],
                y_test[idxs],
            )
        else:
            # Computing interval length (IL)
            IL_mdn = np.mean(
                compute_interval_length(
                    pred_ecp_mdn_test[:, 1], pred_ecp_mdn_test[:, 0]
                )
            )
            IL_gp = np.mean(
                compute_interval_length(pred_ecp_gp_test[:, 1], pred_ecp_gp_test[:, 0])
            )
            IL_bart = np.mean(
                compute_interval_length(
                    pred_ecp_bart_test[:, 1], pred_ecp_bart_test[:, 0]
                )
            )

            IL_uacqrp = np.nan

            IL_uacqrs = np.mean(
                compute_interval_length(
                    uacqr_pred_test["UACQR-S"]["upper"],
                    uacqr_pred_test["UACQR-S"]["lower"],
                )
            )
            IL_cqrr = np.mean(
                compute_interval_length(
                    uacqr_pred_test["CQR-r"]["upper"], uacqr_pred_test["CQR-r"]["lower"]
                )
            )
            IL_cqr = np.mean(
                compute_interval_length(
                    uacqr_pred_test["CQR"]["upper"], uacqr_pred_test["CQR"]["lower"]
                )
            )

            pcorr_mdn = corr_coverage_widths(
                pred_ecp_mdn_test[:, 1], pred_ecp_mdn_test[:, 0], y_test
            )
            pcorr_gp = corr_coverage_widths(
                pred_ecp_gp_test[:, 1], pred_ecp_gp_test[:, 0], y_test
            )
            pcorr_bart = corr_coverage_widths(
                pred_ecp_bart_test[:, 1], pred_ecp_bart_test[:, 0], y_test
            )

            pcorr_uacqrp = np.nan

            pcorr_uacqrs = corr_coverage_widths(
                uacqr_pred_test["UACQR-S"]["upper"],
                uacqr_pred_test["UACQR-S"]["lower"],
                y_test,
            )
            pcorr_cqr = corr_coverage_widths(
                uacqr_pred_test["CQR"]["upper"], uacqr_pred_test["CQR"]["lower"], y_test
            )
            pcorr_cqrr = corr_coverage_widths(
                uacqr_pred_test["CQR-r"]["upper"],
                uacqr_pred_test["CQR-r"]["lower"],
                y_test,
            )

        # creating metric dataframe
        metric_result = pd.DataFrame(
            {
                "Metodo": [
                    "ECP-MDN",
                    "ECP-GP",
                    "ECP-BART",
                    "UACQR-P",
                    "UACQR-S",
                    "CQR-r",
                    "CQR",
                ],
                "Average Coverage": [
                    Average_coverage_mdn,
                    Average_coverage_gp,
                    Average_coverage_bart,
                    Average_coverage_uacqrp,
                    Average_coverage_uacqrs,
                    Average_coverage_cqrr,
                    Average_coverage_cqr,
                ],
                "AISL": [
                    ISL_mdn,
                    ISL_gp,
                    ISL_bart,
                    ISL_uacqrp,
                    ISL_uacqrs,
                    ISL_cqrr,
                    ISL_cqr,
                ],
                "IL": [IL_mdn, IL_gp, IL_bart, IL_uacqrp, IL_uacqrs, IL_cqrr, IL_cqr],
                "pcorr": [
                    pcorr_mdn,
                    pcorr_gp,
                    pcorr_bart,
                    pcorr_uacqrp,
                    pcorr_uacqrs,
                    pcorr_cqrr,
                    pcorr_cqr,
                ],
            }
        )
        all_results.append(metric_result)

        # saving checkpoint each CHECKPOINT_INTERVAL iterations
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                "iteration": i + 1,
                "all_results": all_results,
                #'seeds': seeds
            }
            with open(f"checkpoint_{data_name}.pkl", "wb") as f:
                pickle.dump(checkpoint, f)
            print(f"Checkpoint salvo na iteração {i+1}")

    # computing mean and standard deviation for each methods
    final_results = pd.concat(all_results)
    summary = (
        final_results.groupby("Metodo")
        .agg(
            {
                "Average Coverage": ["mean", "std"],
                "AISL": ["mean", "std"],
                "IL": ["mean", "std"],
                "pcorr": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Remover arquivos de checkpoint
    for f in os.listdir("."):
        if f.startswith("checkpoint_") and f.endswith(".pkl"):
            os.remove(f)

    return all_results, summary


catboost_params = {
    "iterations": 1000,
    "learning_rate": 1e-3,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_strength": 1,
    "bagging_temperature": 1,
    "od_type": "Iter",
    "od_wait": 50,
    "use_best_model": False,
}

uacqr_params = {
    "model_type": "catboost",
    "B": 999,
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
n_it = 50
CHECKPOINT_INTERVAL = 10

if __name__ == "__main__":

    print("We will now compute all conformal statistics for real data")
    model = input("Which model would you like to fit as base model? ")
    data_name = input(
        "Which dataset would you like to use (e.g., 'bike' or 'winewhite')? "
    )
    metrics_filename = input(
        "Enter the filename to save metrics (e.g., 'metrics_bike.csv'): "
    )
    it = int(input("How many iterations? "))

    uacqr_params["base_model_type"] = model
    print("Starting real data experiment")

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

    if data_name == "blog":
        mdn_params["patience"] = 30
        gp_params["patience"] = 30
        mdn_params["hidden_layers"] = [125]
        mdn_params["batch_size"] = 150
        gp_params["batch_size"] = 150

    if data_name == "WEC":
        mdn_params["batch_size"] = 250
        gp_params["batch_size"] = 250

    # Compute metrics for the specified dataset
    all_results, metrics = obtain_metrics_all_methods(
        data,
        "target",
        catboost_params,
        mdn_params,
        gp_params,
        bart_params,
        uacqr_params,
        alpha=alpha,
        rng=rng,
        seed_initial=45,
        n_it=it,
        prop_train=0.5,
        prop_test=0.2,
        data_name=data_name,
    )

    # Save metrics
    print(metrics)
    metrics.to_csv(metrics_filename, index=False)

    # save all results
    with open(f"all_metrics_{data_name}.pkl", "wb") as f:
        pickle.dump(all_results, f)
