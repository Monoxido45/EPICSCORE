# importing the functions from our package
from Epistemic_CP.epistemic_cp import ECP_split
from Epistemic_CP.scores import RegressionScore
from Epistemic_CP.utils import (
    average_coverage,
    average_interval_score_loss,
    compute_interval_length,
    corr_coverage_widths,
    Net_reg,
)

# methods for comparing
from Epistemic_CP.epistemic_cp import (
    MondrianRegressionSplit,
    LocalRegressionSplit,
    RegressionSplit,
)

# base packages
import pandas as pd
import numpy as np
import os
import pickle
import gc

# importing torch functions
import torch
from pytensor.tensor.blas import ldflags

# importing scipy stats
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split

# from experiment import experiment
# original path
original_path = os.getcwd()

# folder path
folder_path = "/Experiments_code"

# fixing random generator and torch seeds
rng = np.random.default_rng(15)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


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
    rng,
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


def obtain_metrics_all_methods(
    data,
    target_column,
    base_params,
    mdn_params,
    gp_params,
    bart_params,
    rng,
    data_name="default",
    is_fitted=True,
    alpha=0.1,
    seed_initial=45,
    n_it=50,
    prop_train=0.5,
    prop_test=0.2,
    random_seed=0,
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
        X = data.drop(columns=[target_column], axis=1)
        y = data[target_column]

        # checking whether there are boolean variables and changing them to int
        bool_columns = X.select_dtypes(include=["bool"]).columns
        X[bool_columns] = X[bool_columns].astype(int)

        X_train_calib, X_test, y_train_calib, y_test = train_test_split(
            X, y, test_size=prop_test, random_state=seed
        )

        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train_calib, y_train_calib, test_size=prop_train, random_state=random_seed
        )

        # fitting base estimator
        model = Net_reg(
            input_dim=X_train.to_numpy().shape[1],
            **base_params,
        ).fit(X_train.to_numpy(), y_train.to_numpy())

        # Preparing Calibration and Test Data
        X_calib = X_calib.to_numpy()
        y_calib = y_calib.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()

        # fitting competing approaches
        # regression split
        reg_split_obj = RegressionSplit(
            model,
            alpha,
            is_fitted,
        )
        reg_split_obj.fit(X_train, y_train)
        reg_split_obj.calibrate(X_calib, y_calib)
        pred_reg_split = reg_split_obj.predict(X_test)

        # fitting mondrian
        mondrian_obj = MondrianRegressionSplit(
            base_model=model,
            alpha=alpha,
            is_fitted=is_fitted,
            split=True,
            k=30,
        )
        mondrian_obj.fit(X_train.to_numpy(), y_train.to_numpy())
        mondrian_obj.calibrate(X_calib, y_calib)
        pred_mondrian = mondrian_obj.predict(X_test)

        # fitting weighted
        weighted_obj = LocalRegressionSplit(
            base_model=model,
            alpha=alpha,
            is_fitted=is_fitted,
        )
        weighted_obj.fit(X_train.to_numpy(), y_train.to_numpy())
        weighted_obj.calibrate(X_calib, y_calib)
        pred_weighted = weighted_obj.predict(X_test)

        # fitting the different ECP methods
        ecp_obj = ECP_split(
            RegressionScore,
            model,
            alpha=alpha,
            is_fitted=True,
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
        Average_coverage_reg_split = average_coverage(
            pred_reg_split[:, 1], pred_reg_split[:, 0], y_test
        )
        Average_coverage_mondrian = average_coverage(
            pred_mondrian[:, 1], pred_mondrian[:, 0], y_test
        )
        Average_coverage_weighted = average_coverage(
            pred_weighted[:, 1], pred_weighted[:, 0], y_test
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
        ISL_reg_split = average_interval_score_loss(
            pred_reg_split[:, 1], pred_reg_split[:, 0], y_test, alpha
        )
        ISL_mondrian = average_interval_score_loss(
            pred_mondrian[:, 1], pred_mondrian[:, 0], y_test, alpha
        )
        ISL_weighted = average_interval_score_loss(
            pred_weighted[:, 1], pred_weighted[:, 0], y_test, alpha
        )

        # computing IL
        IL_mdn = np.mean(
            compute_interval_length(pred_ecp_mdn_test[:, 1], pred_ecp_mdn_test[:, 0])
        )
        IL_gp = np.mean(
            compute_interval_length(pred_ecp_gp_test[:, 1], pred_ecp_gp_test[:, 0])
        )
        IL_bart = np.mean(
            compute_interval_length(pred_ecp_bart_test[:, 1], pred_ecp_bart_test[:, 0])
        )
        IL_reg_split = np.mean(
            compute_interval_length(pred_reg_split[:, 1], pred_reg_split[:, 0])
        )
        IL_weighted = np.mean(
            compute_interval_length(pred_weighted[:, 1], pred_weighted[:, 0])
        )
        IL_mondrian = np.mean(
            compute_interval_length(pred_mondrian[:, 1], pred_mondrian[:, 0])
        )

        # computing pcorr
        pcorr_mdn = corr_coverage_widths(
            pred_ecp_mdn_test[:, 1],
            pred_ecp_mdn_test[:, 0],
            y_test,
        )
        pcorr_gp = corr_coverage_widths(
            pred_ecp_gp_test[:, 1],
            pred_ecp_gp_test[:, 0],
            y_test,
        )
        pcorr_bart = corr_coverage_widths(
            pred_ecp_bart_test[:, 1],
            pred_ecp_bart_test[:, 0],
            y_test,
        )
        pcorr_reg_split = corr_coverage_widths(
            pred_reg_split[:, 1],
            pred_reg_split[:, 0],
            y_test,
        )
        pcorr_weighted = corr_coverage_widths(
            pred_weighted[:, 1],
            pred_weighted[:, 0],
            y_test,
        )
        pcorr_mondrian = corr_coverage_widths(
            pred_mondrian[:, 1],
            pred_mondrian[:, 0],
            y_test,
        )

        # creating metric dataframe
        metric_result = pd.DataFrame(
            {
                "Metodo": [
                    "ECP-MDN",
                    "ECP-GP",
                    "ECP-BART",
                    "reg-split",
                    "weighted",
                    "mondrian",
                ],
                "Average Coverage": [
                    Average_coverage_mdn,
                    Average_coverage_gp,
                    Average_coverage_bart,
                    Average_coverage_reg_split,
                    Average_coverage_weighted,
                    Average_coverage_mondrian,
                ],
                "AISL": [
                    ISL_mdn,
                    ISL_gp,
                    ISL_bart,
                    ISL_reg_split,
                    ISL_weighted,
                    ISL_mondrian,
                ],
                "IL": [
                    IL_mdn,
                    IL_gp,
                    IL_bart,
                    IL_reg_split,
                    IL_weighted,
                    IL_mondrian,
                ],
                "pcorr": [
                    pcorr_mdn,
                    pcorr_gp,
                    pcorr_bart,
                    pcorr_reg_split,
                    pcorr_weighted,
                    pcorr_mondrian,
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
            print(f"Checkpoint saved in iteration {i+1}")

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

    for f in os.listdir("."):
        if f.startswith("checkpoint_") and f.endswith(".pkl"):
            os.remove(f)

    return all_results, summary


# models parameters
net_params = {
    "epochs": 750,
    "batch_size": 35,
    "patience": 30,
    "random_state": 450,
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
    "verbose": 0,
    "normalize_y": True,
    "log_y": False,
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
    "epistemic_model": "BART",
    "m": 100,
    "var": "heteroscedastic",
    "normalize_y": True,
    "type": "normal",
}

alpha = 0.1
CHECKPOINT_INTERVAL = 10

if __name__ == "__main__":
    print(
        "We will now compute all conformal statistics for real data in the regression setting"
    )
    data_name = input(
        "Which dataset would you like to use (e.g., 'bike' or 'winewhite')? "
    )
    metrics_filename = input(
        "Enter the filename to save metrics (e.g., 'metrics_bike.csv'): "
    )
    it = int(input("How many iterations? "))
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

    if data_name == "WEC":
        mdn_params["batch_size"] = 250
        gp_params["batch_size"] = 250

    # Compute metrics for the specified dataset
    all_results, metrics = obtain_metrics_all_methods(
        data,
        "target",
        net_params,
        mdn_params,
        gp_params,
        bart_params,
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
    metrics.to_csv("reg_" + metrics_filename, index=False)

    # save all results
    with open(f"reg_all_metrics_{data_name}.pkl", "wb") as f:
        pickle.dump(all_results, f)
