# Code for generating all data results
import pandas as pd
import os

original_path = os.getcwd()
folder_path = "/Experiments_code/pickle_files/"


def read_metrics_files(file_names, metrics, idx, reg=False, outlier=False):
    # Create DataFrame with datasets names
    dataframe = pd.DataFrame({"Dataset": file_names})
    for file_name in file_names:
        if not reg:
            if not outlier:
                file_path = (
                    original_path
                    + folder_path
                    + f"{file_name}_data/{file_name}_metrics.csv"
                )
            else:
                file_path = (
                    original_path
                    + folder_path
                    + f"{file_name}_data/{file_name}_metrics_catboost_outliers.csv"
                )
        else:
            if not outlier:
                file_path = (
                    original_path
                    + folder_path
                    + f"{file_name}_data/reg_{file_name}_metrics.csv"
                )
            else:
                file_path = (
                    original_path
                    + folder_path
                    + f"{file_name}_data/{file_name}_metrics_regression_outliers.csv"
                )
        try:
            data = pd.read_csv(file_path)

            # Filter the desired metrics in the 'Method' column
            filtered_data = data[data["Method"].isin(metrics)]

            # Add selected metrics to DataFrame
            for metric in metrics:
                if metric in filtered_data["Method"].values:
                    value_mean = filtered_data[filtered_data["Method"] == metric].iloc[
                        0, idx[0]
                    ]
                    value_std = filtered_data[filtered_data["Method"] == metric].iloc[
                        0, idx[1]
                    ]
                    if not pd.isnull(value_mean):
                        value_std = 2 * float(value_std) / (50**0.5)
                        value_mean, value_std = round(float(value_mean), 3), round(
                            value_std, 3
                        )
                        value = f"{value_mean} ({value_std})"
                        dataframe.loc[dataframe["Dataset"] == file_name, metric] = value
                else:
                    dataframe.loc[dataframe["Dataset"] == file_name, metric] = None

        except FileNotFoundError:
            print(f"File {file_path} not found.")
    return dataframe


# Define metrics and datset names
# First for quantile regression
# OBS: In the paper, it is called EPICSCORE, but the CSV data was labeled ECP. They are the same method.
metrics = ["ECP-MDN", "ECP-BART", "ECP-GP", "CQR", "CQR-r", "UACQR-P", "UACQR-S"]
file_names = [
    "airfoil",
    "bike",
    "concrete",
    "cycle",
    "electric",
    "homes",
    "meps19",
    "protein",
    "star",
    "superconductivity",
    "WEC",
    "winered",
    "winewhite",
]


# Saving each results in pickle files
def save_all_res(file_names, metrics, reg=False, outlier=False):
    if not outlier:
        # marginal coverage
        result_cover = read_metrics_files(file_names, metrics, [1, 2], reg=reg)
        # AISL
        result_aisl = read_metrics_files(file_names, metrics, [3, 4], reg=reg)
        # IL
        result_il = read_metrics_files(file_names, metrics, [5, 6], reg=reg)
        # pcor
        result_pcor = read_metrics_files(file_names, metrics, [7, 8], reg=reg)

        if not reg:
            os.makedirs(original_path + "/Experiments_code/results", exist_ok=True)
            result_cover.to_pickle(
                original_path + "/Experiments_code/" + "results/result_cover.pkl"
            )
            result_aisl.to_pickle(
                original_path + "/Experiments_code/" + "results/result_aisl.pkl"
            )
            result_il.to_pickle(
                original_path + "/Experiments_code/" + "results/result_il.pkl"
            )
            result_pcor.to_pickle(
                original_path + "/Experiments_code/" + "results/result_pcor.pkl"
            )
        else:
            os.makedirs(original_path + "Experiments_code/results", exist_ok=True)
            result_cover.to_pickle(
                original_path + "/Experiments_code/" + "results/reg_result_cover.pkl"
            )
            result_aisl.to_pickle(
                original_path + "/Experiments_code/" + "results/reg_result_aisl.pkl"
            )
            result_il.to_pickle(
                original_path + "/Experiments_code/" + "results/reg_result_il.pkl"
            )
            result_pcor.to_pickle(
                original_path + "/Experiments_code/" + "results/reg_result_pcor.pkl"
            )

        return [result_cover, result_aisl, result_il, result_pcor]
    else:
        # ratio
        result_ratio = read_metrics_files(
            file_names,
            metrics,
            [1, 2],
            reg=reg,
            outlier=outlier,
        )
        # coverage outlier
        result_cover = read_metrics_files(
            file_names,
            metrics,
            [3, 4],
            reg=reg,
            outlier=outlier,
        )

        # AISL outlier
        result_aisl = read_metrics_files(
            file_names,
            metrics,
            [5, 6],
            reg=reg,
            outlier=outlier,
        )

        if not reg:
            os.makedirs(original_path + "/Experiments_code/results", exist_ok=True)
            result_ratio.to_pickle(
                original_path
                + "/Experiments_code/"
                + "results/result_ratio_outlier.pkl"
            )
            result_aisl.to_pickle(
                original_path + "/Experiments_code/" + "results/result_aisl_outlier.pkl"
            )
            result_cover.to_pickle(
                original_path
                + "/Experiments_code/"
                + "results/result_cover_outlier.pkl"
            )

        else:
            os.makedirs(original_path + "Experiments_code/results", exist_ok=True)
            result_cover.to_pickle(
                original_path
                + "/Experiments_code/"
                + "results/reg_result_cover_outlier.pkl"
            )
            result_aisl.to_pickle(
                original_path
                + "/Experiments_code/"
                + "results/reg_result_aisl_outlier.pkl"
            )
            result_ratio.to_pickle(
                original_path
                + "/Experiments_code/"
                + "results/reg_result_ratio_outlier.pkl"
            )

        return [result_ratio, result_cover, result_aisl]


res_list = save_all_res(file_names, metrics)
print(res_list)

# marginal coverage
result = read_metrics_files(file_names, metrics, [1, 2])
# AISL
result = read_metrics_files(file_names, metrics, [3, 4])
# IL
result = read_metrics_files(file_names, metrics, [3, 4])
# [mean_metric, sd_metric]
print(result)

# Regression results
metrics = ["ECP-MDN", "ECP-BART", "ECP-GP", "mondrian", "reg-split", "weighted"]
file_names = [
    "airfoil",
    "bike",
    "concrete",
    "cycle",
    "electric",
    "homes",
    "meps19",
    "protein",
    "star",
    "superconductivity",
    "WEC",
    "winered",
    "winewhite",
]

res_list = save_all_res(file_names, metrics, reg=True)
print(res_list)


# new additional outliers/inliers results
# for quantile regression
metrics = ["ECP-MDN", "ECP-BART", "ECP-GP", "CQR", "CQR-r", "UACQR-P", "UACQR-S"]
file_names = [
    "airfoil",
    "bike",
    "concrete",
    "cycle",
    "electric",
    "homes",
    "meps19",
    "protein",
    "star",
    "superconductivity",
    "WEC",
    "winered",
    "winewhite",
]

res_list = save_all_res(file_names, metrics, reg=False, outlier=True)
print(res_list)


# for regression
metrics = ["ECP-MDN", "ECP-BART", "ECP-GP", "Mondrian", "Reg-split", "Weighted"]
file_names = [
    "airfoil",
    "bike",
    "concrete",
    "cycle",
    "electric",
    "homes",
    "meps19",
    "protein",
    "star",
    "superconductivity",
    "WEC",
    "winered",
    "winewhite",
]

res_list = save_all_res(file_names, metrics, reg=True, outlier=True)
print(res_list)


print(f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}\\toprule")
print("Dataset & " + " & ".join(metrics) + "\\\\\\midrule")
for _, row in result.iterrows():
    values = [row["Dataset"]]
    for metric in result.columns[1:]:
        value = row[metric]
        if pd.isna(value):
            values.append("-")
        else:
            values.append(f"{value}")
    print(" & ".join(values) + "\\\\")
print("\\bottomrule\\end{tabular}")
