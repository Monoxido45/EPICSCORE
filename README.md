# Epistemic Conformal Score (EPICSCORE).
Here we provide an implementation of EPICSCORE and notebooks to reproduce the figures and tables from the paper **Epistemic Uncertainty in Conformal Scores: A Unified Approach**.

We demonstrate how to use our method on regression and quantile regression base models in two demo notebooks.

## Installing Dependencies and Package

To install the necessary dependencies and the EPICSCORE package, follow these steps:

1. Navigate to the directory containing the `setup.py` file.

2. Install the local conda environment with all dependencies by running the following command:
    ```bash
    conda env create -f EPICSCORE_env.yml
    ```

3. Activate the conda environment:
    ```bash
    conda activate EPICSCORE_env
    ```

4. Install the EPICSCORE package:
    ```bash
    pip install .
    ```


## Running Real Data Experiments

To run real data experiments, execute the following command to download and process all required data:
```bash
bash data/data_scripts/download_data.sh
```
To execute all experiments for quantile regression, run:
```bash
python Experiments_code/metrics_real_data.py
```
