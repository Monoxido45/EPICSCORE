# Epistemic Conformal Score (EPICSCORE).
Here we provide an implementation of EPICSCORE and notebooks to reproduce the figures and tables from the paper **Epistemic Uncertainty in Conformal Scores: A Unified Approach**.

We demonstrate how to use our method on regression and quantile regression base models in two demo notebooks.

## Installing Dependencies and Package

To install the necessary dependencies and the EPICSCORE package, follow these steps:

1. Navigate to the directory containing the `setup.py` file.

2. Activate conda in the terminal
    ```bash
    source activate
    ```

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
    
Alternatively, if you use Poetry, follow these steps:

1. Navigate to the directory containing the `setup.py` file.

2. Initialize a new Poetry environment:
    ```bash
    poetry init
    ```

3. Add the required dependencies to the Poetry environment:
    ```bash
    xargs poetry add < requirements.txt
    ```


## Running Real Data Experiments

To download and process all required data, execute the following command:
```bash
bash data/data_scripts/download_data.sh
```

To run all experiments for quantile regression, use the command:
```bash
python Experiments_code/metrics_real_data.py
```
