# Federated Epidemic Surveillance

This repository contains the code and analysis pipelines associated with the paper:

**Lyu, Ruiqi, Roni Rosenfeld, and Bryan Wilder. "[Federated Epidemic Surveillance.](https://arxiv.org/abs/2307.02616)" arXiv preprint arXiv:2307.02616 (2023).**

The code in this repository implements various methods for combining statistical evidence (p-values) from multiple data sources—representing different sites or facilities—under a federated surveillance framework. This approach helps detect epidemic signals early and reliably, while allowing for decentralized data management.


## Repository Structure

### `powerSimu/` – Power Simulations and Threshold Calibration

This directory contains scripts to conduct numerical experiments evaluating the statistical power of p-value combination methods and calibrate p-value thresholds.

- **`calibratePthr.py`**:  
  Performs simulations to calibrate adjusted significance thresholds (p-value cutoffs) for different p-value combination methods under various site-sharing configurations.
  
- **`powerCombine.py`**:  
  Uses the calibrated thresholds to evaluate the statistical power of various combination methods. It simulates scenarios under null and alternative hypotheses, checking how often methods detect true signals while controlling false alarms.

### `realData/` – Real-Data Analysis

This directory contains scripts applying the federated surveillance framework to real-world hospitalization and claims data.

- **`hospitalization_lag.py`**:  
  Illustrates how to incorporate real hospitalization data, adjust for reporting lags, and combine site-level p-values into a single alarm signal.
  
- **`hospitalization_otherEvidence.py`**:  
  Similar to `hospitalization_lag.py`, but incorporates additional external evidence (e.g., other data streams like ICU beds or ED visits) into the p-value combination process.
  
- **`claim_lag.py`**:  
  Similar to `hospitalization_lag.py`, and adapts the lag-adjustment analysis for insurance claim data, applying federated surveillance methods to detect signals in claim-based time series.
  
- **`genSum_claim.py`** and **`genSum_hospitalization.py`**:  
  Preprocessing and summary-generation scripts for claims and hospitalization data, respectively. They produce cleaned and aggregated datasets suitable for subsequent analysis.

These scripts rely on your data files (e.g., `hospitalization_summary.csv`, `claim_summary.csv`) and apply multiple p-value combination methods to real datasets.

### `semiSyn/` – Semi-Synthetic Analysis

This directory contains scripts that generate and analyze semi-synthetic datasets derived from real insurance claims. The goal is to explore how federated methods perform under controlled, partially synthetic conditions.

- **`semiSyn.py`**:  
  The main script that generates semi-synthetic scenarios by adding noise and simulating facility-level allocation of counts. Evaluates how well methods detect true epidemic signals.

- **`semiSyn_allocShare.py`**:  
  Examines performance under different site allocation shares, testing how changing the distribution of data across facilities affects detection power.
  
- **`semiSyn_amplification.py`**:  
  Investigates how scaling (amplifying) the underlying signal impacts the detection rates and stability of the federated methods.
  
- **`semiSyn_numSites.py`**:  
  Analyzes how the number of sites (facilities) included in the federated system influences the combined detection performance.

### `plot/` – Visualization

After running the simulations and generating result files, you can use the scripts in `plot/` to visualize and summarize results. 

## Key Methods Implemented

The repository provides several methods for combining p-values from multiple data sources, each method offering different statistical properties:

- **Stouffer’s Method**: Combines p-values using a Z-score transformation.
- **Fisher’s Method**: Combines p-values by summing the log-transformed values.
- **Weighted Fisher’s Method (wFisher)**: Fisher’s method with weighting based on site importance or data volume.
- **Pearson’s Method**: Uses log(1 - p) transformations to combine evidence.
- **Tippett’s Method**: Takes the minimum p-value as the combined statistic.
- **LargestSite**: Selects the p-value from the site with the largest data share or weight.
- **CorrectedStouffer**: An adjusted version of Stouffer’s method that accounts for total counts and underlying proportion (`pi`).

## Data Requirements

- **Real Data**:  
  Ensure that the hospitalization or claims data are available in the expected format (e.g., CSV files similar to what we provided in the `realData/` directory).
  
- **Semi-Synthetic Data**:  
  The semi-synthetic scripts rely on an `all_states.csv` file or similarly structured files, from which they generate scenarios. Ensure paths and filenames match your local setup.

## Dependencies and Setup

**Recommended Dependencies:**

- Python 3.7+  
- NumPy  
- SciPy  
- pandas  
- matplotlib  
- argparse  

## Running the Code

1. **Power Simulations**:
   ```bash
   cd powerSimu
   python calibratePthr.py --share 2 --repeats 10
   python powerCombine.py --repeats 10
   ```
   Adjust arguments as needed. 

2. **Real Data Analysis**:
   ```bash
   cd realData
   python XXX.py
   ```

3. **Semi-Synthetic Analysis**:
   Check `semiSyn.sh`. Adjust arguments as needed. 

## Citation

If you find this code or the methods valuable for your research, please cite the paper:

```
@article{lyu2023federated,
  title={Federated Epidemic Surveillance},
  author={Lyu, Ruiqi and Rosenfeld, Roni and Wilder, Bryan},
  journal={arXiv preprint arXiv:2307.02616},
  year={2023}
}
```
