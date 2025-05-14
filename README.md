# operator_learning-nonlinear_diffusion
operator learning method to solve nonlinear diffusion problems

# Experiment Instructions

This repository contains scripts to reproduce the results presented in the paper. Below are the instructions for running the experiments and processing the results.

# Project Structure
```
operator_learning-nonlinear_diffusion/
├── dataset/
|   ├── nd/
|   └── nd_seq/
├── result/
|   ├── exps/
|   ├── seq_exps/
|   ├── figs/
|   └── result_process.py
├── src/
│   ├── train.py
│   ├── nets.py
│   ├── utils.sh
│   ├── nlayer_exps.sh
│   ├── ntrain_exps.sh
│   ├── modes_exps.sh
│   ├── width_exps.sh
│   ├── superres_exps.sh
│   └── seq_exps.sh
└── README.md
```

# Dataset and Results

Download the dataset and results from link 

# Scripts

The following scripts are provided, each corresponding to specific tables or figures in the paper:

- `default_exps.sh`: Generates results for **Table II**, **Table III**, **Table IV**, **Fig. 7**, and **Fig. 8**.
- `nlayer_exps.sh`, `ntrain_exps.sh`, `modes_exps.sh`, `width_exps.sh`: Generate results for **Fig. 9**.
- `superres_exps.sh`: Generates results for **Table V**.
- `seq_exps.sh`: Generates results for **Fig. 10** and **Table VI**.

## Running the Experiments

To run the experiments, follow these steps:

1. Navigate to the `src` folder:

   ```bash
   cd src
   ```

2. Execute the desired experiment script, specifying the GPU device ID. Replace `xxx.sh` with the script name and `0` with the appropriate GPU device ID (e.g., `0`, `1`, etc.):

   ```bash
   bash xxx.sh device=ID
   ```

   For example, to run `default_exps.sh` on GPU device 0:

   ```bash
   bash default_exps.sh device=0
   ```

3. Repeat the above step for each script (`default_exps.sh`, `nlayer_exps.sh`, `ntrain_exps.sh`, `modes_exps.sh`, `width_exps.sh`, `superres_exps.sh`, `seq_exps.sh`) as needed.

## Processing Results

After all experiments are complete, process the results by running the following command in the `result` folder:

1. Navigate to the `result` folder:

   ```bash
   cd result
   ```

2. Run the result processing script:

   ```bash
   python result_process.py
   ```

This will generate the final results corresponding to the tables and figures mentioned above.

## Notes

- Ensure that the GPU device ID specified is valid and available on your system.
- Run the scripts sequentially if GPU resources are limited.
- The `result_process.py` script assumes that all experiments have been completed successfully.