# Prediction of dephosphorylation sites using fine-tuned protein large models and integrate it as a Galaxy tool

Using transformer-based architecture (ProtTrans - short for Protein Transformer) to predict dephosphorylation sites and integrating it as a galaxy tool.

## Getting started

```bash
cd conda_requirements/
conda env create -f environment.yml
conda activate test_new
```

## Datasets

Datasets are in fasta format and can be found src/input_datasets. Original datasets can be found here https://github.com/dukkakc/DTLDephos/tree/main/dataset

## Project Organization

```plaintext

├── README.md
├── Notebooks                               <- Jupyter notebooks     
├── Plots                                   <- Plots for training history for fine-tuning, UMAP embeddings, CNN with embedding, CNN with sequences and Transformer with sequences
├── CondaRequirements
|   └── environment.yml                     <- The file necessary to recreate the analysis of requirements for the environment
├── src                                     <- Source code
    |
    ├── csv_dataset_files                   <- Fasta files converted into csv files
    ├── data_operations                     <- Script to modify existing transformer
    ├── input_datasets                      <- Train and test fasta files for site Y   
    ├── metrics                             <- Functions to compute prediction metrics
    ├── model_components                    <- Script for sequence classification tasks
    ├── model_training                      <- Scripts to train the model
    ├── neural_network_architectures        <- Scripts for CNN and Transformer architectures
    └── visualization                       <- Scripts to create visualizations of data and results
├── Galaxy_tool                             <- XML tool and python script for running the tool in galaxy 
    ├── tool.xml                   
    ├── fine_tuning.py
```
