# NeuralLasso

This repository collects codes for the paper 
"Boby Mathew, Andreas Hauptmann, Jens Léon and Mikko J. Sillanpää, NeuralLasso: Neural Networks Meet Lasso in Genomic Prediction, Frontiers in Plant Science, 2022"
available at https://doi.org/10.3389/fpls.2022.800161


# File description
call_GenoNet.py   - Main caller script that loads and prepares the data 
GenoNet_Load.py   - Loader helper to load and set up datasctructure
GenoNet_main.py   - Main class for the training
sparsitySearch.py - Can be used to perform sparsity search to determine sparsity parameter

# Data needed
The sample scripts are prepared for rice field data publicly available at http://www.ricediversity.org/data/

# Python packages
- Tensorflow (scripts are written for v1 in compatibility mode)
- pandas
- scikit learn
- numpy
- shutil


Updated: May, 2022