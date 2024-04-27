# Predicting Award Winning Research Papers at Publication Time

This repository contains the code for training and testing the machine learning models described in the research and thesis project "Predicting Award Winning Research Papers at Publication Time" by Riccardo Vella.

Installation
------------

To install just clone the repository first clone the repository

```
git clone https://github.com/riccardovella/award_predictor
```

It is preferred to create an environment for installing packages

```
cd award_predictor
python3 -m venv env
source env/bin/activate
```

Install requirements using pip 

```
pip install -r requirements.txt
```

Usage
-----


    Usage:
        python src/award_predictor/main.py [<path>] [options] 

    positional arguments:
        <path>                              The directory where models and results are saved

    options:
        -h, --help                          show this help message and exit
        --gm_data_path GM_DATA_PATH         The path of the graph measures dataset
        --tx_data_path TX_DATA_PATH         The path of the text features dataset
        -hd HIDDEN, --hidden HIDDEN         The number of nodes in the hidden layer
        -l LR, --lr LR                      The learning rate
        -e EPOCHS, --epochs EPOCHS          The number of epochs
        -s SEED, --seed SEED                The random seed
        -n NAME, --name NAME                The name of the model
        -u, --unique_name                   Makes the name of the model unique by adding the date
        -m                                  Instantiates a mixed model
        --gm_model_path GM_MODEL_PATH       The path of the graph measures trained model
        --tx_model_path TX_MODEL_PATH       The path of the text features trained model

Examples
--------

The [scripts](scripts/) folder contains predefined scripts and examples for running the code. A brief description can be found in each script file.

An easy solution to train and test all models is to run the following scripts

```
./scripts/train_gm.sh 
./scripts/train_tx.sh 
# mixed has to be last
./scripts/train_mixed.sh 
```