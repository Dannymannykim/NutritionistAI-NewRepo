import torch
import math
import yaml
import numpy as np

def compile_args(path):
    """
    Compiles and returns the arguments required for the external, process configuration.

    Args:
        path (str): The .yaml path to the configuration or environment data.

    Returns:
        dict: external_args, train_args, tune_args, hyperparameters
    """

    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    external_args = config['external']
    train_args = config['train']
    tune_args = config.get('tune', {})
    hyperparameters = config.get('hyperparameters', {})

    return external_args, train_args, tune_args, hyperparameters