import yaml
import torch
import optuna
import os
import pandas as pd
import sys
from batteryml.pipeline import Pipeline
import shutil

NUM_OF_TRIALS=50

def objective(trial, model_type, model):
    data = 'hust'
    config_dir = 'configs/baselines/'
    config_path = os.path.join(config_dir,model_type, model, f'{data}.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    if model_type=='nn_models':
        hyperparameters = {
            # "in_channels": trial.suggest_int("in_channels", 1, 3, step=1),
            "channels": trial.suggest_int("channels", 8, 64, step=8),
            "epochs": trial.suggest_int("epochs", 100, 1500, step=100),
            "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
            "interp_dim": trial.suggest_int("interp_dim", 600, 2000, step=200),
            # "input_height": trial.suggest_int("input_height",100, 600, step=100),
            # "input_width": trial.suggest_int("input_width",250, 10000, step=250),

        }
    config['model']['channels'] = hyperparameters['channels']
    config['model']['epochs'] = hyperparameters['epochs']
    config['model']['batch_size'] = hyperparameters['batch_size']
    config['feature']['interp_dim'] = hyperparameters['interp_dim']
    # config['model']['input_height'] = hyperparameters['input_height']
    # config['model']['input_width'] = hyperparameters['input_width']
    with open(config_path, 'w') as stream:
        yaml.safe_dump(config, stream)
    try:
        pipeline = Pipeline(config_path=config_path, workspace=f'workspaces/cnn')
        # train_loss , test_loss = pipeline.train()
        model, dataset = pipeline.train(device='cuda', skip_if_executed=False)
        train_prediction = model.predict(dataset, data_type='train')
        train_loss = dataset.evaluate(train_prediction, 'MAPE', data_type='train')
        test_prediction = model.predict(dataset, data_type='test')
        test_loss = dataset.evaluate(test_prediction, 'RMSE', data_type='test')
    except:
        return 10000

    return test_loss

results_dict = {}

# Define a list of model types to iterate over
nn_models = ['mlp' ,'cnn','lstm']
model_types=['sklearn','nn_models']

model_type='nn_models'
for model in nn_models:
    print("___________________________________________________"+model + "__________________________________________________")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_type, model), n_trials=NUM_OF_TRIALS)
    best_params = study.best_params
    best_test_loss = study.best_value
    results_dict[model] = {'best_params': best_params, 'best_test_loss': best_test_loss}
    # results_dict[model] = {'best params': best_params, 'best values': study.best_trial.values}

# Print the best parameters and test loss for each model type
for model_type, result in results_dict.items():
    best_params = result['best_params']
    best_test_loss = result['best_test_loss']
    print(f"Model type: {model_type}")
    print(f"Best parameters: {best_params}")
    print(f"Best test loss: {best_test_loss}")
file_path = "./optuna-logs/optuna_results_RMSE.txt"
with open(file_path, 'a') as file:
    # Write data to the file
    file.write("\n")
    file.write(str(results_dict))