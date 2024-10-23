from __future__ import print_function, absolute_import, division

import json
import os
import time
import torch
import numpy as np

import var_cnn
import df
import evaluate
import preprocess_data
import data_generator


def update_config(config, updates):
    """Updates config dict and config file with updates dict."""
    config.update(updates)
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)


def is_valid_mixture(mixture):
    """Check if mixture is a 2D array with strings representing the models."""
    assert isinstance(mixture, list) and len(mixture) > 0
    for inner_comb in mixture:
        assert isinstance(inner_comb, list) and len(inner_comb) > 0
        for model in inner_comb:
            assert model in ['dir', 'time', 'metadata']


def train_and_val(config, model, optimizer, criterion, mixture_num, sub_model_name):
    """Train and validate model."""
    print(f'training {config["model_name"]} {sub_model_name} model')

    train_size = int((num_mon_sites * num_mon_inst_train + num_unmon_sites_train) * 0.95)
    train_steps = train_size // batch_size
    val_size = int((num_mon_sites * num_mon_inst_train + num_unmon_sites_train) * 0.05)
    val_steps = val_size // batch_size

    train_time_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(data_generator.generate(config, 'training_data', mixture_num)):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i >= train_steps:
                break
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_generator.generate(config, 'validation_data', mixture_num)):
                inputs, labels = data
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                if i >= val_steps:
                    break
    
    train_time_end = time.time()
    print(f'Total training time: {train_time_end - train_time_start}')


def predict(config, model, mixture_num, sub_model_name):
    """Compute and save final predictions on test set."""
    print(f'generating predictions for {config["model_name"]} {sub_model_name} model')

    if config["model_name"] == 'var-cnn':
        model.load_state_dict(torch.load('model_weights.pth'))

    test_size = num_mon_sites * num_mon_inst_test + num_unmon_sites_test
    test_steps = test_size // batch_size

    test_time_start = time.time()

    model.eval()
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(data_generator.generate(config, 'test_data', mixture_num)):
            inputs, _ = data
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            if i >= test_steps:
                break

    test_time_end = time.time()

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    np.save(f'{predictions_dir}{sub_model_name}_model', np.concatenate(predictions, axis=0))

    print(f'Total test time: {test_time_end - test_time_start}')


with open('config.json') as config_file:
    config = json.load(config_file)
    if config['model_name'] == 'df':
        update_config(config, {'mixture': [['dir']], 'batch_size': 128})

num_mon_sites = config['num_mon_sites']
num_mon_inst_test = config['num_mon_inst_test']
num_mon_inst_train = config['num_mon_inst_train']
num_mon_inst = num_mon_inst_test + num_mon_inst_train
num_unmon_sites_test = config['num_unmon_sites_test']
num_unmon_sites_train = config['num_unmon_sites_train']
num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

data_dir = config['data_dir']
model_name = config['model_name']
mixture = config['mixture']
batch_size = config['batch_size']
predictions_dir = config['predictions_dir']
epochs = config['var_cnn_max_epochs'] if model_name == 'var-cnn' else config['df_epochs']
is_valid_mixture(mixture)

if not os.path.exists(f'{data_dir}{num_mon_sites}_{num_mon_inst}_{num_unmon_sites_train}_{num_unmon_sites_test}.pth'):
    preprocess_data.main(config)

for mixture_num, inner_comb in enumerate(mixture):
    model, optimizer, criterion, callbacks = var_cnn.get_model(config, mixture_num) if model_name == 'var-cnn' else df.get_model(config)

    sub_model_name = '_'.join(inner_comb)
    train_and_val(config, model, optimizer, criterion, mixture_num, sub_model_name)
    predict(config, model, mixture_num, sub_model_name)

print('evaluating mixture on test data...')
evaluate.main(config)
