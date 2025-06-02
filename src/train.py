import os
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
from search_results import load_trial_results, generate_summary_table, get_kfold_mean_metrics, plot_learning_curves
import torch
import random
from pytorch_lightning import seed_everything
from dataset.tracking_dataset import load_data_list
from sklearn.model_selection import train_test_split
from train.train_lstm import train_lstm_pipeline
import json

ALL_VIDEOS = [
    ('session1', 'center'),
    ('session1', 'left'),
    ('session1', 'right'),
    ('session2', 'center'),
    ('session2', 'left'),
    ('session2', 'right'),
    ('session3', 'center'),
    ('session3', 'left'),
    ('session3', 'right'),
    ('session4', 'center'),
    ('session4', 'left'),
    ('session4', 'right'),
    ('session5', 'center'),
    ('session5', 'left'),
    ('session5', 'right'),
    ('session6', 'center'),
    ('session6', 'left'),
    ('session6', 'right'),
]


def train(full_list, train_idx, val_idx, test_idx, input_size, hparams):
    data_module, model, trainer, logger, result = train_lstm_pipeline(
        # Dataset
        data_list=full_list,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,

        # DataModule params
        batch_size=4,
        split_ratio=(0.64, 0.16, 0.2),
        num_workers=4,

        # LSTM params
        input_size=input_size,
        hidden_size=int(hparams['hidden_size']),
        num_layers=int(hparams['num_layers']),
        output_size=1,
        learning_rate=hparams['lr'],
        dropout=hparams['dropout'],
        weight_decay=hparams['weight_decay'],

        # Logger
        log_path='test_models',

        # Trainer
        patience=7,
        epochs=100,
        log_every_n_steps=10,

        progress_bar=True,
    )


if __name__ == '__main__':
    seed = 47
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)

    full_list = load_data_list(ALL_VIDEOS, results_folder='results', feature_path='all_features')
    all_idx = np.arange(len(full_list))
    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=seed, shuffle=True
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, random_state=seed, shuffle=False
    )
    train_list = [full_list[i] for i in train_idx]
    val_list = [full_list[i] for i in val_idx]
    test_list = [full_list[i] for i in test_idx]

    hparams = {
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.06465945488805763,
        'lr': 0.0004388375502916408,
        'weight_decay': 3.549240290879743e-05,
    }

    input_size = len(train_list[0]['features'][0])

    train(full_list, train_idx, val_idx, test_idx, input_size, hparams)

