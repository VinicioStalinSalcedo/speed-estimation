# src/evaluation/visualize.py
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(log_dir):
    df = pd.read_csv(f"{log_dir}/metrics.csv")
    df.groupby('epoch').mean()[['train_loss','val_loss']].plot(title='Loss')
    plt.show()
