"""Utility functions for the transformers notebooks"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_history(history, metric='accuracy'):
    loss = history.history[metric]
    val_loss = history.history[f'val_{metric}']
    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(epochs, loss, 'o', color='tab:blue',
            label=f'Training {metric.title()}')
    ax.plot(epochs, val_loss, color='tab:orange',
            label=f'Validation {metric.title()}', lw=2.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.title())
    ax.spines[['top', 'right']].set_visible(False)

    return fig, ax


def get_model_performance(model, test_ds, model_name):
    loss, accuracy = model.evaluate(test_ds)

    return pd.DataFrame(
        [[f'{accuracy * 100:.2f}%', loss]],
        index=[model_name],
        columns=['Accuracy', 'Loss']
    )
