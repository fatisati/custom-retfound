import itertools

from trainer import Trainer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class CompareExperiments:
    def generate_trainers(self, item_to_test, submit=False):
        """Run or save multiple experiments by varying parameters according to the item_to_test."""
        keys, values = zip(*item_to_test.items())
        trainers = []
        for value_combination in tqdm(itertools.product(*values)):
            params = dict(zip(keys, value_combination))
            trainers.append(Trainer(device="cpu", **params))
        return trainers

    def get_results(self, trainers):
        """Get the results of the experiments."""
        results = []

        for trainer in tqdm(trainers):
            cm, report, all_labels, all_preds = trainer.get_model_out("test")
            results.append(report)
        return results

    def compare_bar(self, trainers, results):
        """
        Compare F1 scores (macro avg, weighted avg, etc.) of different trainers.

        Parameters:
        - trainers: list of trainer objects, each having a `model_name` attribute.
        - results: list of dictionaries, where each dictionary is structured like a sklearn classification report.
        """
        # Metrics to compare
        metrics = ["macro avg", "weighted avg"]

        # Extract F1 scores for the selected metrics
        f1_scores = []
        for result in results:
            f1_scores.append([result[metric]['f1-score'] for metric in metrics])

        # Transpose to organize data by metric
        f1_scores = np.array(f1_scores).T

        # Bar width and positions
        bar_width = 0.2
        x = np.arange(len(metrics))  # Number of metrics

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (trainer, scores) in enumerate(zip(trainers, f1_scores.T)):
            ax.bar(
                x + i * bar_width,
                scores,
                bar_width,
                label=trainer.model_name
            )

        # Customize the plot
        ax.set_title('Comparison of F1 Scores Between Trainers', fontsize=16)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_xticks(x + (len(trainers) - 1) * bar_width / 2)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(title='Models', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Display the plot
        plt.tight_layout()
        plt.show()

