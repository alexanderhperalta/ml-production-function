import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

class HistogramPlotter:
    def __init__(self, df, cols_per_row=3, bins=20, color='darkgreen'):
        self.df = df
        self.cols_per_row = cols_per_row
        self.bins = bins
        self.color = color

    def plot(self):
        num_cols = len(self.df.columns)
        num_rows = math.ceil(num_cols / self.cols_per_row)

        fig, axes = plt.subplots(nrows=num_rows, ncols=self.cols_per_row, figsize=(15, 4 * num_rows))
        axes = axes.flatten()

        for i, col in enumerate(self.df.columns):
            skewness_value = skew(self.df[col], nan_policy='omit')
            sns.histplot(self.df[col], ax=axes[i], kde=True, bins=self.bins, color=self.color)
            axes[i].set_title(col, fontsize=12)
            axes[i].set_xlabel(f'{col} (Skew = {skewness_value:.2f})', fontsize=10)
            axes[i].tick_params(axis='x', labelrotation=30, labelsize=8)
            axes[i].tick_params(axis='y', labelsize=8)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=2.0)
        plt.show()