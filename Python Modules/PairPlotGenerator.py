import matplotlib.pyplot as plt
import seaborn as sns
import string

class PairPlotGenerator:
    def __init__(self, dfs, main_var='Employees'):
        """
        Initialize the pair plot generator.

        Parameters:
        dfs (dict): Dictionary of DataFrames to plot.
        main_var (str): The main variable to use for the x-axis.
        """
        self.dfs = dfs
        self.main_var = main_var

    def generate_plots(self):
        """
        Generate scatter plots for each DataFrame in the dictionary.
        """
        for name, df_part in self.dfs.items():
            self._plot_pairwise(df_part, name)

    def _plot_pairwise(self, df_part, name):
        """
        Create pairwise scatter plots for a single DataFrame.
        
        Parameters:
        df_part (pd.DataFrame): DataFrame to plot.
        name (str): Name label for the DataFrame (used in titles and labels).
        """
        other_vars = [col for col in df_part.columns if col != self.main_var]

        fig, axes = plt.subplots(1, len(other_vars), figsize=(5 * len(other_vars), 4))
        if len(other_vars) == 1:
            axes = [axes]  # make iterable

        for ax, var in zip(axes, other_vars):
            sns.scatterplot(x=df_part[self.main_var], y=df_part[var], color='darkgreen', ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')

        plt.tight_layout()
        plt.show()

        # Print labels underneath
        print(f"Pairplots for {name} (x-axis vs y-axis):")
        for i, var in enumerate(other_vars):
            label = string.ascii_uppercase[i]  # A, B, C, ...
            print(f"  Plot {label}: {self.main_var} vs {var}")
