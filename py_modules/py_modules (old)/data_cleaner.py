import pandas as pd

class DataCleaner:
    def __init__(self, df, threshold=0.04, drop_top_n_rows=6):
        """
        Initialize with a DataFrame and optional threshold for % blank values
        and number of top rows to drop from the filtered summary.
        """
        self.original_df = df
        self.threshold = threshold
        self.drop_top_n_rows = drop_top_n_rows
        self.blank_summary_df = None
        self.filtered_columns = []
        self.cleaned_summary_df = None

    def summarize_blanks(self):
        """
        Count blank (NaN) values in each column and compute % of blanks.
        """
        blanks = self.original_df.isna().sum()
        columns = list(self.original_df.columns)
        zeros = self.(original_df == 0).sum()
        summary = pd.DataFrame({
            'Column Names': columns,
            '# of Blank Values': blanks,
            '% Blank Values': blanks / len(self.original_df),
            '# of Zeros': = zeros,
            '% Zeros': zeros / len(self.original_df)
        })
        self.blank_summary_df = summary
        return summary
    
    def filter_columns(self):
        """
        Keep columns with less than the threshold % of blank values.
        Drop top N rows from the filtered summary DataFrame.
        """
        if self.blank_summary_df is None:
            self.summarize_blanks()
        
        filtered_df = self.blank_summary_df[
            self.blank_summary_df['% Blank Values'] < self.threshold
        ].dropna().reset_index()

        self.filtered_columns = list(filtered_df['index'].unique())
        self.cleaned_summary_df = filtered_df.iloc[self.drop_top_n_rows:, :]
        return self.cleaned_summary_df

    def get_filtered_column_names(self):
        return self.filtered_columns

    def get_cleaned_summary_df(self):
        return self.cleaned_summary_df
