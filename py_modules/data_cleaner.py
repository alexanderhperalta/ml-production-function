import pandas as pd

class DataCleaner:
    def __init__(self, df, variable_of_interest, threshold=0.04, drop_top_n_rows=6):
        self.original_df = df
        self.variable_of_interest = variable_of_interest
        self.threshold = threshold
        self.drop_top_n_rows = drop_top_n_rows
        self.blank_summary_df = None
        self.filtered_columns = []
        self.cleaned_summary_df = None

    def summarize_columns(self):
        blanks = self.original_df.isna().sum()
        zeros = (self.original_df == 0).sum()
        summary = pd.DataFrame({
            '# of NaN Values': blanks,
            '% Nan Values': blanks / len(self.original_df),
            '# of Zeros': zeros,
            '% Zeros': zeros / len(self.original_df)
        })
        self.blank_summary_df = summary
        return summary

    def correlation(self):
        variable_of_interest = self.variable_of_interest
        if self.blank_summary_df is None:
            self.summarize_columns()

        correlations = {}
        for column in self.original_df.columns:
            if column != variable_of_interest:
                try:
                    correlations[column] = self.original_df[variable_of_interest].corr(self.original_df[column])
                except:
                    correlations[column] = None
            else:
                correlations[column] = None

        corr_series = pd.Series(correlations, name=f'Correlation with {variable_of_interest}')
        if f'Correlation with {variable_of_interest}' not in self.blank_summary_df.columns:
            self.blank_summary_df = self.blank_summary_df.join(corr_series)
        #self.blank_summary_df = self.blank_summary_df.reset_index(inplace=True)
        return self.blank_summary_df

    def filter_columns(self):
        if self.blank_summary_df is None:
            self.summarize_columns()

        # Ensure 'index' column exists
        if 'index' not in self.blank_summary_df.columns:
            self.blank_summary_df = self.blank_summary_df.reset_index()

        # Step 1: Drop top N rows
        self.cleaned_summary_df = self.blank_summary_df.iloc[self.drop_top_n_rows:, :].reset_index(drop=True)

        # Step 2: Filter based on % Nan Values
        self.filtered_df = self.cleaned_summary_df[
            self.cleaned_summary_df['% Nan Values'] < self.threshold
        ].reset_index(drop=True)

        # Step 3: Save the final list of filtered column names
        self.filtered_columns = self.filtered_df['index'].tolist()

        return self.filtered_df
    
    def get_filtered_column_names(self):
        return self.filtered_columns

    def get_cleaned_summary_df(self):
        return self.cleaned_summary_df
