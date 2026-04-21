import numpy as np

class LTMFinancialProcessor:
    def __init__(self, df, mode='quarterly'):
        self.df = df.copy()
        self.mode = mode
        self.flow_cols = ['Sales/Turnover (Net)', 'Net Income (Loss)', 
                          'Selling, General and Administrative Expenses', 
                          'Cost of Goods Sold', 'Depreciation and Amortization', 
                          'Operating Income After Depreciation',
                          'Operating Income Before Depreciation']
        
        self.stock_cols = ['Assets - Total', 'Common/Ordinary Equity - Total', 
                           'Liabilities - Total', 'Current Liabilities - Total', 
                           'Current Assets - Total', 'Inventories - Total', 
                           'Cash and Short-Term Investments', 
                           'Intangible Assets - Total', 'Receivables - Total']

    def _get_base_values(self):
        if self.mode == 'quarterly':            
            # Rolling 4-quarter sum for flows
            flows = self.df.groupby('Ticker Symbol')[self.flow_cols].rolling(window=4, min_periods=4).sum().reset_index(level=0, drop=True)
            # Rolling 4-quarter mean for stocks
            stocks = self.df.groupby('Ticker Symbol')[self.stock_cols].rolling(window=4, min_periods=4).mean().reset_index(level=0, drop=True)
            debt = self.df['Long-Term Debt - Total']
        else:
            flows = self.df[self.flow_cols]
            stocks = self.df[self.stock_cols]
            debt = self.df['Long-Term Debt - Total']
            
        return flows, stocks, debt

    def calculate_ratios(self):
        flows, stocks, debt = self._get_base_values()
        
        # Create a copy of the original DF to append columns to
        df = self.df.copy()
        
        # Safe denominators
        rev = flows['Sales/Turnover (Net)'].replace(0, np.nan)
        assets = stocks['Assets - Total'].replace(0, np.nan)
        equity = stocks['Common/Ordinary Equity - Total'].replace(0, np.nan)
        curr_liab = stocks['Current Liabilities - Total'].replace(0, np.nan)
        total_liab = stocks['Liabilities - Total'].replace(0, np.nan)
        curr_assets = stocks['Current Assets - Total'].replace(0, np.nan)

        # 1. Efficiency
        df['Operating Margin'] = flows['Operating Income After Depreciation'] / rev
        df['Net Profit Margin'] = flows['Net Income (Loss)'] / rev
        df['SGA Intensity'] = flows['Selling, General and Administrative Expenses'] / rev
        df['COGS Efficiency'] = flows['Cost of Goods Sold'] / rev
        df['Depreciation Intensity'] = flows['Depreciation and Amortization'] / rev

        # 2. Solvency
        df['Debt to Equity Ratio'] = debt / equity
        df['Equity Multiplier'] = assets / equity
        df['Debt Ratio'] = total_liab / assets
        df['Current Liabilities Mix'] = curr_liab / total_liab

        # 3. Liquidity
        df['Current Ratio'] = curr_assets / curr_liab
        df['Inventory Intensity'] = stocks['Inventories - Total'] / curr_assets
        df['Cash Coverage'] = stocks['Cash and Short-Term Investments'] / curr_liab
        df['Intangibles Ratio'] = stocks['Intangible Assets - Total'] / assets
        df['Receivables Intensity'] = stocks['Receivables - Total'] / curr_assets

        # 4. Quality
        df['Accrual Ratio'] = (flows['Operating Income Before Depreciation'] - flows['Net Income (Loss)']) / assets
        df['Log_Assets'] = np.log10(assets)
        df['Log_Revenue'] = np.log10(rev)

        # Return the DataFrame with BOTH original columns AND new ratios
        # (Dropping NaNs ensures we only keep rows with valid LTM history)
        return df.dropna()