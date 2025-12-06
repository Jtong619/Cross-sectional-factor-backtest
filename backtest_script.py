"""
Cross-sectional factor backtesting process
===================================================

- Loads monthly stock level data - reference data, returns and factor values
- Every month, for each factor:
     → the process ranks and allocates stocks into 5 buckets (quintiles)
     → Q1 = "best" stocks (high earnings/book yield etc.)
     → Q5 = "worst" stocks
- Calculates equal-weighted returns for Q1 to Q5, and the universe
- Builds a summary report including:
     → Recent 1/3/6/12-month outperformance
     → Annualized return, volatility, Sharpe ratio
     → Year-over-year and YTD outperformance + hit rates
     → Maximum drawdowns over different horizons
- Followed by subsequent pages highlighting return metrics for each factor
- Saves everything into a multi-sheet MS excel file
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# =============================================================================
# 1. Configuration
# =============================================================================
DATA_FILE = 'data/SP500_factor_data.csv'
NUM_GROUPS = 5
MIN_STOCKS = 10
ANALYST_NEEDED = 1
HIGHER_BETTER = True

# Reference data
IGNORE_COLS = ['Name', 'Sedol', 'Global', 'Region', 'Market',
               'GICSL1', 'GICSL3', 'GICSL4', 'Fin_REIT']

# =============================================================================
# 2. Check for sufficient data
# =============================================================================
def has_sufficient_data(series, n_months=None, min_coverage=0.90):
    if len(series) == 0 or pd.isna(series.iloc[-1]):
        return False
    if n_months is None:
        window = series
    else:
        window = series.iloc[-n_months:]
    return window.notna().mean() >= min_coverage

# =============================================================================
# 3. Load data
# =============================================================================
print("Loading data...")
df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
df = df[df['Coverage'] == ANALYST_NEEDED]
df['Returns'] = df['Returns'] / 100

factors = [c for c in df.columns if df[c].dtype in ['float64', 'int64']
           and c not in ['Date', 'Symbol', 'Returns', 'Coverage', 'MktCap'] + IGNORE_COLS]

print(f"Found {len(factors)} factors. Starting backtest...")

# =============================================================================
# 4. MAIN LOOP
# =============================================================================
with pd.ExcelWriter('Backtest_Results.xlsx', engine='openpyxl') as writer:
    summary_data = []

    for factor in factors:
        print(f" → Testing: {factor}")

        # 4.1 Prepare Data
        cols = ['Date', 'Symbol', 'Returns', factor]
        temp = df[cols].dropna().copy()

        # Filter dates that don't have enough stocks to form valid buckets
        valid_dates = temp.groupby('Date').size()
        temp = temp[temp['Date'].isin(valid_dates[valid_dates >= NUM_GROUPS * MIN_STOCKS].index)]

        if temp.empty:
            continue

        # 4.2 Ranking (Vectorized)
        try:
            temp['Bin'] = temp.groupby('Date')[factor].transform(
                lambda x: pd.qcut(x, NUM_GROUPS, labels=False, duplicates='drop')
            )
        except ValueError:
            continue

        # Map Bins to Q1..Q5 labels based on direction
        target_labels = range(1, NUM_GROUPS + 1)
        if HIGHER_BETTER:
            label_map = {b: f'Q{l}' for b, l in zip(range(NUM_GROUPS), reversed(target_labels))}
        else:
            label_map = {b: f'Q{l}' for b, l in zip(range(NUM_GROUPS), target_labels)}

        temp['Group'] = temp['Bin'].map(label_map)

        # 4.3 Aggregation
        monthly = temp.pivot_table(index='Date', columns='Group', values='Returns', aggfunc='mean')
        monthly = monthly.reindex(columns=[f'Q{i}' for i in range(1, NUM_GROUPS + 1)])

        universe = temp.groupby('Date')['Returns'].mean()
        excess_returns = monthly.sub(universe, axis=0).add_suffix('_vs_Mkt')

        # 4.4 Save Detail Sheet with TRUE % formatting
        output = pd.concat([
            monthly,
            universe.rename('Market'),
            excess_returns,
            (1 + monthly).cumprod().add_suffix('_Cum'),
            (1 + universe).cumprod().rename('Market_Cum')
        ], axis=1)

        # Write to Excel first
        sheet_name = factor[:31]
        output.to_excel(writer, sheet_name=sheet_name)

        # APPLY TRUE EXCEL % FORMAT TO ALL RETURN COLUMNS (except Sharpe later)
        worksheet = writer.sheets[sheet_name]
        for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row,
                                       min_col=2, max_col=worksheet.max_column):
            for cell in row:
                cell.number_format = '0.00%'

        # 4.5 Calc Summary Metrics
        q1 = monthly['Q1']
        excess = q1 - universe

        stats = {'Factor': factor}

        for m in [1, 3, 6, 12]:
            if has_sufficient_data(q1, m) and has_sufficient_data(universe, m):
                stats[f'Q1-Univ ({m}M)'] = (1 + q1.tail(m)).prod() - (1 + universe.tail(m)).prod()
            else:
                stats[f'Q1-Univ ({m}M)'] = np.nan

        if has_sufficient_data(q1) and has_sufficient_data(universe):
            ann_q1 = ((1 + q1).prod()) ** (12 / len(q1)) - 1
            ann_univ = ((1 + universe).prod()) ** (12 / len(universe)) - 1
            stats['Ann Outperf'] = ann_q1 - ann_univ
            stats['Ann Vol'] = excess.std() * np.sqrt(12)
            stats['Sharpe'] = stats['Ann Outperf'] / stats['Ann Vol'] if stats['Ann Vol'] > 0 else np.nan
        else:
            stats['Ann Outperf'] = stats['Ann Vol'] = stats['Sharpe'] = np.nan

        # Dynamic yearly + YTD Q1-Univ outperformance
        return_dates = monthly.index + pd.offsets.MonthEnd(1)
        years = sorted(return_dates.year.unique())
        current_year = pd.Timestamp.today().year

        for year in years:
            year_start = pd.Timestamp(f'{year}-01-01')
            year_end = pd.Timestamp(f'{year}-12-31')

            if year == current_year:
                year_end = return_dates.max()
                label = f"YTD '{str(year)[-2:]}'"
            else:
                label = str(year)

            q1_year = q1[(q1.index + pd.offsets.MonthEnd(1) >= year_start) & (q1.index + pd.offsets.MonthEnd(1) <= year_end)]
            univ_year = universe[(universe.index + pd.offsets.MonthEnd(1) >= year_start) & (universe.index + pd.offsets.MonthEnd(1) <= year_end)]

            if len(q1_year) > 0 and len(univ_year) > 0:
                stats[f'Q1-Univ ({label})'] = (1 + q1_year).prod() - (1 + univ_year).prod()
            else:
                stats[f'Q1-Univ ({label})'] = np.nan

        # Hit Rate (Inception + Yearly/YTD)
        stats['Hit Rate'] = (excess > 0).mean() if has_sufficient_data(excess) else np.nan

        return_dates = monthly.index + pd.offsets.MonthEnd(1)
        years = sorted(return_dates.year.unique())
        current_year = pd.Timestamp.today().year

        for year in years:
            year_start = pd.Timestamp(f'{year}-01-01')
            year_end = pd.Timestamp(f'{year}-12-31')

            if year == current_year:
                year_end = return_dates.max()
                label = f"YTD '{str(year)[-2:]}'"
            else:
                label = str(year)

            q1_year = q1[(q1.index + pd.offsets.MonthEnd(1) >= year_start) & (q1.index + pd.offsets.MonthEnd(1) <= year_end)]
            univ_year = universe[(universe.index + pd.offsets.MonthEnd(1) >= year_start) & (universe.index + pd.offsets.MonthEnd(1) <= year_end)]
            excess_year = q1_year - univ_year

            if len(excess_year) > 0:
                stats[f'Hit Rate ({label})'] = (excess_year > 0).mean()
            else:
                stats[f'Hit Rate ({label})'] = np.nan

        # Rolling Max Drawdown for 1M, 3M, 6M, 12M
        for m in [1, 3, 6, 12]:
            if has_sufficient_data(excess, m):
                rolling_q1 = (1 + q1).rolling(m).apply(np.prod, raw=True) - 1
                rolling_univ = (1 + universe).rolling(m).apply(np.prod, raw=True) - 1
                rolling_excess = rolling_q1 - rolling_univ
                stats[f'Max DD ({m}M)'] = rolling_excess.min()
            else:
                stats[f'Max DD ({m}M)'] = np.nan

        # Average # of stocks within Q1–Q5
        counts = temp.groupby(['Date', 'Group']).size().unstack(fill_value=0)
        avg_stocks = counts.mean()
        avg_stocks.index = [f'{i} count' for i in avg_stocks.index]
        for idx, val in avg_stocks.items():
            stats[idx] = round(val, 1)

        # Annualized Monthly Turnover for Q1 and Universe
        # Two-way turnover = (# in + # out) / # in previous month
        # Then multiplying by 12

        # Get Q1 membership per month
        q1_members = temp[temp['Group'] == 'Q1'].groupby('Date')['Symbol'].apply(set)

        # Get Universe membership (all stocks that passed screening)
        universe_members = temp.groupby('Date')['Symbol'].apply(set)


        # Annualized Monthly Turnover — exclude first month
        def calc_turnover(members_series):
            if len(members_series) < 2:
                return np.nan

            turnover = []
            dates = members_series.index[1:]  # ← Start from second month
            prev_sets = list(members_series.iloc[:-1])

            for prev_set, curr_set, date in zip(prev_sets, members_series.iloc[1:], dates):
                if len(prev_set) == 0:
                    turnover.append(np.nan)
                else:
                    in_new = len(curr_set - prev_set)
                    out_old = len(prev_set - curr_set)
                    turnover.append((in_new + out_old) / len(prev_set))

            return pd.Series(turnover, index=dates).mean() * 12  # Annualized


        q1_members = temp[temp['Group'] == 'Q1'].groupby('Date')['Symbol'].apply(set)
        universe_members = temp.groupby('Date')['Symbol'].apply(set)

        stats['Q1 Turnover (Ann)'] = calc_turnover(q1_members)
        stats['Universe Turnover (Ann)'] = calc_turnover(universe_members)

        summary_data.append(stats)

    # Format summary sheet
    if summary_data:
        summary = pd.DataFrame(summary_data).set_index('Factor').round(4)

        stock_cols = [c for c in summary.columns if c.endswith('count')]
        other_cols = [c for c in summary.columns if c not in stock_cols]

        new_order = (
                sorted(stock_cols) +
                [c for c in summary.columns if any(f'({m}M)' in c for m in [1, 3, 6, 12]) and 'Max DD' not in c] +
                ['Ann Outperf'] +
                sorted([c for c in summary.columns if c.startswith('Q1-Univ (') and 'M)' not in c],
                       key=lambda x: (1 if 'YTD' in x else 0, x)) +
                ['Hit Rate'] +
                sorted([c for c in summary.columns if c.startswith('Hit Rate (') and 'Inception' not in c],
                       key=lambda x: (1 if 'YTD' in x else 0, x)) +
                ['Ann Vol', 'Sharpe'] +
                [f'Max DD ({m}M)' for m in [1, 3, 6, 12]] +
                ['Q1 Turnover (Ann)', 'Universe Turnover (Ann)']
        )

        summary = summary[new_order]
        summary.to_excel(writer, sheet_name='Summary')

        ws = writer.sheets['Summary']
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row,
                                min_col=2, max_col=ws.max_column):
            for cell in row:
                col_name = summary.columns[cell.column - 2]  # Adjust for Factor index
                if 'Sharpe' in col_name or 'count' in col_name or 'Turnover' in col_name:
                    cell.number_format = '0.00'
                else:
                    cell.number_format = '0.00%'
    else:
        pd.DataFrame({'Result': ['No valid factors']}).to_excel(writer, sheet_name='Summary')

    # Move Summary to front
    workbook = writer.book
    workbook._sheets.insert(0, workbook._sheets.pop(workbook._sheets.index(workbook['Summary'])))

print("Done! Open 'Backtest_Results.xlsx'")