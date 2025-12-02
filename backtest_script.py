import pandas as pd
import numpy as np
import warnings
import xlsxwriter
import matplotlib.pyplot as plt

# Suppress SettingWithCopyWarning, which often occurs with Pandas filtering
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- 1. CONFIGURATION ---
# This section defines the "Rulebook" for the backtesting process.
# Update the DATA_FILE path below.
DATA_FILE = 'data/SP500_factor_data.csv'

# Define core columns based on CSV file data structure.
DATE_COL = 'Date'
ASSET_COL = 'Symbol'
RETURN_COL = 'Returns'

# Essential Metadata
MARKET_CAP_COL = 'MktCap'
SECTOR_COL = 'GICSL2' #Keep GICSL2 for potential sector-neutral analysis

# Screening/Reference Columns
ANALYST_COVERAGE_COL = 'Coverage'

# --- DYNAMIC FACTOR CONFIGURATION ---

# EXCLUSION LIST: Columns that are in the CSV but should NOT be treated as factors.
METADATA_EXCLUSION_COLUMNS = [
    'Name',
    'Sedol',
    'Global',
    'Region',
    'Market',
    'GICSL1',
    'GICSL3',
    'GICSL4',
    'Fin_REIT',
]

# The factor direction is defined here for factor ranking.
# True: Higher the better (F1 = High; FN = Low). False: Low the better (F1 = Low; FN = High).
HIGH_VS_LOW = True

# Manual Renames: Use this to apply clean names internally while matching the CSV source name.
MANUAL_FACTOR_RENAMES = {
    # Add any factors here. Example: 'Factor_Alpha_01': 'Alpha_Clean'
}

NUM_FRACTILES = 5  # Define the number of fractiles
MIN_STOCKS_PER_FRACTILE = 10 # Minimum number of stocks required in a fractile (e.g., Q1 or Q5) for its return to be considered valid.
ANALYST_COVERAGE_THRESHOLD = 1  # Shows 1 if the stock has more than three analyst coverage.


# --- 2. DATA LOADING AND CLEANING ---
def load_and_clean_data():
    """
    Loads the CSV, performs initial cleaning, dynamically detects factor columns (by exclusion),
    and applies the analyst coverage screening.

    Returns:
        data (pd.DataFrame): The clean, screened panel data.
        clean_factor_cols (list): List of detected factor column names.
        factor_map (dict): Dictionary mapping factors to their direction (True/False).
    """

    print(f"Loading data from {DATA_FILE}...")

    try:
        df = pd.read_csv(
            DATA_FILE,
            parse_dates=[DATE_COL]  # Convert the date column to datetime objects
        )
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_FILE}. Please check the path.")
        return None, None, None
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None, None, None

    # --- IDENTIFY CORE COLUMNS AND RENAME ---
    core_csv_names = [DATE_COL, ASSET_COL, RETURN_COL, ANALYST_COVERAGE_COL, MARKET_CAP_COL, SECTOR_COL]
    core_internal_names = ['date', 'asset', 'ret', 'analyst_flag', 'mkt_cap', 'sector']

    # Check if all required core columns exist
    missing_core_cols = [col for col in core_csv_names if col not in df.columns]
    if missing_core_cols:
        print(f"ERROR: Missing core columns in CSV: {missing_core_cols}")
        return None, None, None

    # 1. Build the rename mapping dictionary
    rename_mapping = dict(zip(core_csv_names, core_internal_names))
    rename_mapping.update(MANUAL_FACTOR_RENAMES)

    # Rename columns and create a clean copy of the DataFrame
    data = df.rename(columns=rename_mapping).copy()

    # --- DATETIME NORMALIZATION ---
    # Convert the internal 'date' column to datetime objects, explicitly setting the time to midnight.
    # This ensures the column remains a proper Pandas datetime64[ns] type while removing the timestamp.
    data['date'] = data['date'].dt.normalize()

    # --- CONVERT RETURNS FROM PERCENTAGE TO DECIMAL ---
    # In the datafile, the 'ret' column contains percentage values (e.g., 1.5 for 1.5%),
    # which must be converted to decimal (0.015) for compounding formulas to work correctly.
    data['ret'] = data['ret'] / 100

    # 2. Identify Factor Columns (Dynamic Detection by Exclusion)
    # Combine the core internal names with the names of the columns to exclude
    internal_names_to_exclude = core_internal_names + [
        MANUAL_FACTOR_RENAMES.get(col, col) for col in METADATA_EXCLUSION_COLUMNS
    ]

    # A column is a factor if:
    # a) it is NOT in the exclusion list
    # b) it is a numerical data type
    clean_factor_cols = [
        col for col in data.columns
        if col not in internal_names_to_exclude and data[col].dtype in [np.float64, np.int64]
    ]

    # 3. Clean and Screen Data
    # --- UNIVERSE SCREENING (Analyst Coverage) ---
    # Apply the screening rule: only including stocks with sufficient analyst coverage (>3)
    data = data[data['analyst_flag'] == ANALYST_COVERAGE_THRESHOLD].copy()

    # 4. Create Factor Direction Map
    factor_map = {
        factor: HIGH_VS_LOW
        for factor in clean_factor_cols
    }

    print(
        f"Data loaded and screened: {len(data['asset'].unique())} unique tickers over {len(data['date'].unique())} months.")
    print(f"Dynamically Detected Factors: {len(clean_factor_cols)} factors detected.")
    return data, clean_factor_cols, factor_map

# --- 3. FACTOR RETURNS CALCULATION LOGIC ---
def calculate_factor_return(data, factor_name, factor_map):
    """
    Calculates all return series for a single factor: monthly F1 to FN returns,
    monthly universe returns, the factor premium (F1-FN), and factor outperformance vs. universe (F1-Univ).

    Args:
        data (pd.DataFrame): The clean panel data.
        factor_name (str): The name of the factor column to analyze.
        factor_map (dict): The direction map for factors.

    Returns:
        tuple:
            (pd.DataFrame): Monthly returns for each fractile (F1 to FN), indexed by 'date'.
            (pd.Series): Monthly equal-weighted return of the entire screened universe, indexed by 'date'.
            (pd.Series): Monthly factor premium (F1 - FN), indexed by 'date'.
            (pd.Series): Monthly factor Q1 outperformance vs. universe (F1 - Universe), indexed by 'date'.
    """

    # 0. Determine Factor Direction
    is_high_good = factor_map.get(factor_name, HIGH_VS_LOW)

    # 1. Handle Missing Factor Data (drops tickers that cannot be ranked)
    factor_data = data.dropna(subset=[factor_name]).copy()

    # 2. Cross-Sectional Ranking (The core of the strategy)

    def rank_assets_by_month(x):
        """
        Custom function to check for data sufficiency before ranking.
        x is the Series of factor values for a single month.
        """
        # --- FACTOR COUNT CHECK ---
        # If the number of stocks is less than the number of fractiles (e.g., < 5), pd.qcut will fail.
        # We must return NaN for the entire month's ranking to skip this month.
        if len(x) < NUM_FRACTILES:
            return pd.Series(np.nan, index=x.index)
        else:
            # Otherwise, perform the standard ranking (0 = lowest, 4 = highest).
            return pd.qcut(x, NUM_FRACTILES, labels=False, duplicates='drop')

    # Apply the ranking function
    factor_data['fractile_raw'] = factor_data.groupby('date')[factor_name].transform(rank_assets_by_month)

    # Filter out NaNs resulting from months with insufficient total data
    factor_data = factor_data.dropna(subset=['fractile_raw']).copy()

    # 3. Assign Final Fractile (1 to N, where 1 is the best-performing fractile)
    # Quantiling is used below.
    if is_high_good:
        # High Factor Value (raw_index=4) should be Q1 (final_quantile=1)
        factor_data['quantile'] = NUM_FRACTILES - factor_data['fractile_raw']
    else:
        # Low Factor Value (raw_index=0) should be Q1 (final_quantile=1)
        factor_data['quantile'] = factor_data['fractile_raw'] + 1

    # 4. Calculate Average Return per Quantile (Equal-Weighted Portfolio)
    # This is the base data for requested item #1 (Monthly average returns for each fractile)
    monthly_stats = factor_data.groupby(['date', 'quantile'])['ret'].agg(['mean', 'count']).reset_index()

    # --- IMPLEMENT MINIMUM STOCKS PER FRACTILE CHECK ---
    # Only keep groups where the stock count meets the minimum threshold
    valid_returns = monthly_stats[monthly_stats['count'] >= MIN_STOCKS_PER_FRACTILE]

    # Rename the mean column and drop the count column to get the final monthly returns
    monthly_returns = valid_returns.rename(columns={'mean': 'ret'}).drop(columns=['count'])

    # 5. Get Monthly Quantile Returns (WIDE FORMAT)
    monthly_factor_return = monthly_returns.pivot(index='date', columns='quantile', values='ret')

    # 6. Calculate Monthly Universe Return (Requested item #2)
    # Mean return of all stocks that successfully passed screening and ranking for this factor/month.
    monthly_universe_return = factor_data.groupby('date')['ret'].mean()

    # --- CHECK FOR EMPTY RESULTS ---
    # If monthly_factor_return is empty (no data passed screening), return empty objects.
    # Also check if Q1 or QN are missing, which prevents premium calculation.
    if monthly_factor_return.empty or 1 not in monthly_factor_return.columns or NUM_FRACTILES not in monthly_factor_return.columns:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    # 7. Calculate the Long-Short Factor Premium (Q1-Q5)
    monthly_factor_premium = monthly_factor_return[1] - monthly_factor_return[NUM_FRACTILES]

    # 8. Calculate Quantile Outperformance vs. Universe
    # Subtract the monthly universe return from each quantile's monthly return.
    monthly_factor_outperformance = monthly_factor_return.sub(monthly_universe_return, axis=0)

    return monthly_factor_return, monthly_universe_return, monthly_factor_premium, monthly_factor_outperformance


# --- 4. NEW FINANCIAL METRICS FUNCTIONS FOR SUMMARY TABLE ---

def get_n_month_return(return_series, n_months):
    """
    Calculates the compounded return over the last N months.
    Args:
        return_series (pd.Series): Time series of monthly returns (Q1-Univ).
        n_months (int): Number of months to look back.
    Returns:
        float: The compounded return over the specified period.
    """
    if len(return_series) < n_months:
        return np.nan
    # Calculates the product of (1 + R) for the last N months and subtracts 1.
    return (1 + return_series.iloc[-n_months:]).prod() - 1

def calculate_period_return(return_series, start_date, end_date):
    """
    Calculates the compounded return between two specific index dates (inclusive of end date).
    Returns for a given month are indexed by that month's end.
    The return period includes all returns indexed (date > start_date) and (date <= end_date).
    Example: Year 2024 Return (Dec'23 -> Dec'24) needs returns from Jan'24 to Dec'24.
    If start_date is 2023-12-31 and end_date is 2024-12-31, the returns for 2024-01-31...2024-12-31 are compounded.
    """
    # Filter returns where index is greater than the start date and less than or equal to the end date
    # We use strict inequality for start_date because the return is indexed at the end of the return month
    period_returns = return_series[(return_series.index > start_date) & (return_series.index <= end_date)]

    if period_returns.empty:
        return np.nan

    # Compound the returns
    return (1 + period_returns).prod() - 1

def calculate_annualized_metrics(return_series):
    """
    Calculates annualized return, volatility, and Sharpe ratio since inception.
    Assumes a risk-free rate of 0 for Sharpe ratio calculation.

    Args:
        return_series (pd.Series): Time series of monthly returns (Q1-Univ).

    Returns:
        tuple: (Annualized Return, Annualized Volatility, Sharpe Ratio)
    """
    n_months = len(return_series)
    if n_months == 0:
        return np.nan, np.nan, np.nan

    # Total Cumulative Return
    cumulative_return = (1 + return_series).prod() - 1

    # Annualized Return: 100 x ((Index end/Index start)^(12/# of months-1)-1)
    # (1 + Cumulative)^(12/N) - 1
    annualized_ret = ((1 + cumulative_return) ** (12 / n_months)) - 1

    # Annualized Volatility: Monthly Std Dev * sqrt(12)
    annualized_vol = return_series.std() * np.sqrt(12)

    # Sharpe Ratio: Annualized Return / Annualized Volatility (assuming Rf=0)
    sharpe_ratio = annualized_ret / annualized_vol if annualized_vol > 0 else np.nan

    return annualized_ret, annualized_vol, sharpe_ratio

def calculate_hit_rate(return_series):
    """
    Calculates the factor outperforming hit rate (% of months Q1 > Univ) since inception.

    Args:
        return_series (pd.Series): Time series of monthly returns (Q1-Univ).

    Returns:
        float: Hit Rate (as a percentage, e.g., 55.5 for 55.5%)
    """
    total_months = len(return_series)
    if total_months == 0:
        return np.nan
    # Count months where return is positive (i.e., Q1 outperformance > 0)
    winning_months = (return_series > 0).sum()
    return (winning_months / total_months) # Returns as a decimal


def calculate_max_drawdown(q1_series, univ_series, window_months=None):
    """
    Calculates the maximum drawdown defined as:
    minimum(Rolling N-month fractile returns - rolling N-month universe returns)

    Args:
        q1_series (pd.Series): Monthly returns for Q1.
        univ_series (pd.Series): Monthly returns for Universe.
        window_months (int, optional): The rolling window.

    Returns:
        float: Max Drawdown (minimum rolling outperformance)
    """
    # Defensive check: Ensure inputs are Pandas Series before proceeding.
    if not isinstance(q1_series, pd.Series) or not isinstance(univ_series, pd.Series):
        return np.nan

    # Safety check for empty data
    if len(q1_series) == 0 or len(univ_series) == 0:
        return np.nan

    if window_months is None or window_months == 1:
        # For 1-month, it is simply the minimum monthly difference
        return (q1_series - univ_series).min()
    else:
        # 1. Calculate Rolling N-month Return for Q1
        # Formula: (1+R1)*(1+R2)*... - 1 over the rolling window
        q1_rolling = (1 + q1_series).rolling(window=window_months).apply(np.prod, raw=True) - 1

        # 2. Calculate Rolling N-month Return for Universe
        univ_rolling = (1 + univ_series).rolling(window=window_months).apply(np.prod, raw=True) - 1

        # 3. Calculate the Difference (Rolling Q1 - Rolling Univ)
        diff_series = q1_rolling - univ_rolling

        # 4. Return the Minimum Value (The Max Drawdown)
        return diff_series.min()

def summarize_premium(premium_series, factor_name):
    """Prints the key performance metrics of the factor premium time series."""

    # (Details for Part 4 will go here)

    print("-" * 40)


def plot_cumulative_returns(premium_df):
    """Generates and saves a cumulative returns plot for all factor premiums."""

    # (Details for the Hybrid Output will go here)

    pass  # Placeholder for plotting logic


def main_backtest_runner():
    """
    Runs the backtest for all factors, and outputs detailed results
    (monthly returns, cumulative returns, and universe returns) for each factor
    into seperate sheets within the output file.
    """

    data, factor_list, factor_map = load_and_clean_data()

    if data is None or not factor_list:
        print("Backtest cannot run due to data loading errors or no factors found.")
        return

    EXCEL_OUTPUT_FILE = 'Backtest_results.xlsx'

    print(f"\n--- Starting backtest for {len(factor_list)} factors ---")

    # --- Initialize list to store all factor summary rows ---
    summary_data = []


    # Use pandas ExcelWriter with the xlsxwriter engine
    try:
        # Open the Excel writer to manage multiple sheets
        with pd.ExcelWriter(EXCEL_OUTPUT_FILE, engine='xlsxwriter') as writer:

            # --- Define the Date Format ---
            # Get the xlsxwriter workbook object to create custom formats
            workbook = writer.book
            # Date Format changed to YYYY-MM-DD
            date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            # Percentage format for cumulative returns
            pct_format = workbook.add_format({'num_format': '0.00%'})
            # Numeric format for Sharpe Ratio
            num_format = workbook.add_format({'num_format': '0.00'})

            # Iterate over all detected factors
            for factor_to_test in factor_list:
                print(f"Processing factor: {factor_to_test}")

                # NOTE: Calling the defined function
                monthly_factor_return, monthly_universe_return, monthly_factor_premium, monthly_factor_outperformance = calculate_factor_return(
                    data, factor_to_test, factor_map
                )

                # Extract the Q1-Univ series for summary calculations
                monthly_q1_vs_univ = monthly_factor_outperformance[1].dropna()

                # Check for empty data
                if monthly_q1_vs_univ.empty:
                    print(f"    Warning: Factor '{factor_to_test}' resulted in no valid monthly outperformance data. Skipping summary.")
                    continue

                # --- NEW: Align Q1 and Univ absolute return series to the common valid index ---
                valid_index = monthly_q1_vs_univ.index

                # Extract the ABSOLUTE Q1 return series for accurate compounding
                # We filter it by the valid index to ensure alignment with monthly_q1_vs_univ
                monthly_q1_return = monthly_factor_return[1].loc[valid_index]

                # Extract the ABSOLUTE Universe return series
                # We filter it by the valid index to ensure alignment
                monthly_universe_return_summary = monthly_universe_return.loc[valid_index]
                # --- END NEW ALIGNMENT ---

                # --- FIX START: Ensure all time series share the same index (the shifted one) ---

                # Create shifted copies of the return series to align with the cumulative returns date
                monthly_factor_return_shifted = monthly_factor_return.copy()
                monthly_universe_return_shifted = monthly_universe_return.copy()
                monthly_factor_outperformance_shifted = monthly_factor_outperformance.copy()
                monthly_factor_premium_shifted = monthly_factor_premium.copy()

                monthly_factor_return_shifted.index = monthly_factor_return_shifted.index + pd.offsets.MonthEnd(1)
                monthly_universe_return_shifted.index = monthly_universe_return_shifted.index + pd.offsets.MonthEnd(1)
                monthly_factor_outperformance_shifted.index = monthly_factor_outperformance_shifted.index + pd.offsets.MonthEnd(1)
                monthly_factor_premium_shifted.index = monthly_factor_premium_shifted.index + pd.offsets.MonthEnd(1)

                # Monthly cumulative fractile returns (Compounding applied: (1 + R).cumprod() - 1)
                # This performs the compounding calculation: (1 + R1) * (1 + R2) * ... * (1 + Rn) - 1
                # The index shift for cumulative returns is now done here *after* cumprod,
                # ensuring it aligns with the shifted monthly returns.
                cumulative_fractile_returns = (1 + monthly_factor_return).cumprod() - 1
                cumulative_fractile_returns.index = cumulative_fractile_returns.index + pd.offsets.MonthEnd(1)

                # Monthly cumulative universe returns (Compounding applied: (1 + R).cumprod() - 1)
                cumulative_universe_returns = (1 + monthly_universe_return).cumprod() - 1
                cumulative_universe_returns.index = cumulative_universe_returns.index + pd.offsets.MonthEnd(1)
                cumulative_universe_returns.name = f'Univ_CumRet'

                # --- FIX END ---


                # --------------------
                # --- SUMMARY TABLE CALCULATIONS ---
                # --------------------

                summary_row = {'Factor': factor_to_test}

                # --- 1-4. Fixed-Period Outperformance (Q1-Univ) ---
                for months in [1, 3, 6, 12]:
                    # Calculate Q1 Cumulative Return
                    # Using the aligned monthly_q1_return series
                    q1_cum_ret = get_n_month_return(monthly_q1_return, months)

                    # Calculate Universe Cumulative Return
                    # Using the aligned monthly_universe_return_summary series
                    univ_cum_ret = get_n_month_return(monthly_universe_return_summary, months)

                    # Outperformance is the difference of the compounded returns
                    outperformance = q1_cum_ret - univ_cum_ret

                    summary_row[f'Q1-Univ Ret ({months}M)'] = outperformance

                # --- 5. Annualized Q1-Univ Outperformance FIX ---
                # To match the logic: Annualized(Q1 Index) - Annualized(Univ Index)

                # 1. Calculate Annualized Return for Q1 (using absolute returns)
                ann_ret_q1, _, _ = calculate_annualized_metrics(monthly_q1_return)

                # 2. Calculate Annualized Return for Universe (using absolute returns)
                ann_ret_univ, _, _ = calculate_annualized_metrics(monthly_universe_return_summary)

                # 3. Calculate the difference for Annualized Outperformance
                ann_ret_outperformance = ann_ret_q1 - ann_ret_univ

                summary_row['Annualized Q1-Univ Outperformance'] = ann_ret_outperformance

                # --- 9, 10. Annualized Volatility and Sharpe Ratio (Q1-Univ) ---
                # These must use the excess return series (monthly_q1_vs_univ)
                # We reuse the function, but only extract the Volatility and Sharpe.
                # The first return value is discarded (as it's now calculated above).
                _, ann_vol, sharpe = calculate_annualized_metrics(monthly_q1_vs_univ)

                summary_row['Annualized Q1-Univ Volatility'] = ann_vol
                summary_row['Sharpe Ratio (Q1-Univ)'] = sharpe

                # --- DYNAMIC CALENDAR-BASED YEAR-OVER-YEAR FACTOR OUTPERFORMANCE (Q1-Univ) ---

                # Get the shifted index which represents the *end* of the return period
                # This matches the user's logic: 2024 returns are from Jan'24 to Dec'24, indexed at month ends
                shifted_index = monthly_factor_return_shifted.index

                # Identify the first and last available dates in the SHIFTED index
                first_shifted_date = shifted_index.min()
                last_shifted_date = shifted_index.max()

                # Get all unique years present in the SHIFTED index (e.g., 2024, 2025)
                # This automatically excludes 2023 if the shifted index starts in Jan 2024
                present_years = sorted(shifted_index.year.unique())

                for year in present_years:
                    # Define the start date as Dec 31st of the PREVIOUS year
                    # This captures returns starting from Jan 31st of the current year
                    start_date = pd.to_datetime(f'{year - 1}-12-31')

                    # Define the end of the current year
                    end_date_eoy = pd.to_datetime(f'{year}-12-31')

                    # Determine if this is a Full Year or YTD calculation
                    if end_date_eoy > last_shifted_date:
                        # Current year is incomplete -> YTD
                        end_date = last_shifted_date
                        col_name = f"Q1-Univ Ret (YTD '{str(year)[-2:]})"  # e.g., YTD '25
                        is_ytd = True
                    else:
                        # Full year passed -> Annual return
                        end_date = end_date_eoy
                        col_name = f'Q1-Univ Ret ({year})'  # e.g., 2024
                        is_ytd = False

                    # Calculate Compounded Returns for this period using the ABSOLUTE returns
                    # Note: calculate_period_return expects the index to match the period end dates
                    # We pass the UN-shifted returns but use UN-shifted dates logic?
                    # NO, we must use the raw returns aligned to the shifted dates logic.
                    # OR easier: calculate_period_return expects index > start and <= end.
                    # If we use the SHIFTED index (e.g., 2024-01-31), then start_date (2023-12-31) correctly captures Jan return.

                    # Let's align the absolute returns to the shifted index for calculation
                    monthly_q1_return_shifted = monthly_q1_return.copy()
                    monthly_q1_return_shifted.index = monthly_q1_return_shifted.index + pd.offsets.MonthEnd(1)

                    monthly_universe_return_shifted_calc = monthly_universe_return_summary.copy()
                    monthly_universe_return_shifted_calc.index = monthly_universe_return_shifted_calc.index + pd.offsets.MonthEnd(1)

                    q1_period_ret = calculate_period_return(monthly_q1_return_shifted, start_date, end_date)
                    univ_period_ret = calculate_period_return(monthly_universe_return_shifted_calc, start_date, end_date)

                    if pd.isna(q1_period_ret) or pd.isna(univ_period_ret):
                        yearly_ret = np.nan
                    else:
                        yearly_ret = q1_period_ret - univ_period_ret

                    summary_row[col_name] = yearly_ret

                    # --- 7/8. Hit Rates ---
                    # Calculate Hit Rate for this specific period
                    # We need the Q1-Univ monthly difference series aligned to the shifted index
                    monthly_q1_vs_univ_shifted = monthly_q1_vs_univ.copy()
                    monthly_q1_vs_univ_shifted.index = monthly_q1_vs_univ_shifted.index + pd.offsets.MonthEnd(1)

                    period_excess_returns = monthly_q1_vs_univ_shifted[
                        (monthly_q1_vs_univ_shifted.index > start_date) &
                        (monthly_q1_vs_univ_shifted.index <= end_date)
                        ]

                    period_hit_rate = calculate_hit_rate(period_excess_returns)

                    if is_ytd:
                        hit_rate_col = f"Hit Rate (YTD '{str(year)[-2:]})"
                    else:
                        hit_rate_col = f'Hit Rate ({year})'

                    summary_row[hit_rate_col] = period_hit_rate


                # --- 7. Total Hit Rate ---
                summary_row['Hit Rate (Q1 > Univ) Since Inception'] = calculate_hit_rate(monthly_q1_vs_univ)

                # --- 11-14. Max Drawdown (Q1-Univ) ---
                # FIX: The function now requires Q1 returns and Universe returns separately.
                # We use the returns series available at this point:
                # monthly_factor_return_shifted (with Q1 in column 1) and monthly_universe_return_shifted.
                for months in [1, 3, 6, 12]:
                    # Use .loc[:, 1] to access the Q1 column (labeled as integer 1 before renaming)
                    # Use .loc[:] to safely access the full Universe Series
                    summary_row[f'Max DD (Q1-Univ) ({months}M)'] = calculate_max_drawdown(
                        monthly_factor_return_shifted.loc[:, 1],
                        monthly_universe_return_shifted.loc[:],
                        months
                    )

                summary_data.append(summary_row)


                # --------------------
                # --- INDIVIDUAL FACTOR SHEET OUTPUT ---
                # --------------------

                # --- CHECK FOR EMPTY DATA BEFORE PROCEEDING ---
                if monthly_factor_return.empty:
                     print(f"   Warning: Factor '{factor_to_test}' resulted in no valid return data. Skipping.")
                     continue

                # 1. Monthly average returns for each fractile (Q1, Q2, Q3, Q4, Q5)
                # Rename columns for clear identification
                # We use the SHIFTED version here for consistency in the final concatenated DF index.
                monthly_factor_return_shifted = monthly_factor_return_shifted.rename(
                    columns={i: f'Q{i}' for i in range(1, NUM_FRACTILES + 1)}
                )

                # 2. Monthly average returns for the overall universe
                # We use the SHIFTED version here.
                monthly_universe_return_shifted.name = f'Univ'

                # --- RENAME Fractile Outperformance vs Universe ---
                # We use the SHIFTED version here.
                monthly_factor_outperformance_shifted = monthly_factor_outperformance_shifted.rename(
                    columns={i: f'Q{i}_Vs_Univ' for i in range(1, NUM_FRACTILES + 1)}
                )

                # We rename the cumulative columns using the new short names (Q1 -> Q1_CumRet)
                new_cum_names = {i: f'Q{i}_CumRet' for i in range(1, NUM_FRACTILES + 1)}
                cumulative_fractile_returns = cumulative_fractile_returns.rename(columns=new_cum_names)

                # Combine all requested data into a single DataFrame
                # Now we ONLY use the series that have the consistent (shifted) index.
                all_results_df = pd.concat([
                    monthly_factor_return_shifted,           # <- SHIFTED Index
                    monthly_universe_return_shifted,         # <- SHIFTED Index
                    cumulative_fractile_returns,  # <- SHIFTED Index
                    cumulative_universe_returns,  # <- SHIFTED Index
                    monthly_factor_outperformance_shifted,   # <- SHIFTED Index
                    monthly_factor_premium_shifted.rename(f'{factor_to_test}_Prem') # <- SHIFTED Index
                ], axis=1)

                # 6. Output to Excel (One sheet per factor)
                # Use a safe sheet name (truncated to 31 chars max)
                sheet_name = factor_to_test.replace(':', '').replace('/', '')[:31]

                # Write the DataFrame to a sheet
                all_results_df.to_excel(writer, sheet_name=sheet_name, index=True)

                # --- Apply Custom Date Formatting ---
                # Get the worksheet object created by to_excel
                worksheet = writer.sheets[sheet_name]

                # The index column (date) is column A (or 0) in Excel. Data starts on row 2
                data_rows = len(all_results_df)

                # Format is applied from A2 down to the last row of data in Column A
                worksheet.set_column('A2:A{}'.format(data_rows + 1), None, date_format)

                # Apply percentage format to all return columns
                # Data columns start at B (column 1).
                worksheet.set_column('B:Z', None, pct_format)


            # --------------------
            # --- SUMMARY SHEET GENERATION ---
            # --------------------

            summary_df = pd.DataFrame(summary_data).set_index('Factor')

            # Define the order of columns as requested
            fixed_columns = [
                'Q1-Univ Ret (1M)',
                'Q1-Univ Ret (3M)',
                'Q1-Univ Ret (6M)',
                'Q1-Univ Ret (12M)',
                'Annualized Q1-Univ Outperformance',
            ]

            # Dynamic Year-over-Year (YoY) columns
            # Ensure we sort the dynamic columns to maintain chronological order
            yoy_ret_cols = [col for col in summary_df.columns if 'Q1-Univ Ret' in col and 'M)' not in col and 'Annualized' not in col]

            # Fixed Hit Rate/Volatility/Drawdown columns
            fixed_columns_2 = [
                'Hit Rate (Q1 > Univ) Since Inception',
            ]

            # Dynamic YoY Hit Rate columns
            yoy_hit_cols = [col for col in summary_df.columns if 'Hit Rate' in col and 'Since Inception' not in col]

            fixed_columns_3 = [
                'Annualized Q1-Univ Volatility',
                'Sharpe Ratio (Q1-Univ)',
                'Max DD (Q1-Univ) (1M)',
                'Max DD (Q1-Univ) (3M)',
                'Max DD (Q1-Univ) (6M)',
                'Max DD (Q1-Univ) (12M)',
            ]

            # Final column order
            final_cols = fixed_columns + yoy_ret_cols + fixed_columns_2 + yoy_hit_cols + fixed_columns_3

            summary_df = summary_df[final_cols]

            # Write Summary DataFrame to Excel
            summary_df.to_excel(writer, sheet_name='Summary', index=True)

            # --- Apply Formatting to Summary Sheet ---
            summary_worksheet = writer.sheets['Summary']

            # Apply percentage format to most columns
            # Skip column A (Factor Name) and the Sharpe Ratio column

            # Find column index for Sharpe Ratio
            sharpe_col_idx = summary_df.columns.get_loc('Sharpe Ratio (Q1-Univ)') + 1  # +1 for the Factor index column

            for i, col_name in enumerate(summary_df.columns):
                # Column index (0 for A, 1 for B, etc.)
                excel_col_idx = i + 1

                if excel_col_idx == sharpe_col_idx:
                    # Apply numeric format for Sharpe Ratio
                    summary_worksheet.set_column(excel_col_idx, excel_col_idx, None, num_format)
                else:
                    # Apply percentage format to all other metric columns
                    summary_worksheet.set_column(excel_col_idx, excel_col_idx, None, pct_format)

            # --- Sheet Reordering: Move 'Summary' sheet to the first position (index 0) ---
            summary_idx = [ws.name for ws in workbook.worksheets()].index('Summary')
            # Pop the worksheet object from its current position and insert it at the start (index 0)
            workbook.worksheets_objs.insert(0, workbook.worksheets_objs.pop(summary_idx))
            # Additionally, activate the Summary sheet so it's the one displayed when the user opens the file
            summary_worksheet.activate()

        print(f"\nDetailed return series for all factors saved to multi-sheet Excel: {EXCEL_OUTPUT_FILE}")
        print(f"Summary table generated in sheet: 'Summary'")

    except Exception as e:
        print(f"Warning: Could not save to Excel (Check xlsxwriter installation): {e}")

    print(f"\n--- Backtest Complete ---")
    print("---------------------")

if __name__ == '__main__':
    main_backtest_runner()