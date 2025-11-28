import pandas as pd
import numpy as np
import warnings
import xlsxwriter
import matplotlib.pyplot as plt

# Suppress SettingWithCopyWarning, which often occurs with Pandas filtering
warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)

# --- 1. CONFIGURATION ---
# This section defines the "Rulebook" for the backtester.

# IMPORTANT: Update the DATA_FILE path below to match your actual file name and location.
DATA_FILE = 'data/SP500_factor_data.csv'

# Define core columns based on your data structure (MUST match CSV headers exactly)
DATE_COL = 'Date'
ASSET_COL = 'Symbol'
RETURN_COL = 'Returns'

# Essential Metadata
MARKET_CAP_COL = 'MktCap'
SECTOR_COL = 'GICSL2'

# Screening/Reference Columns (MUST match CSV headers exactly)
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

# The factor direction is defined here for the general case.
# True: High factor value is Good (Long Q1). False: Low factor value is Good (Long Q1).
HIGH_VS_LOW = True

# Manual Renames: Use this to apply clean names internally while matching the CSV source name.
MANUAL_FACTOR_RENAMES = {
    # Add any factors here if you need to rename them from the CSV header for internal use.
}

NUM_FRACTILES = 5  # Fractiles (e.g., Quintiles Q1 to Q5)
ANALYST_COVERAGE_THRESHOLD = 1  # Only consider stocks where Coverage = 1


# --- 2. DATA LOADING AND CLEANING ---
def load_and_clean_data():
    """
    Loads the CSV, performs initial cleaning, dynamically detects factor columns (by exclusion),
    and applies the analyst coverage screening.

    Returns: A clean DataFrame, a list of factor column names, and a map for factor direction.
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

    # Safety Check: Ensure the required core columns exist in the loaded data
    missing_core_cols = [col for col in core_csv_names if col not in df.columns]
    if missing_core_cols:
        print(f"ERROR: Missing core columns in CSV: {missing_core_cols}")
        return None, None, None

    # 1. Build the rename mapping dictionary
    rename_mapping = dict(zip(core_csv_names, core_internal_names))
    rename_mapping.update(MANUAL_FACTOR_RENAMES)

    # Rename columns and create a clean copy of the DataFrame
    data = df.rename(columns=rename_mapping).copy()

    # 2. Identify Factor Columns (Dynamic Detection by Exclusion)
    internal_names_to_exclude = core_internal_names + [
        MANUAL_FACTOR_RENAMES.get(col, col) for col in METADATA_EXCLUSION_COLUMNS
    ]

    clean_factor_cols = [
        col for col in data.columns
        if col not in internal_names_to_exclude and data[col].dtype in [np.float64, np.int64]
    ]

    # 3. Clean and Screen Data
    # Drop any row where the forward return ('ret') is missing.
    data = data.dropna(subset=['ret']).copy()

    # --- UNIVERSE SCREENING (Analyst Coverage) ---
    # Apply the screening rule: only assets with sufficient analyst coverage
    data = data[data['analyst_flag'] == ANALYST_COVERAGE_THRESHOLD].copy()

    # 4. Create Factor Direction Map
    factor_direction_map = {
        factor: HIGH_VS_LOW
        for factor in clean_factor_cols
    }

    print(
        f"Data loaded and screened: {len(data['asset'].unique())} unique assets over {len(data['date'].unique())} months.")
    print(f"Dynamically Detected Factors for Analysis ({len(clean_factor_cols)}): {clean_factor_cols}")
    return data, clean_factor_cols, factor_direction_map


def calculate_factor_premium(data, factor_name, factor_direction_map):
    """
    Calculates all return series for a single factor: monthly fractile returns,
    monthly universe returns, the factor premium, and fractile outperformance vs. universe.

    Args:
        data (pd.DataFrame): The clean panel data.
        factor_name (str): The name of the factor column to analyze.
        factor_direction_map (dict): The direction map for factors.

    Returns:
        tuple: (monthly_factor_return, monthly_universe_return, monthly_factor_premium, monthly_factor_outperformance)
    """

    # 0. Determine Factor Direction
    is_high_good = factor_direction_map.get(factor_name, HIGH_VS_LOW)

    # 1. Handle Missing Factor Data (drops assets that cannot be ranked)
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
    factor_data['quantile_raw'] = factor_data.groupby('date')[factor_name].transform(rank_assets_by_month)

    # Filter out NaNs resulting from months with insufficient total data
    factor_data = factor_data.dropna(subset=['quantile_raw']).copy()

    # 3. Assign Final Quantile (1-5, where 1 is the best-performing quantile)
    if is_high_good:
        # High Factor Value (raw_index=4) should be Q1 (final_quantile=1)
        factor_data['quantile'] = NUM_FRACTILES - factor_data['quantile_raw']
    else:
        # Low Factor Value (raw_index=0) should be Q1 (final_quantile=1)
        factor_data['quantile'] = factor_data['quantile_raw'] + 1

    # 4. Calculate Average Return per Quantile (Equal-Weighted Portfolio)
    # This is the base data for requested item #1 (Monthly average returns for each fractile)
    monthly_returns = factor_data.groupby(['date', 'quantile'])['ret'].mean().reset_index()

    # 5. Get Monthly Fractile Returns (WIDE FORMAT)
    monthly_factor_return = monthly_returns.pivot(index='date', columns='quantile', values='ret')

    # 6. Calculate Monthly Universe Return (Requested item #2)
    # Mean return of all stocks that successfully passed screening and ranking for this factor/month.
    monthly_universe_return = factor_data.groupby('date')['ret'].mean()

    # 7. Calculate the Long-Short Factor Premium
    monthly_factor_premium = monthly_factor_return[1] - monthly_factor_return[NUM_FRACTILES]

    # 8. Calculate Fractile Outperformance vs. Universe
    # Subtract the monthly universe return from each fractile's monthly return.
    monthly_factor_outperformance = monthly_factor_return.sub(monthly_universe_return, axis=0)

    return monthly_factor_return, monthly_universe_return, monthly_factor_premium, monthly_factor_outperformance


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
    Runs the backtest, focuses on the first factor, and outputs detailed results
    (monthly returns, cumulative returns, and universe returns) for verification.
    """

    data, factor_list, factor_map = load_and_clean_data()

    if data is None or not factor_list:
        print("Backtest cannot run due to data loading errors or no factors found.")
        return

    # --- FOCUS ON THE FIRST FACTOR FOR DETAILED TESTING OUTPUT ---
    factor_to_test = factor_list[0]
    print(f"\n--- Running detailed test output for factor: {factor_to_test} ---")

    # NOTE: The function now returns 4 values, named according to the new convention
    monthly_factor_return, monthly_universe_return, monthly_factor_premium, monthly_factor_outperformance = calculate_factor_premium(
        data, factor_to_test, factor_map
    )

    # 1. Monthly average returns for each fractile (Q1, Q2, Q3, Q4, Q5)
    # Rename columns for clear identification in the output CSV
    monthly_factor_return = monthly_factor_return.rename(
        columns={i: f'Q{i}' for i in range(1, NUM_FRACTILES + 1)}
    )

    # 2. Monthly average returns for the overall universe
    monthly_universe_return = monthly_universe_return.rename(f'Univ')

    # --- RENAME Fractile Outperformance vs Universe ---
    # The calculation is done in calculate_factor_premium. We just need to rename the columns here.
    monthly_factor_outperformance = monthly_factor_outperformance.rename(
        columns={i: f'Q{i}_Vs_Univ' for i in range(1, NUM_FRACTILES + 1)}
    )

    # 3. Monthly cumulative fractile returns
    # Cumulative returns: (1 + R_t) * (1 + R_{t-1}) * ... - 1
    cumulative_fractile_returns = (1 + monthly_factor_return).cumprod() - 1
    # We rename the cumulative columns using the new short names (Q1 -> Q1_CumRet)
    new_cum_names = {col: f'{col}_CumRet' for col in monthly_factor_return.columns}
    cumulative_fractile_returns = cumulative_fractile_returns.rename(columns=new_cum_names)

    # 4. Monthly cumulative universe returns
    cumulative_universe_returns = (1 + monthly_universe_return).cumprod() - 1
    cumulative_universe_returns = cumulative_universe_returns.rename(f'Univ_CumRet')

    # 5. Combine all requested data into a single DataFrame
    all_results_df = pd.concat([
        monthly_factor_return,
        monthly_universe_return,
        monthly_factor_outperformance,
        cumulative_fractile_returns,
        cumulative_universe_returns,
        monthly_factor_premium.rename(f'{factor_to_test}_Prem')
    ], axis=1)

    # 6. Output to CSV
    CSV_OUTPUT_FILE = 'backtest_results_for_testing.csv'
    all_results_df.to_csv(CSV_OUTPUT_FILE, index=True)

    # 7. Output to Excel (Often preferred for financial analysis and presentation)
    EXCEL_OUTPUT_FILE = 'backtest_results_for_testing.xlsx'

    # Use pandas ExcelWriter with the xlsxwriter engine
    try:
        with pd.ExcelWriter(EXCEL_OUTPUT_FILE, engine='xlsxwriter') as writer:
            all_results_df.to_excel(writer, sheet_name='Backtest_Results', index=True)
        print(f"Detailed return series also saved to Excel: {EXCEL_OUTPUT_FILE}")
    except Exception as e:
        print(f"Warning: Could not save to Excel (Check xlsxwriter installation): {e}")

    print(f"\n--- Test Complete ---")
    print(f"Detailed return series for factor '{factor_to_test}' saved to CSV: {CSV_OUTPUT_FILE}")
    print(
        "This file now includes the monthly outperformance of each fractile (Q1-Q5) versus the overall screened universe.")
    print("---------------------")


if __name__ == '__main__':
    main_backtest_runner()