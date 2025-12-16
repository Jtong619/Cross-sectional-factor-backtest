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
import os
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- OpenAI Client (optional - only needed if USE_AI_WRITEUPS is True) ---
client = None
try:
    from openai import OpenAI
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
        )
except ImportError:
    pass  # OpenAI not installed - AI write-ups will be skipped

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# --- Data Settings ---
DATA_FILE = 'data/SP500_factor_data.csv'
ANALYST_NEEDED = 1

# --- Quintile/Fractile Settings ---
NUM_GROUPS = 5                # Number of buckets (quintiles = 5)
MIN_STOCKS = 10               # Minimum stocks per group to include a date
HIGHER_BETTER = True          # True = high factor value → Q1 (best)

# --- Reference Columns (not treated as factors) ---
REFERENCE_COLS = ['Name', 'Symbol', 'Date', 'Sedol', 'Returns', 'Weight', 'MktCap',
                  'Global', 'Region', 'Market', 'GICSL1', 'GICSL2', 'GICSL3', 'GICSL4',
                  'Fin_FLAG', 'REIT_FLAG', 'Coverage']

# --- Ex-Finance Factors (use non-financial universe for turnover) ---
EX_FINANCE_FACTORS = ['12MFFCFY', 'ROIC', 'FCFConv']

# --- Write-up Settings ---
USE_AI_WRITEUPS_PAGE1 = False  # Page 1: False = manual write-up, True = AI-generated
USE_AI_WRITEUPS_PAGE2 = False  # Page 2: False = template-based, True = AI-generated

# --- Multi-Factor Rating (MFR) Definitions ---
# Each MFR is an equal-weighted average of z-scores
#   - 'required': Factors that MUST have data (no MFR if any missing)
#   - 'optional': Factors included if available (for all stocks)
#   - 'nonfin_only': Factors included ONLY for non-financial stocks (Fin_FLAG != 1)
MFR_DEFINITIONS = {
    'Value': {
        'required': ['12MFEY', '12MTBY'],
        'optional': [],
        'nonfin_only': []
    },
    'Growth': {
        'required': ['N2YEPSg', 'Sustg'],
        'optional': ['L2YEPSg'],
        'nonfin_only': ['N2YSLSg']
    },
    'Yield': {
        'required': ['TOTALYLD'],
        'optional': [],
        'nonfin_only': ['12MFFCFY']
    },
    'Revision': {
        'required': ['EPSRev', 'RecRev', 'NER'],
        'optional': [],
        'nonfin_only': []
    },
    'Momentum': {
        'required': ['PMOM_6-1M', 'PMOM_12-1M'],
        'optional': [],
        'nonfin_only': []
    },
    'Quality': {
        'required': ['N2YROE', 'EarnsCert'],
        'optional': [],
        'nonfin_only': ['ROIC', 'FCFConv']
    },
    'GARP': {
        'required': ['PEGYLD', 'PSGYLD'],
        'optional': [],
        'nonfin_only': []
    },
    'LowRisk': {
        'required': ['LowBETA', 'Size', 'LowVol'],
        'optional': [],
        'nonfin_only': []
    },
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def has_sufficient_data(series, n_months=None, min_coverage=0.90):
    """Check if series has sufficient non-null data."""
    if len(series) == 0 or pd.isna(series.iloc[-1]):
        return False
    if n_months is None:
        window = series
    else:
        window = series.iloc[-n_months:]
    return window.notna().mean() >= min_coverage


def winsorize(series, lower=0.05, upper=0.95):
    """Winsorize at 5th and 95th percentile."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def calc_zscore(series):
    """Calculate z-score using winsorized mean and std."""
    win = winsorize(series)
    return (series - win.mean()) / win.std()


def assign_quintile(pct_rank, higher_better=True):
    """Assign quintile group based on percentile rank."""
    if higher_better:
        if pct_rank > 0.8:
            return 'Q1'
        elif pct_rank > 0.6:
            return 'Q2'
        elif pct_rank < 0.2:
            return 'Q5'
        elif pct_rank < 0.4:
            return 'Q4'
        else:
            return 'Q3'
    else:
        if pct_rank > 0.8:
            return 'Q5'
        elif pct_rank > 0.6:
            return 'Q4'
        elif pct_rank < 0.2:
            return 'Q1'
        elif pct_rank < 0.4:
            return 'Q2'
        else:
            return 'Q3'


def prepare_factor_data(df, factor, num_groups=NUM_GROUPS, min_stocks=MIN_STOCKS, 
                        higher_better=HIGHER_BETTER, shift_dates=False):
    """
    Prepare factor data for backtesting: filter, rank, and pivot.
    
    Returns:
        tuple: (monthly_returns, universe_returns, stock_data) or (None, None, None) if insufficient data
        
    Parameters:
        df: DataFrame with stock data
        factor: Factor column name
        num_groups: Number of quintile groups
        min_stocks: Minimum stocks per group
        higher_better: Whether higher factor values are better
        shift_dates: If True, shift dates by 1 month for display purposes
    """
    cols = ['Date', 'Symbol', 'Returns', factor]
    temp = df[cols].dropna().copy()
    
    valid_dates = temp.groupby('Date').size()
    temp = temp[temp['Date'].isin(valid_dates[valid_dates >= num_groups * min_stocks].index)]
    
    if temp.empty:
        return None, None, None
    
    # Ranking using PERCENTRANK approach (handles ties properly)
    temp['PctRank'] = temp.groupby('Date')[factor].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    
    # Assign quintiles based on fixed percentile cutoffs
    temp['Group'] = temp['PctRank'].apply(lambda x: assign_quintile(x, higher_better))
    
    # Aggregate to monthly quintile returns
    monthly = temp.pivot_table(index='Date', columns='Group', values='Returns', aggfunc='mean')
    monthly = monthly.reindex(columns=[f'Q{i}' for i in range(1, num_groups + 1)])
    
    # Universe (factor-available stocks) returns
    universe = temp.groupby('Date')['Returns'].mean()
    
    # Optionally shift dates (for display: factor date → return date)
    if shift_dates:
        monthly.index = monthly.index + pd.offsets.MonthEnd(1)
        universe.index = universe.index + pd.offsets.MonthEnd(1)
    
    return monthly, universe, temp


def calc_turnover(members_series):
    """Calculate annualized turnover from a series of member sets."""
    if len(members_series) < 2:
        return np.nan

    turnover = []
    dates = members_series.index[1:]
    prev_sets = list(members_series.iloc[:-1])

    for prev_set, curr_set, date in zip(prev_sets, members_series.iloc[1:], dates):
        if len(prev_set) == 0:
            turnover.append(np.nan)
        else:
            in_new = len(curr_set - prev_set)
            out_old = len(prev_set - curr_set)
            turnover.append((in_new + out_old) / len(prev_set))

    return pd.Series(turnover, index=dates).mean() * 12


def generate_style_writeup(style_4q_opf, style_ytd_opf, df, mfr_names, current_year):
    """
    Generate a single unified write-up analyzing style dynamics.
    
    Returns:
        str: Combined write-up text for the right panel (single text box)
    """
    sorted_4q = sorted(style_4q_opf.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
    sorted_ytd = sorted(style_ytd_opf.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
    
    best_4q = sorted_4q[0] if sorted_4q else ("N/A", 0)
    worst_4q = sorted_4q[-1] if sorted_4q else ("N/A", 0)
    best_ytd = sorted_ytd[0] if sorted_ytd else ("N/A", 0)
    worst_ytd = sorted_ytd[-1] if sorted_ytd else ("N/A", 0)
    
    q4_start = pd.Timestamp(f'{current_year}-10-01')
    
    sector_data = []
    if 'GICSL1' in df.columns:
        recent_df = df[df['Date'] >= q4_start].copy()
        if len(recent_df) > 0:
            sector_returns = recent_df.groupby('GICSL1').agg({
                'Returns': 'mean',
                'Symbol': 'count'
            }).rename(columns={'Returns': 'AvgReturn', 'Symbol': 'Count'})
            sector_returns = sector_returns.sort_values('AvgReturn', ascending=False)
            for sector, row in sector_returns.head(3).iterrows():
                sector_data.append(f"{sector}: {row['AvgReturn']:.1%}")
            for sector, row in sector_returns.tail(3).iterrows():
                sector_data.append(f"{sector}: {row['AvgReturn']:.1%}")
    
    lowrisk_underperforming = any(s == 'LowRisk' for s, v in sorted_4q[-3:] if not np.isnan(v))
    garp_value_leading = any(s in ['GARP', 'Value'] for s, v in sorted_4q[:3] if not np.isnan(v))
    
    if lowrisk_underperforming and garp_value_leading:
        market_regime = "risk-on"
        regime_context = "The pattern of GARP/Value outperformance with LowRisk underperformance is consistent with a risk-on market environment where investors favor growth-at-reasonable-price over defensive positioning."
    elif lowrisk_underperforming:
        market_regime = "risk-on"
        regime_context = "LowRisk underperformance amid strong equity returns suggests a risk-on environment where defensive factors lag."
    else:
        market_regime = "mixed"
        regime_context = "Factor performance suggests a transitional market regime."
    
    performance_data = f"""
Style Performance Data for {current_year}:

4Q'{str(current_year)[-2:]} Q1-Q5 Outperformance (ranked best to worst):
{chr(10).join([f"  {s}: {v:.1%}" for s, v in sorted_4q if not np.isnan(v)])}

Full Year {current_year} Q1-Q5 Outperformance (ranked best to worst):
{chr(10).join([f"  {s}: {v:.1%}" for s, v in sorted_ytd if not np.isnan(v)])}

Best 4Q performer: {best_4q[0]} ({best_4q[1]:.1%})
Worst 4Q performer: {worst_4q[0]} ({worst_4q[1]:.1%})
Best YTD performer: {best_ytd[0]} ({best_ytd[1]:.1%})
Worst YTD performer: {worst_ytd[0]} ({worst_ytd[1]:.1%})

Top/Bottom Sector Returns (4Q):
{chr(10).join(sector_data) if sector_data else 'Sector data not available'}

Market Regime Assessment: {market_regime}
{regime_context}
"""

    second_best = sorted_4q[1][0] if len(sorted_4q) > 1 else 'Value'
    
    # Single unified write-up (3 bullet points)
    writeup = f"""- US styles have witnessed significant volatility in Q4, led by {best_4q[0]}, followed by {second_best}, while {worst_4q[0]} has underperformed significantly, highlighting strong market risk-on appetite, favoring growth-oriented strategy.
- However, the market preference on GARP and Value at the expense of Growth also reflects investor preference on capturing growth exposure at a reasonable price.
- Also, US growth and low-risk style returns have diverged significantly since the "Liberation Day", suggesting that defensive styles such as LowRisk and Quality could remain under pressure with liquidity and macro tailwinds in play next year."""

    # --- AI WRITE-UP (optional - set USE_AI_WRITEUPS_PAGE1 = True to enable) ---
    if USE_AI_WRITEUPS_PAGE1 and client:
        prompt = f"""You are a quantitative equity analyst. Based on this data:
{performance_data}

Write 5 bullet points analyzing style performance. Target 120 words total. Start each with a dash."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )
            writeup = response.choices[0].message.content.strip()
        except Exception as e:
            print(f" → AI fallback: {e}")

    return writeup


# =============================================================================
# 3. LOAD DATA
# =============================================================================
print("Loading data...")
df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
df = df[df['Coverage'] == ANALYST_NEEDED]
df['Returns'] = df['Returns'] / 100

factors = [c for c in df.columns if c not in REFERENCE_COLS]

# =============================================================================
# 4. CONSTRUCT MULTI-FACTOR RATINGS (MFRs)
# =============================================================================
print("Constructing MFRs...")

# Calculate z-scores for all factors by month (vectorized)
zscore_df = pd.DataFrame(index=df.index)
for factor in factors:
    zscore_df[factor] = df.groupby('Date')[factor].transform(calc_zscore)

# Create MFR columns
is_financial = df['Fin_FLAG'] == 1

for mfr_name, config in MFR_DEFINITIONS.items():
    required = config['required']
    optional = config.get('optional', [])
    nonfin_only = config.get('nonfin_only', [])
    
    # Check if all required factors have data
    required_mask = pd.Series(True, index=df.index)
    for f in required:
        if f in zscore_df.columns:
            required_mask &= zscore_df[f].notna()
        else:
            required_mask = pd.Series(False, index=df.index)
            break
    
    # Sum z-scores and count for averaging
    z_sum = pd.Series(0.0, index=df.index)
    z_count = pd.Series(0, index=df.index)
    
    # Add required factors
    for f in required:
        if f in zscore_df.columns:
            mask = zscore_df[f].notna()
            z_sum = z_sum.where(~mask, z_sum + zscore_df[f].fillna(0))
            z_count = z_count.where(~mask, z_count + 1)
    
    # Add optional factors (for all stocks)
    for f in optional:
        if f in zscore_df.columns:
            mask = zscore_df[f].notna()
            z_sum = z_sum.where(~mask, z_sum + zscore_df[f].fillna(0))
            z_count = z_count.where(~mask, z_count + 1)
    
    # Add non-financial only factors
    for f in nonfin_only:
        if f in zscore_df.columns:
            mask = zscore_df[f].notna() & ~is_financial
            z_sum = z_sum.where(~mask, z_sum + zscore_df[f].fillna(0))
            z_count = z_count.where(~mask, z_count + 1)
    
    # Calculate MFR as average z-score where required factors are available
    mfr_scores = z_sum / z_count
    mfr_scores = mfr_scores.where(required_mask, np.nan)
    df[mfr_name] = mfr_scores

# Add MFRs to factors list (MFRs first for PDF ordering)
mfr_names = list(MFR_DEFINITIONS.keys())
individual_factors = factors.copy()
factors = mfr_names + individual_factors

print(f"Found {len(factors)} factors (including {len(mfr_names)} MFRs). Starting backtest...")

# =============================================================================
# 5. MAIN BACKTEST LOOP - EXCEL OUTPUT
# =============================================================================
with pd.ExcelWriter('Backtest_Results.xlsx', engine='openpyxl') as writer:
    summary_data = []

    for factor in factors:
        print(f" → Testing: {factor}")

        # Use shared helper function
        monthly, universe, temp = prepare_factor_data(df, factor, shift_dates=False)
        
        if monthly is None:
            continue

        # Calculate excess returns
        excess_returns = monthly.sub(universe, axis=0).add_suffix('_vs_Mkt')

        # Create display version with shifted dates
        monthly_display = monthly.copy()
        monthly_display.index = monthly_display.index + pd.offsets.MonthEnd(1)
        universe_display = universe.copy()
        universe_display.index = universe_display.index + pd.offsets.MonthEnd(1)
        excess_display = excess_returns.copy()
        excess_display.index = excess_display.index + pd.offsets.MonthEnd(1)
        
        output = pd.concat([
            monthly_display,
            universe_display.rename('Market'),
            excess_display,
            (1 + monthly_display).cumprod().add_suffix('_Cum'),
            (1 + universe_display).cumprod().rename('Market_Cum')
        ], axis=1)

        sheet_name = factor[:31]
        output.to_excel(writer, sheet_name=sheet_name)

        worksheet = writer.sheets[sheet_name]
        for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row,
                                       min_col=2, max_col=worksheet.max_column):
            for cell in row:
                cell.number_format = '0.00%'

        # Calculate summary metrics
        q1 = monthly['Q1']
        excess = q1 - universe

        stats = {'Factor': factor}

        # Recent period outperformance
        for m in [1, 3, 6, 12]:
            if has_sufficient_data(q1, m) and has_sufficient_data(universe, m):
                stats[f'Q1-Univ ({m}M)'] = (1 + q1.tail(m)).prod() - (1 + universe.tail(m)).prod()
            else:
                stats[f'Q1-Univ ({m}M)'] = np.nan

        # Annualized metrics
        if has_sufficient_data(q1) and has_sufficient_data(universe):
            ann_q1 = ((1 + q1).prod()) ** (12 / len(q1)) - 1
            ann_univ = ((1 + universe).prod()) ** (12 / len(universe)) - 1
            stats['Ann Outperf'] = ann_q1 - ann_univ
            stats['Ann Vol'] = excess.std() * np.sqrt(12)
            stats['Sharpe'] = stats['Ann Outperf'] / stats['Ann Vol'] if stats['Ann Vol'] > 0 else np.nan
        else:
            stats['Ann Outperf'] = stats['Ann Vol'] = stats['Sharpe'] = np.nan

        # Year-over-year and YTD metrics (combined loop)
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

            if len(q1_year) > 0 and len(univ_year) > 0:
                stats[f'Q1-Univ ({label})'] = (1 + q1_year).prod() - (1 + univ_year).prod()
                stats[f'Hit Rate ({label})'] = (excess_year > 0).mean()
            else:
                stats[f'Q1-Univ ({label})'] = np.nan
                stats[f'Hit Rate ({label})'] = np.nan

        # Overall hit rate
        stats['Hit Rate'] = (excess > 0).mean() if has_sufficient_data(excess) else np.nan

        # Max drawdowns
        for m in [1, 3, 6, 12]:
            if has_sufficient_data(excess, m):
                rolling_q1 = (1 + q1).rolling(m).apply(np.prod, raw=True) - 1
                rolling_univ = (1 + universe).rolling(m).apply(np.prod, raw=True) - 1
                rolling_excess = rolling_q1 - rolling_univ
                stats[f'Max DD ({m}M)'] = rolling_excess.min()
            else:
                stats[f'Max DD ({m}M)'] = np.nan

        # Stock counts per quintile
        counts = temp.groupby(['Date', 'Group']).size().unstack(fill_value=0)
        avg_stocks = counts.mean()
        avg_stocks.index = [f'{i} count' for i in avg_stocks.index]
        for idx, val in avg_stocks.items():
            stats[idx] = round(val, 1)

        # Turnover calculations
        q1_members = temp[temp['Group'] == 'Q1'].groupby('Date')['Symbol'].apply(set)
        factor_universe_members = temp.groupby('Date')['Symbol'].apply(set)

        stats['Q1 Turnover (Ann)'] = calc_turnover(q1_members)
        stats['Factor Univ Turnover (Ann)'] = calc_turnover(factor_universe_members)
        
        # Overall universe turnover
        if factor in EX_FINANCE_FACTORS:
            overall_univ = df[df['Fin_FLAG'] != 1][['Date', 'Symbol', 'Returns']].dropna()
        else:
            overall_univ = df[['Date', 'Symbol', 'Returns']].dropna()
        
        overall_univ = overall_univ[overall_univ['Date'].isin(temp['Date'].unique())]
        overall_universe_members = overall_univ.groupby('Date')['Symbol'].apply(set)
        stats['Overall Univ Turnover (Ann)'] = calc_turnover(overall_universe_members)

        summary_data.append(stats)

    # Format summary sheet
    if summary_data:
        summary = pd.DataFrame(summary_data).set_index('Factor').round(4)

        stock_cols = [c for c in summary.columns if c.endswith('count')]

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
                ['Q1 Turnover (Ann)', 'Factor Univ Turnover (Ann)', 'Overall Univ Turnover (Ann)']
        )

        new_order = [c for c in new_order if c in summary.columns]
        summary = summary[new_order]
        summary.to_excel(writer, sheet_name='Summary')

        ws = writer.sheets['Summary']
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row,
                                min_col=2, max_col=ws.max_column):
            for cell in row:
                col_idx = cell.column - 2
                if col_idx >= 0 and col_idx < len(summary.columns):
                    col_name = summary.columns[col_idx]
                    if 'Sharpe' in col_name or 'count' in col_name:
                        cell.number_format = '0.00'
                    else:
                        cell.number_format = '0.00%'
    else:
        pd.DataFrame({'Result': ['No valid factors']}).to_excel(writer, sheet_name='Summary')

    workbook = writer.book
    workbook._sheets.insert(0, workbook._sheets.pop(workbook._sheets.index(workbook['Summary'])))

# =============================================================================
# 6. GENERATE PDF REPORT
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

# Use a style that works across matplotlib versions
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')

exhibit_num = 1
with PdfPages("Factor_Backtest_Report.pdf") as pdf:
    
    # ==========================================================================
    # SUMMARY PAGE 1: Style and Factor Performance Overview
    # ==========================================================================
    
    # Calculate Q1-Q5 OPF for 4Q'25 and 2025 for all factors
    current_year = pd.Timestamp.today().year
    q4_start = pd.Timestamp(f'{current_year}-10-01')
    year_start = pd.Timestamp(f'{current_year}-01-01')
    
    style_4q_opf = {}
    style_ytd_opf = {}
    style_cum_data = {}  # For cumulative lines since 2024
    
    factor_4q_opf = {}
    factor_ytd_opf = {}
    
    for factor in factors:
        monthly, universe, temp = prepare_factor_data(df, factor, shift_dates=True)
        if monthly is None:
            continue
        
        # Q1-Q5 spread
        q1_q5 = monthly['Q1'] - monthly['Q5']
        
        # 4Q data (Oct, Nov, Dec of current year)
        q4_data = q1_q5[q1_q5.index >= q4_start]
        ytd_data = q1_q5[q1_q5.index >= year_start]
        
        # Calculate cumulative OPF for the period
        q4_opf = (1 + q4_data).prod() - 1 if len(q4_data) > 0 else np.nan
        ytd_opf = (1 + ytd_data).prod() - 1 if len(ytd_data) > 0 else np.nan
        
        if factor in mfr_names:
            style_4q_opf[factor] = q4_opf
            style_ytd_opf[factor] = ytd_opf
            
            # Cumulative Q1-Q5 since 2024
            since_2024 = q1_q5[q1_q5.index >= pd.Timestamp('2024-01-01')]
            if len(since_2024) > 0:
                cum_q1_q5 = (1 + since_2024).cumprod() - 1
                # Add starting point at 0
                start_date = since_2024.index.min() - pd.offsets.MonthEnd(1)
                cum_q1_q5 = pd.concat([pd.Series([0.0], index=[start_date]), cum_q1_q5])
                style_cum_data[factor] = cum_q1_q5
        else:
            factor_4q_opf[factor] = q4_opf
            factor_ytd_opf[factor] = ytd_opf
    
    # --- Generate Write-up ---
    print(" → Generating write-up for summary page...")
    summary_writeup = generate_style_writeup(
        style_4q_opf, style_ytd_opf, df, mfr_names, current_year
    )
    
    # --- Create Summary Page 1 ---
    from matplotlib.lines import Line2D
    import textwrap
    
    fig = plt.figure(figsize=(8.27, 11.69))
    q4_label = f"Q4'{str(current_year)[-2:]}"
    fig.text(0.06, 0.95, f'S&P 500 - Style and factor {q4_label} return summary', fontsize=14, fontweight='bold', 
             ha='left', color='#006d77')
    
    # Chart 1: Style 4Q'25 and 2025 Q1-Q5 OPF (narrower to fit dual axis)
    ax1 = fig.add_axes([0.08, 0.72, 0.48, 0.16])  # [left, bottom, width, height]
    
    # Sort styles by 4Q OPF (high to low)
    sorted_styles = sorted(style_4q_opf.keys(), key=lambda s: style_4q_opf.get(s, 0), reverse=True)
    x_pos = np.arange(len(sorted_styles))
    q4_vals = [style_4q_opf.get(s, 0) for s in sorted_styles]
    ytd_vals = [style_ytd_opf.get(s, 0) for s in sorted_styles]
    
    # Determine if we need dual axis (different scales)
    q4_range = max(abs(min(q4_vals)), abs(max(q4_vals))) if q4_vals else 0.1
    ytd_range = max(abs(min(ytd_vals)), abs(max(ytd_vals))) if ytd_vals else 0.1
    use_dual_axis = abs(q4_range - ytd_range) / max(q4_range, ytd_range) > 0.5
    
    bars = ax1.bar(x_pos, q4_vals, width=0.6, color='steelblue', alpha=0.8, label="4Q'25")
    ax1.axhline(0, color='#d0d2d6', linewidth=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_styles, rotation=90, ha='center', fontsize=7)
    ax1.set_ylabel("4Q'25 Q1-Q5 OPF (%)", fontsize=7, color='black')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax1.tick_params(axis='y', labelsize=6, colors='black')
    ax1.grid(False)
    
    # Add dots for 2025 OPF
    if use_dual_axis:
        ax1b = ax1.twinx()
        ax1b.scatter(x_pos, ytd_vals, color='darkorange', s=15, zorder=5, label='2025')
        ax1b.set_ylabel('2025 Q1-Q5 OPF (%)', fontsize=7, color='black')
        ax1b.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax1b.tick_params(axis='y', labelsize=6, colors='black')
        ax1b.grid(False)
        # Align zero on both axes with proportional scaling
        y1_abs_max = max(abs(min(q4_vals)), abs(max(q4_vals))) * 1.15
        y2_abs_max = max(abs(min(ytd_vals)), abs(max(ytd_vals))) * 1.15
        ax1.set_ylim(-y1_abs_max, y1_abs_max)
        ax1b.set_ylim(-y2_abs_max, y2_abs_max)
        for spine in ax1b.spines.values():
            spine.set_visible(False)
    else:
        ax1.scatter(x_pos, ytd_vals, color='darkorange', s=15, zorder=5, label='2025')
    
    ax1.set_xlim(-0.5, len(sorted_styles) - 0.5)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    
    # Legend
    legend_elements = [plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.8, label="4Q'25"),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', 
                              markersize=8, label='2025 (RHS)')]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=6, framealpha=0.9)
    
    # Chart title and formatting lines (stop at chart margin)
    pos1 = ax1.get_position()
    line_left = pos1.x0 - 0.02
    line_right = pos1.x1 + 0.02
    fig.add_artist(plt.Line2D([line_left, line_right], [pos1.y1 + 0.02, pos1.y1 + 0.02], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
    fig.add_artist(plt.Line2D([line_left, line_right], [pos1.y0 - 0.05, pos1.y0 - 0.05], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
    fig.text(line_left, pos1.y1 + 0.025, f"Exhibit 1: Style Q1-Q5 Outperformance — 4Q'{str(current_year)[-2:]} vs {current_year}", 
             fontsize=9, fontweight='bold', ha='left')
    fig.text(line_left, pos1.y0 - 0.06, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')
    
    # Chart 2: Style Cumulative Q1-Q5 since 2024 (2/3 width, text on right)
    ax2 = fig.add_axes([0.08, 0.46, 0.55, 0.16])
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (style, cum_data) in enumerate(style_cum_data.items()):
        x_dates = mdates.date2num(cum_data.index.to_pydatetime())
        ax2.plot(x_dates, cum_data.values, linewidth=1.5, label=style, color=colors[i % len(colors)])
    
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_ylabel('Cumulative Q1-Q5 OPF (%)', fontsize=7)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax2.tick_params(axis='x', rotation=90, labelsize=6)
    ax2.tick_params(axis='y', labelsize=6)
    ax2.legend(fontsize=6, loc='best', ncol=2, framealpha=0.9)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # Chart title and formatting lines (stop at chart margin)
    pos2 = ax2.get_position()
    line_left2 = pos2.x0 - 0.02
    line_right2 = pos2.x1 + 0.02
    fig.add_artist(plt.Line2D([line_left2, line_right2], [pos2.y1 + 0.02, pos2.y1 + 0.02], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
    fig.add_artist(plt.Line2D([line_left2, line_right2], [pos2.y0 - 0.04, pos2.y0 - 0.04], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
    fig.text(line_left2, pos2.y1 + 0.025, 'Exhibit 2: Style Cumulative Q1-Q5 Outperformance (Since Jan 2024)', 
             fontsize=9, fontweight='bold', ha='left')
    fig.text(line_left2, pos2.y0 - 0.05, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')
    
    # Single unified write-up (right side, spanning both charts)
    bullets = [b.strip() for b in summary_writeup.replace('•', '-').split('\n') if b.strip().startswith('-')]
    text_x = 0.68
    text_y = pos1.y1 + 0.01  # Start at top of Chart 1
    
    for bullet in bullets[:3]:  # Up to 3 bullets
        bullet_text = textwrap.fill(bullet, width=26, subsequent_indent='  ')
        num_lines = bullet_text.count('\n') + 1
        fig.text(text_x, text_y, bullet_text, fontsize=10, ha='left', va='top', 
                 linespacing=1.25, color='#006d77')
        text_y -= 0.016 * num_lines + 0.008  # Dynamic spacing based on line count
    
    # Chart 3: All 22 factors 4Q'25 and 2025 Q1-Q5 OPF (full width)
    ax3 = fig.add_axes([0.08, 0.14, 0.87, 0.22])
    
    # Sort factors by 4Q OPF (high to low)
    sorted_factors = sorted(factor_4q_opf.keys(), key=lambda f: factor_4q_opf.get(f, 0), reverse=True)
    x_pos3 = np.arange(len(sorted_factors))
    q4_vals3 = [factor_4q_opf.get(f, 0) for f in sorted_factors]
    ytd_vals3 = [factor_ytd_opf.get(f, 0) for f in sorted_factors]
    
    # Determine if we need dual axis
    q4_range3 = max(abs(min(q4_vals3)), abs(max(q4_vals3))) if q4_vals3 else 0.1
    ytd_range3 = max(abs(min(ytd_vals3)), abs(max(ytd_vals3))) if ytd_vals3 else 0.1
    use_dual_axis3 = abs(q4_range3 - ytd_range3) / max(q4_range3, ytd_range3) > 0.5
    
    bars3 = ax3.bar(x_pos3, q4_vals3, width=0.6, color='steelblue', alpha=0.8)
    ax3.axhline(0, color='#d0d2d6', linewidth=0.8)
    ax3.set_xticks(x_pos3)
    ax3.set_xticklabels(sorted_factors, rotation=90, ha='center', fontsize=6)
    ax3.set_ylabel("4Q'25 Q1-Q5 OPF (%)", fontsize=7, color='black')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax3.tick_params(axis='y', labelsize=6, colors='black')
    ax3.grid(False)
    
    if use_dual_axis3:
        ax3b = ax3.twinx()
        ax3b.scatter(x_pos3, ytd_vals3, color='darkorange', s=15, zorder=5)
        ax3b.set_ylabel('2025 Q1-Q5 OPF (%)', fontsize=7, color='black')
        ax3b.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax3b.tick_params(axis='y', labelsize=6, colors='black')
        ax3b.grid(False)
        # Align zero on both axes with proportional scaling
        y3_abs_max = max(abs(min(q4_vals3)), abs(max(q4_vals3))) * 1.15
        y3b_abs_max = max(abs(min(ytd_vals3)), abs(max(ytd_vals3))) * 1.15
        ax3.set_ylim(-y3_abs_max, y3_abs_max)
        ax3b.set_ylim(-y3b_abs_max, y3b_abs_max)
        for spine in ax3b.spines.values():
            spine.set_visible(False)
    else:
        ax3.scatter(x_pos3, ytd_vals3, color='darkorange', s=15, zorder=5)
    
    ax3.set_xlim(-0.5, len(sorted_factors) - 0.5)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    
    # Legend for chart 3
    legend_elements3 = [plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.8, label="4Q'25"),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', 
                               markersize=8, label='2025 (RHS)')]
    ax3.legend(handles=legend_elements3, loc='upper right', fontsize=6, framealpha=0.9)
    
    # Chart title and formatting lines (full width including y-axis labels)
    pos3 = ax3.get_position()
    line_left3 = pos3.x0 - 0.02
    line_right3 = pos3.x1 + 0.02
    fig.add_artist(plt.Line2D([line_left3, line_right3], [pos3.y1 + 0.02, pos3.y1 + 0.02], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
    fig.add_artist(plt.Line2D([line_left3, line_right3], [pos3.y0 - 0.08, pos3.y0 - 0.08], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
    fig.text(line_left3, pos3.y1 + 0.025, f"Exhibit 3: Individual Factor Q1-Q5 Outperformance — 4Q'{str(current_year)[-2:]} vs {current_year}", 
             fontsize=9, fontweight='bold', ha='left')
    fig.text(line_left3, pos3.y0 - 0.09, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')
    
    pdf.savefig(fig)
    plt.close()
    print(" → Added summary page 1")
    
    # ==========================================================================
    # SUMMARY PAGE 2 - Long-term Performance
    # ==========================================================================
    
    # Calculate long-term data since 2021
    since_2021 = pd.Timestamp('2021-01-01')
    style_cum_since_2021 = {}
    style_hitrate = {}
    factor_ann_opf = {}
    
    for factor in factors:
        monthly, universe, temp = prepare_factor_data(df, factor, shift_dates=False)
        if monthly is None:
            continue
        
        q1_q5 = monthly['Q1'] - monthly['Q5']
        data_since_2021 = q1_q5[q1_q5.index >= since_2021]
        
        if factor in mfr_names:
            # Cumulative since 2021
            if len(data_since_2021) > 0:
                cum_q1_q5 = (1 + data_since_2021).cumprod() - 1
                start_date = data_since_2021.index.min() - pd.offsets.MonthEnd(1)
                cum_q1_q5 = pd.concat([pd.Series([0.0], index=[start_date]), cum_q1_q5])
                style_cum_since_2021[factor] = cum_q1_q5
            
            # Hit rate since 2021
            if len(data_since_2021) > 0:
                style_hitrate[factor] = (data_since_2021 > 0).mean()
        else:
            # Annualized OPF since 2021 for individual factors
            if len(data_since_2021) >= 12:
                total_return = (1 + data_since_2021).prod() - 1
                n_years = len(data_since_2021) / 12
                ann_opf = (1 + total_return) ** (1 / n_years) - 1
                factor_ann_opf[factor] = ann_opf
    
    # Generate page 2 write-up
    sorted_cum = sorted(style_cum_since_2021.items(), key=lambda x: x[1].iloc[-1] if len(x[1]) > 0 else 0, reverse=True)
    best_cum_styles = [s for s, _ in sorted_cum[:3]]
    worst_cum_styles = [s for s, _ in sorted_cum[-3:]]
    
    sorted_hitrate = sorted(style_hitrate.items(), key=lambda x: x[1], reverse=True)
    high_hitrate = [(s, h) for s, h in sorted_hitrate if h > 0.5]
    low_hitrate = [(s, h) for s, h in sorted_hitrate if h <= 0.5]
    
    sorted_factors_list = sorted(factor_ann_opf.items(), key=lambda x: x[1], reverse=True)
    best_factors = [f for f, _ in sorted_factors_list[:3]]
    worst_factors = [f for f, _ in sorted_factors_list[-3:]]
    
    page2_data = f"""
Long-term Performance Data (Since Jan 2021):

Best cumulative Q1-Q5 styles: {', '.join(best_cum_styles)}
Worst cumulative Q1-Q5 styles: {', '.join(worst_cum_styles)}

Hit rates above 50%: {', '.join([f"{s} ({h:.0%})" for s, h in high_hitrate])}
Hit rates at or below 50%: {', '.join([f"{s} ({h:.0%})" for s, h in low_hitrate])}

Best annualized factors: {', '.join(best_factors)}
Worst annualized factors: {', '.join(worst_factors)}
"""
    
    # AI write-up for page 2
    page2_writeup = f"""- Since 2021, {best_cum_styles[0]} and {best_cum_styles[1]} styles have delivered the strongest Q1-Q5 outperformance, while {worst_cum_styles[0]} has lagged significantly.
- Hit rate analysis shows {high_hitrate[0][0]} ({high_hitrate[0][1]:.0%}) delivers most consistent returns, while {low_hitrate[-1][0]} ({low_hitrate[-1][1]:.0%}) struggles to beat 50%.
- Factor-level analysis favors {best_factors[0]} and {best_factors[1]} at the expense of {worst_factors[-1]} and {worst_factors[-2]}."""
    
    if USE_AI_WRITEUPS_PAGE2 and client:
        prompt_p2 = f"""You are a quantitative equity analyst. Based on this data:
{page2_data}

Write exactly 3 bullet points (max 100 words total):
1. Which styles outperformed the most since 2021
2. Are styles delivering consistent returns based on hit rates
3. What factors work best at the expense of what

Start each bullet with a dash. Be concise."""
        
        try:
            response_p2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_p2}],
                max_tokens=300,
                temperature=0.7
            )
            page2_writeup = response_p2.choices[0].message.content.strip()
        except Exception as e:
            print(f" → AI fallback for page 2: {e}")
    
    fig2 = plt.figure(figsize=(8.27, 11.69))
    fig2.text(0.06, 0.95, 'S&P 500 - Style and factor long-term performance summary', fontsize=14, fontweight='bold', 
             ha='left', color='#006d77')
    
    # Chart 1: Cumulative style Q1-Q5 outperformance since 2021 (narrower for text)
    ax1 = fig2.add_axes([0.08, 0.72, 0.55, 0.16])
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (style, cum_data) in enumerate(style_cum_since_2021.items()):
        x_dates = mdates.date2num(cum_data.index.to_pydatetime())
        ax1.plot(x_dates, cum_data.values, linewidth=1.5, label=style, color=colors[i % len(colors)])
    
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_ylabel('Cumulative Q1-Q5 OPF (%)', fontsize=7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax1.tick_params(axis='x', rotation=90, labelsize=6)
    ax1.tick_params(axis='y', labelsize=6)
    ax1.legend(fontsize=5, loc='upper left', ncol=2, framealpha=0.9)
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # Chart 1 title and formatting
    pos1 = ax1.get_position()
    line_left = pos1.x0 - 0.02
    line_right = pos1.x1 + 0.02
    fig2.add_artist(plt.Line2D([line_left, line_right], [pos1.y1 + 0.02, pos1.y1 + 0.02], color='#d0d2d6', linewidth=0.8, transform=fig2.transFigure))
    fig2.add_artist(plt.Line2D([line_left, line_right], [pos1.y0 - 0.04, pos1.y0 - 0.04], color='#d0d2d6', linewidth=0.8, transform=fig2.transFigure))
    fig2.text(line_left, pos1.y1 + 0.025, 'Exhibit 4: Style Cumulative Q1-Q5 Outperformance (Since Jan 2021)', 
             fontsize=9, fontweight='bold', ha='left')
    fig2.text(line_left, pos1.y0 - 0.05, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')
    
    # Chart 2: Style long-term Q1-Q5 hit rate (narrower for text)
    ax2 = fig2.add_axes([0.08, 0.46, 0.55, 0.16])
    
    styles_sorted = [s for s, _ in sorted_hitrate]
    hitrate_vals = [h for _, h in sorted_hitrate]
    x_pos2 = np.arange(len(styles_sorted))
    
    bars2 = ax2.bar(x_pos2, hitrate_vals, width=0.6, color='steelblue', alpha=0.8)
    ax2.axhline(0.5, color='red', linewidth=1.2, linestyle='--', label='50%')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(styles_sorted, rotation=90, ha='center', fontsize=7)
    ax2.set_ylabel('Q1-Q5 Hit Rate (%)', fontsize=7)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.tick_params(axis='y', labelsize=6)
    ax2.set_ylim(0, 1)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # Chart 2 title and formatting
    pos2 = ax2.get_position()
    line_left2 = pos2.x0 - 0.02
    line_right2 = pos2.x1 + 0.02
    fig2.add_artist(plt.Line2D([line_left2, line_right2], [pos2.y1 + 0.02, pos2.y1 + 0.02], color='#d0d2d6', linewidth=0.8, transform=fig2.transFigure))
    fig2.add_artist(plt.Line2D([line_left2, line_right2], [pos2.y0 - 0.05, pos2.y0 - 0.05], color='#d0d2d6', linewidth=0.8, transform=fig2.transFigure))
    fig2.text(line_left2, pos2.y1 + 0.025, 'Exhibit 5: Style Long-term Q1-Q5 Hit Rate (Since Jan 2021)', 
             fontsize=9, fontweight='bold', ha='left')
    fig2.text(line_left2, pos2.y0 - 0.06, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')
    
    # Page 2 write-up (right side, spanning both charts)
    bullets_p2 = [b.strip() for b in page2_writeup.replace('•', '-').split('\n') if b.strip().startswith('-')]
    text_x_p2 = 0.68
    text_y_p2 = pos1.y1 + 0.01
    
    for bullet in bullets_p2[:3]:
        bullet_text = textwrap.fill(bullet, width=26, subsequent_indent='  ')
        num_lines = bullet_text.count('\n') + 1
        fig2.text(text_x_p2, text_y_p2, bullet_text, fontsize=10, ha='left', va='top', 
                 linespacing=1.25, color='#006d77')
        text_y_p2 -= 0.016 * num_lines + 0.008
    
    # Chart 3: Factor annualized Q1-Q5 outperformance since 2021
    ax3 = fig2.add_axes([0.08, 0.14, 0.87, 0.22])
    
    sorted_factors_ann = sorted(factor_ann_opf.keys(), key=lambda f: factor_ann_opf.get(f, 0), reverse=True)
    x_pos3 = np.arange(len(sorted_factors_ann))
    ann_vals = [factor_ann_opf.get(f, 0) for f in sorted_factors_ann]
    
    bars3 = ax3.bar(x_pos3, ann_vals, width=0.6, color='steelblue', alpha=0.8)
    ax3.axhline(0, color='#d0d2d6', linewidth=0.8)
    ax3.set_xticks(x_pos3)
    ax3.set_xticklabels(sorted_factors_ann, rotation=90, ha='center', fontsize=6)
    ax3.set_ylabel('Annualized Q1-Q5 OPF (%)', fontsize=7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax3.tick_params(axis='y', labelsize=6)
    ax3.grid(False)
    ax3.set_xlim(-0.5, len(sorted_factors_ann) - 0.5)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    
    # Chart 3 title and formatting
    pos3 = ax3.get_position()
    line_left3 = pos3.x0 - 0.02
    line_right3 = pos3.x1 + 0.02
    fig2.add_artist(plt.Line2D([line_left3, line_right3], [pos3.y1 + 0.02, pos3.y1 + 0.02], color='#d0d2d6', linewidth=0.8, transform=fig2.transFigure))
    fig2.add_artist(plt.Line2D([line_left3, line_right3], [pos3.y0 - 0.08, pos3.y0 - 0.08], color='#d0d2d6', linewidth=0.8, transform=fig2.transFigure))
    fig2.text(line_left3, pos3.y1 + 0.025, 'Exhibit 6: Factor Annualized Q1-Q5 Outperformance (Since Jan 2021)', 
             fontsize=9, fontweight='bold', ha='left')
    fig2.text(line_left3, pos3.y0 - 0.09, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')
    
    pdf.savefig(fig2)
    plt.close()
    print(" → Added summary page 2")
    
    # ==========================================================================
    # Q4 ATTRIBUTION ANALYSIS PAGE (Page 3)
    # ==========================================================================
    
    # Get current quarter for dynamic title
    latest_date = df['Date'].max()
    q4_quarter = f"Q4'{latest_date.strftime('%y')}"
    
    # Q4 raw dates (Sep 30, Oct 31, Nov 30 -> Oct, Nov, Dec returns)
    q4_raw_dates = pd.to_datetime(['2025-09-30', '2025-10-31', '2025-11-30'])
    df_q4 = df[df['Date'].isin(q4_raw_dates)].copy()
    
    def analyze_mfr_q1(df_subset, factor_list):
        results = []
        for date in df_subset['Date'].unique():
            month_data = df_subset[df_subset['Date'] == date].copy()
            valid_mask = month_data[factor_list + ['Returns']].notna().all(axis=1)
            month_valid = month_data[valid_mask].copy()
            
            for f in factor_list:
                month_valid[f] = winsorize(month_valid[f])
            for f in factor_list:
                mean_val = month_valid[f].mean()
                std_val = month_valid[f].std()
                month_valid[f'{f}_z'] = (month_valid[f] - mean_val) / std_val
            
            month_valid['MFR'] = month_valid[[f'{f}_z' for f in factor_list]].mean(axis=1)
            month_valid['PctRank'] = month_valid['MFR'].rank(pct=True, method='average')
            month_valid['Quintile'] = month_valid['PctRank'].apply(lambda x: assign_quintile(x, True))
            
            display_date = (date + pd.offsets.MonthEnd(1)).strftime('%b %Y')
            month_valid['Display_Month'] = display_date
            
            q1_stocks = month_valid[month_valid['Quintile'] == 'Q1'].copy()
            results.append(q1_stocks[['Date', 'Display_Month', 'Name', 'Symbol', 'GICSL1', 'Returns', 'MFR']])
        
        return pd.concat(results) if results else pd.DataFrame()
    
    # Analyze GARP and Low Risk
    garp_q1 = analyze_mfr_q1(df_q4, ['PEGYLD', 'PSGYLD'])
    lowrisk_q1 = analyze_mfr_q1(df_q4, ['LowBETA', 'LowVol'])
    
    # Aggregate stock data
    def format_months_short(months_list):
        months_sorted = sorted(set(months_list))
        if len(months_sorted) == 3:
            return "Oct, Nov, Dec'25"
        elif len(months_sorted) == 2:
            short_names = [m.replace(' 2025', "'25").replace('Oct ', 'Oct').replace('Nov ', 'Nov').replace('Dec ', 'Dec') for m in months_sorted]
            return ', '.join([s.split()[0] if ' ' in s else s[:3] for s in months_sorted]).replace(' 2025', '') + "'25"
        else:
            return months_sorted[0].replace(' 2025', "'25")
    
    def get_stock_summary(q1_data):
        stock_agg = q1_data.groupby(['Symbol', 'Name', 'GICSL1']).agg({
            'Returns': 'mean',
            'Display_Month': lambda x: format_months_short(x.tolist())
        }).reset_index()
        stock_agg.columns = ['Symbol', 'Name', 'Sector', 'Avg_Return', 'Months']
        return stock_agg
    
    def get_sector_summary(q1_data):
        sector_agg = q1_data.groupby('GICSL1').agg({
            'Returns': 'mean',
            'Symbol': 'nunique'
        }).reset_index()
        sector_agg.columns = ['Sector', 'Avg_Return', 'Unique_Stocks']
        return sector_agg.sort_values('Avg_Return', ascending=False)
    
    garp_stock = get_stock_summary(garp_q1).sort_values('Avg_Return', ascending=False)
    garp_sector = get_sector_summary(garp_q1)
    lr_stock = get_stock_summary(lowrisk_q1).sort_values('Avg_Return', ascending=True)
    lr_sector = get_sector_summary(lowrisk_q1)
    
    # Create Page 3
    fig3 = plt.figure(figsize=(8.27, 11.69))
    
    # Page title
    fig3.text(0.06, 0.95, f'S&P 500 - {q4_quarter} Style Attribution Analysis', fontsize=14, fontweight='bold', 
             ha='left', color='#006d77')
    
    # Key insights line (wrapped)
    key_insight = "Key Insight: Semiconductor stocks (SNDK, MCHP, MU) drove GARP outperformance;\ndefensive sectors (Utilities, Consumer Staples) weighed on Low Risk."
    fig3.text(0.06, 0.92, key_insight, fontsize=9, ha='left', style='italic', color='#333333', linespacing=1.3)
    
    # Table 1: GARP Q1 Key Performers (top left)
    ax1 = fig3.add_axes([0.06, 0.60, 0.42, 0.28])
    ax1.axis('off')
    
    garp_top10 = garp_stock.head(10)
    table1_data = [[row['Symbol'], row['Sector'][:20], f"{row['Avg_Return']:+.1%}", row['Months']] 
                   for _, row in garp_top10.iterrows()]
    
    table1 = ax1.table(
        cellText=table1_data,
        colLabels=['Stock', 'Sector', 'Avg Ret', 'Months in Q1'],
        colWidths=[0.12, 0.38, 0.15, 0.35],
        loc='upper center',
        cellLoc='left'
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(7)
    table1.scale(1, 1.4)
    for j in range(4):
        cell = table1[(0, j)]
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor('#006d77')
    for i in range(1, len(table1_data) + 1):
        for j in range(4):
            cell = table1[(i, j)]
            cell.set_facecolor('#f0f7f7' if i % 2 == 0 else 'white')
            cell.set_edgecolor('#d0d2d6')
    
    fig3.text(0.06, 0.89, f'Exhibit 7: GARP Q1 Key Performers ({q4_quarter})', fontsize=9, fontweight='bold')
    
    # Table 2: GARP Q1 Sector Contribution (top right)
    ax2 = fig3.add_axes([0.52, 0.60, 0.42, 0.28])
    ax2.axis('off')
    
    table2_data = [[row['Sector'][:22], f"{row['Avg_Return']:+.1%}", str(int(row['Unique_Stocks']))] 
                   for _, row in garp_sector.iterrows()]
    
    table2 = ax2.table(
        cellText=table2_data,
        colLabels=['Sector', 'Avg Return', '# Stocks'],
        colWidths=[0.55, 0.25, 0.20],
        loc='upper center',
        cellLoc='left'
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(7)
    table2.scale(1, 1.4)
    for j in range(3):
        cell = table2[(0, j)]
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor('#006d77')
    for i in range(1, len(table2_data) + 1):
        for j in range(3):
            cell = table2[(i, j)]
            cell.set_facecolor('#f0f7f7' if i % 2 == 0 else 'white')
            cell.set_edgecolor('#d0d2d6')
    
    fig3.text(0.52, 0.89, f'Exhibit 8: GARP Q1 Sector Contribution ({q4_quarter})', fontsize=9, fontweight='bold')
    
    # Table 3: Low Risk Q1 Key Laggards (bottom left)
    ax3_attr = fig3.add_axes([0.06, 0.18, 0.42, 0.28])
    ax3_attr.axis('off')
    
    lr_bottom10 = lr_stock.head(10)
    table3_data = [[row['Symbol'], row['Sector'][:20], f"{row['Avg_Return']:+.1%}", row['Months']] 
                   for _, row in lr_bottom10.iterrows()]
    
    table3 = ax3_attr.table(
        cellText=table3_data,
        colLabels=['Stock', 'Sector', 'Avg Ret', 'Months in Q1'],
        colWidths=[0.12, 0.38, 0.15, 0.35],
        loc='upper center',
        cellLoc='left'
    )
    table3.auto_set_font_size(False)
    table3.set_fontsize(7)
    table3.scale(1, 1.4)
    for j in range(4):
        cell = table3[(0, j)]
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor('#006d77')
    for i in range(1, len(table3_data) + 1):
        for j in range(4):
            cell = table3[(i, j)]
            cell.set_facecolor('#f0f7f7' if i % 2 == 0 else 'white')
            cell.set_edgecolor('#d0d2d6')
    
    fig3.text(0.06, 0.47, f'Exhibit 9: Low Risk Q1 Key Laggards ({q4_quarter})', fontsize=9, fontweight='bold')
    
    # Table 4: Low Risk Q1 Sector Contribution (bottom right)
    ax4 = fig3.add_axes([0.52, 0.18, 0.42, 0.28])
    ax4.axis('off')
    
    table4_data = [[row['Sector'][:22], f"{row['Avg_Return']:+.1%}", str(int(row['Unique_Stocks']))] 
                   for _, row in lr_sector.iterrows()]
    
    table4 = ax4.table(
        cellText=table4_data,
        colLabels=['Sector', 'Avg Return', '# Stocks'],
        colWidths=[0.55, 0.25, 0.20],
        loc='upper center',
        cellLoc='left'
    )
    table4.auto_set_font_size(False)
    table4.set_fontsize(7)
    table4.scale(1, 1.4)
    for j in range(3):
        cell = table4[(0, j)]
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor('#006d77')
    for i in range(1, len(table4_data) + 1):
        for j in range(3):
            cell = table4[(i, j)]
            cell.set_facecolor('#f0f7f7' if i % 2 == 0 else 'white')
            cell.set_edgecolor('#d0d2d6')
    
    fig3.text(0.52, 0.47, f'Exhibit 10: Low Risk Q1 Sector Contribution ({q4_quarter})', fontsize=9, fontweight='bold')
    
    # Source footer
    fig3.text(0.06, 0.06, 'Source: FactSet, S&P 500 data. Sector avg return = simple average of all stock-month observations in Q1.', 
             fontsize=6, ha='left', style='italic')
    
    pdf.savefig(fig3)
    plt.close()
    print(" → Added Q4 attribution page")
    
    # ==========================================================================
    # FACTOR DEFINITIONS PAGE
    # ==========================================================================
    
    # Read factor definitions from Excel file
    factor_def_file = 'Factor_definitions.xlsx'
    if os.path.exists(factor_def_file):
        factor_def_df = pd.read_excel(factor_def_file)
        
        # Extract the relevant data (starting from row 5 where headers are)
        factor_data = []
        current_style = None
        for idx, row in factor_def_df.iterrows():
            style_val = row.get('Unnamed: 4', None)
            factor_val = row.get('Unnamed: 5', None)
            definition_val = row.get('Unnamed: 6', None)
            
            if pd.notna(style_val):
                current_style = style_val
            
            if pd.notna(factor_val) and pd.notna(definition_val):
                factor_data.append({
                    'Style': current_style if current_style != 'Styles' else '',
                    'Factor': factor_val if factor_val != 'Factors' else '',
                    'Definition': definition_val if definition_val != 'Style/factor definition' else ''
                })
        
        # Remove header row if captured
        factor_data = [d for d in factor_data if d['Factor'] != 'Factors' and d['Factor'] != '']
        
        if factor_data:
            fig_def = plt.figure(figsize=(8.27, 11.69))
            fig_def.text(0.06, 0.95, 'S&P 500 - Style and Factor Definitions', fontsize=14, fontweight='bold', 
                        ha='left', color='#006d77')
            
            # Create a table using matplotlib
            ax_table = fig_def.add_axes([0.06, 0.10, 0.88, 0.80])
            ax_table.axis('off')
            
            # Build table data with merged style cells
            table_data = []
            prev_style = None
            for d in factor_data:
                style_display = d['Style'] if d['Style'] != prev_style else ''
                table_data.append([style_display, d['Factor'], d['Definition']])
                prev_style = d['Style']
            
            # Create table
            col_widths = [0.12, 0.15, 0.73]
            table = ax_table.table(
                cellText=table_data,
                colLabels=['Style', 'Factor', 'Definition'],
                colWidths=col_widths,
                loc='upper center',
                cellLoc='left'
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            
            # Header styling
            for j in range(3):
                cell = table[(0, j)]
                cell.set_text_props(fontweight='bold', color='white')
                cell.set_facecolor('#006d77')
                cell.set_height(0.04)
            
            # Alternate row colors and style cells
            for i in range(1, len(table_data) + 1):
                for j in range(3):
                    cell = table[(i, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#f0f7f7')
                    else:
                        cell.set_facecolor('white')
                    cell.set_edgecolor('#d0d2d6')
                    
                    # Bold style names
                    if j == 0 and table_data[i-1][0] != '':
                        cell.set_text_props(fontweight='bold')
            
            # Source footer
            fig_def.text(0.06, 0.06, 'Source: FactSet', fontsize=6, ha='left', style='italic')
            
            pdf.savefig(fig_def)
            plt.close()
            print(" → Added factor definitions page")
    
    exhibit_num = 11  # Next exhibit starts at 11 (after 10 exhibits on summary + attribution pages)
    
    # ==========================================================================
    # FACTOR PAGES (one per factor)
    # ==========================================================================
    for factor in factors:
        # Use shared helper function (with shifted dates for display)
        monthly, universe, temp = prepare_factor_data(df, factor, shift_dates=True)
        
        if monthly is None:
            continue

        fig = plt.figure(figsize=(8.27, 11.69))
        gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.30,
                              top=0.89, bottom=0.08, left=0.12, right=0.95)
        if factor in mfr_names:
            page_title = f'{factor} — Style return summary'
        else:
            page_title = f'{factor} — Factor return summary'
        fig.text(0.06, 0.94, page_title, fontsize=12, fontweight='bold', 
                 ha='left', color='#006d77')

        # Chart 1: Cumulative outperformance
        cum_q = (1 + monthly).cumprod()
        cum_u = (1 + universe).cumprod()
        cum_excess = cum_q.sub(cum_u.values, axis=0)
        first_date = monthly.index.min() - pd.offsets.MonthEnd(1)
        start_row = pd.DataFrame([[0.0] * len(cum_excess.columns)], index=[first_date], columns=cum_excess.columns)
        cum_excess = pd.concat([start_row, cum_excess])

        ax1 = fig.add_subplot(gs[0, 0])
        x_dates = mdates.date2num(cum_excess.index.to_pydatetime())
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, col in enumerate(cum_excess.columns):
            ax1.plot(x_dates, cum_excess[col].values, linewidth=1.8, label=col, color=colors[i % len(colors)])
        ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax1.set_ylabel('(Equal-weighted index)', fontsize=6)
        handles, labels = ax1.get_legend_handles_labels()
        labels = ['Q1 (High)' if l == 'Q1' else 'Q5 (Low)' if l == 'Q5' else l for l in labels]
        ax1.legend(handles, labels, fontsize=6, loc='best', ncol=5, framealpha=0.9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax1.set_xlim(x_dates.min(), x_dates.max())
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
        ax1.tick_params(axis='x', rotation=90, labelsize=6)
        ax1.tick_params(axis='y', labelsize=6)
        ax1.grid(False)
        for spine in ax1.spines.values():
            spine.set_visible(False)
        pos1 = ax1.get_position()
        line_left = pos1.x0 - 0.06
        fig.add_artist(plt.Line2D([line_left, pos1.x1], [pos1.y1, pos1.y1], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([line_left, pos1.x1], [pos1.y0 - 0.04, pos1.y0 - 0.04], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.text(line_left, pos1.y1 + 0.005, f'Exhibit {exhibit_num}: {factor} — Q1 to Q5 Cumulative OPF', fontsize=8, fontweight='bold', ha='left')
        fig.text(line_left, pos1.y0 - 0.05, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')

        # Chart 2: Q1 OPF vs Q1-Q5 OPF
        ax2 = fig.add_subplot(gs[0, 1])
        q1_opf = cum_excess['Q1']
        q1_q5_opf = cum_excess['Q1'] - cum_excess['Q5']
        ax2.plot(x_dates, q1_opf.values, linewidth=1.8, label='Q1 OPF')
        ax2.plot(x_dates, q1_q5_opf.values, linewidth=1.8, label='Q1-Q5 OPF')
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_ylabel('(Equal-weighted index)', fontsize=6)
        ax2.legend(fontsize=6, loc='best', framealpha=0.9)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax2.set_xlim(x_dates.min(), x_dates.max())
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
        ax2.tick_params(axis='x', rotation=90, labelsize=6)
        ax2.tick_params(axis='y', labelsize=6)
        ax2.grid(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        pos2 = ax2.get_position()
        line_left2 = pos2.x0 - 0.06
        fig.add_artist(plt.Line2D([line_left2, pos2.x1], [pos2.y1, pos2.y1], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([line_left2, pos2.x1], [pos2.y0 - 0.04, pos2.y0 - 0.04], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.text(line_left2, pos2.y1 + 0.005, f'Exhibit {exhibit_num + 1}: Cumulative Q1 OPF vs Q1-Q5 OPF', fontsize=8, fontweight='bold', ha='left')
        fig.text(line_left2, pos2.y0 - 0.05, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')

        # Chart 3: Q1 vs Universe monthly OPF (last 12 months)
        ax3 = fig.add_subplot(gs[1, 0])
        last_12_q1_univ = (monthly['Q1'] - universe).tail(12)
        last_12_q1_univ.index = last_12_q1_univ.index.strftime("%b'%y")
        last_12_q1_univ.plot(kind='bar', ax=ax3, color='steelblue', width=0.8)
        ax3.axhline(0, color='black', linewidth=0.8)
        ax3.set_ylabel('Equal-weighted OPF (Q1-univ, %)', fontsize=6)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax3.tick_params(axis='x', rotation=90, labelsize=6)
        ax3.tick_params(axis='y', labelsize=6)
        ax3.set_xlabel('')
        ax3.grid(False)
        y_min, y_max = last_12_q1_univ.min(), last_12_q1_univ.max()
        padding = max(abs(y_max), abs(y_min)) * 0.15
        ax3.set_ylim(y_min - padding if y_min < 0 else -padding * 0.1, y_max + padding if y_max > 0 else padding * 0.1)
        for bar, val in zip(ax3.patches, last_12_q1_univ.values):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = padding * 0.1 if height >= 0 else -padding * 0.1
            ax3.text(bar.get_x() + bar.get_width()/2, height + offset, f'{val:.1%}', ha='center', va=va, fontsize=6)
        for spine in ax3.spines.values():
            spine.set_visible(False)
        pos3 = ax3.get_position()
        line_left3 = pos3.x0 - 0.06
        fig.add_artist(plt.Line2D([line_left3, pos3.x1], [pos3.y1, pos3.y1], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([line_left3, pos3.x1], [pos3.y0 - 0.04, pos3.y0 - 0.04], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.text(line_left3, pos3.y1 + 0.005, f'Exhibit {exhibit_num + 2}: Q1 vs Universe — Monthly OPF (Last 12M)', fontsize=8, fontweight='bold', ha='left')
        fig.text(line_left3, pos3.y0 - 0.05, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')

        # Chart 4: Q1 vs Q5 monthly OPF (last 12 months)
        ax4 = fig.add_subplot(gs[1, 1])
        last_12_q1_q5 = (monthly['Q1'] - monthly['Q5']).tail(12)
        last_12_q1_q5.index = last_12_q1_q5.index.strftime("%b'%y")
        last_12_q1_q5.plot(kind='bar', ax=ax4, color='darkorange', width=0.8)
        ax4.axhline(0, color='black', linewidth=0.8)
        ax4.set_ylabel('Equal-weighted OPF (Q1-Q5, %)', fontsize=6)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax4.tick_params(axis='x', rotation=90, labelsize=6)
        ax4.tick_params(axis='y', labelsize=6)
        ax4.set_xlabel('')
        ax4.grid(False)
        y_min4, y_max4 = last_12_q1_q5.min(), last_12_q1_q5.max()
        padding4 = max(abs(y_max4), abs(y_min4)) * 0.15
        ax4.set_ylim(y_min4 - padding4 if y_min4 < 0 else -padding4 * 0.1, y_max4 + padding4 if y_max4 > 0 else padding4 * 0.1)
        for bar, val in zip(ax4.patches, last_12_q1_q5.values):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = padding4 * 0.1 if height >= 0 else -padding4 * 0.1
            ax4.text(bar.get_x() + bar.get_width()/2, height + offset, f'{val:.1%}', ha='center', va=va, fontsize=6)
        for spine in ax4.spines.values():
            spine.set_visible(False)
        pos4 = ax4.get_position()
        line_left4 = pos4.x0 - 0.06
        fig.add_artist(plt.Line2D([line_left4, pos4.x1], [pos4.y1, pos4.y1], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([line_left4, pos4.x1], [pos4.y0 - 0.04, pos4.y0 - 0.04], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.text(line_left4, pos4.y1 + 0.005, f'Exhibit {exhibit_num + 3}: Q1 vs Q5 — Monthly OPF (Last 12M)', fontsize=8, fontweight='bold', ha='left')
        fig.text(line_left4, pos4.y0 - 0.05, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')

        # Chart 5: Hit rate (% of Q1 stocks outperforming)
        ax5 = fig.add_subplot(gs[2, 0])
        q1_stocks = temp[temp['Group'] == 'Q1'].copy()
        q1_stocks = q1_stocks.merge(universe.rename('Univ_Ret').reset_index(), on='Date')
        q1_stocks['Beat_Univ'] = q1_stocks['Returns'] > q1_stocks['Univ_Ret']
        hit_rate_by_month = q1_stocks.groupby('Date')['Beat_Univ'].mean() * 100
        hit_rate_12m = hit_rate_by_month.tail(12)
        hit_rate_12m.index = hit_rate_12m.index.strftime("%b'%y")
        hit_rate_12m.plot(kind='bar', ax=ax5, color='darkseagreen', width=0.8)
        ax5.set_ylabel('(%)', fontsize=6)
        ax5.set_ylim(0, 115)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax5.tick_params(axis='x', rotation=90, labelsize=6)
        ax5.tick_params(axis='y', labelsize=6)
        ax5.set_xlabel('')
        ax5.grid(False)
        ax5.axhline(50, color='black', linewidth=0.8, linestyle='--')
        for bar, val in zip(ax5.patches, hit_rate_12m.values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.0f}%', ha='center', va='bottom', fontsize=6)
        for spine in ax5.spines.values():
            spine.set_visible(False)
        pos5 = ax5.get_position()
        line_left5 = pos5.x0 - 0.06
        fig.add_artist(plt.Line2D([line_left5, pos5.x1], [pos5.y1, pos5.y1], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([line_left5, pos5.x1], [pos5.y0 - 0.04, pos5.y0 - 0.04], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.text(line_left5, pos5.y1 + 0.005, f'Exhibit {exhibit_num + 4}: % of Q1 stocks outperforming', fontsize=8, fontweight='bold', ha='left')
        fig.text(line_left5, pos5.y0 - 0.05, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')

        # Chart 6: Q1 vs Q5 scatter
        ax6 = fig.add_subplot(gs[2, 1])
        q1_ret_12m = monthly['Q1'].tail(12)
        q5_ret_12m = monthly['Q5'].tail(12)
        ax6.scatter(q5_ret_12m.values, q1_ret_12m.values, color='purple', s=40, alpha=0.8)
        ax6.set_xlabel('Q5 Monthly Returns (%)', fontsize=6)
        ax6.set_ylabel('Q1 Monthly Returns (%)', fontsize=6)
        ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax6.tick_params(axis='x', labelsize=6)
        ax6.tick_params(axis='y', labelsize=6)
        all_vals = list(q1_ret_12m.values) + list(q5_ret_12m.values)
        axis_min = min(all_vals) - 0.01
        axis_max = max(all_vals) + 0.01
        ax6.set_xlim(axis_min, axis_max)
        ax6.set_ylim(axis_min, axis_max)
        ax6.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', linewidth=0.8, alpha=0.6)
        ax6.set_aspect('equal', adjustable='box')
        ax6.grid(False)
        for spine in ax6.spines.values():
            spine.set_visible(False)
        pos6 = ax6.get_position()
        line_left6 = pos6.x0 - 0.06
        fig.add_artist(plt.Line2D([line_left6, pos6.x1], [pos6.y1, pos6.y1], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([line_left6, pos6.x1], [pos6.y0 - 0.04, pos6.y0 - 0.04], color='#d0d2d6', linewidth=0.8, transform=fig.transFigure))
        fig.text(line_left6, pos6.y1 + 0.005, f'Exhibit {exhibit_num + 5}: Q1 vs Q5 Monthly Returns (Last 12M)', fontsize=8, fontweight='bold', ha='left')
        fig.text(line_left6, pos6.y0 - 0.05, 'Source: FactSet, S&P 500 data', fontsize=6, ha='left', style='italic')

        pdf.savefig(fig)
        plt.close()
        print(f" → Added page for: {factor}")
        exhibit_num += 6

print("PDF report saved: Factor_Backtest_Report.pdf")
print("Done! Open 'Backtest_Results.xlsx'")
