# S&P 500 Cross-Sectional Factor Backtesting System

A quantitative backtesting framework that evaluates investment styles and factors by ranking S&P 500 stocks into quintiles (Q1-Q5) and measuring their long-short excess returns.

## Features

- **8 Styles & 22 Factors Tested**: Value, Growth, Yield, Revision, Momentum, Quality, GARP, Low Risk, and individual metrics
- **Multi-Factor Ratings (MFRs)**: Composite scores combining related winsorized factor z-scores
- **PDF Reports**: 
  - Summary pages with style performance charts
  - Q4 attribution analysis (top performers/laggards by style)
  - Factor definitions reference
  - Individual factor performance pages
- **Excel Output**: Detailed quintile returns, hit rates, Sharpe ratios, and drawdown metrics
- **Optional AI Commentary**: OpenAI integration for automated write-ups (requires API key)

## Requirements

```
pip install pandas numpy matplotlib openpyxl openai
```

## Usage

1. Place your stock data in `data/SP500_factor_data.csv`
2. Run `python backtest.py`
3. Outputs: `Factor_Backtest_Report.pdf` and `Backtest_Results.xlsx`

## Configuration

Key settings in `backtest.py` (search for these variables):
- `USE_AI_WRITEUPS_PAGE1` / `USE_AI_WRITEUPS_PAGE2`: Toggle AI-generated write-ups on/off
- `writeup = f"""...`: Custom Page 1 commentary
- `page2_writeup = f"""...`: Custom Page 2 commentary
- `q4_raw_dates`: Date range for Q4 attribution analysis

## Methodology

- Backtest is done based on S&P 500 universe with at least 3 analyst coverage
- Stocks are ranked monthly by factor value, and are divided into quintiles
- Q1 = highest-ranked (best), Q5 = lowest-ranked
- Equal-weighted US-dollar total returns are calculated for each quintile
