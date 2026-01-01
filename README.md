MAD-Velocity

MAD-Velocity is a systematic momentum trading framework built on Moving Average Distance (MAD) and its first- and second-order dynamics. The project provides a transparent, interpretable alternative to classical trend-following rules by explicitly modeling both the magnitude and the dynamics of price deviations from trend.

This repository contains the full research codebase supporting the paper:

MAD-Velocity: A Novel Technical Indicator and Momentum-Driven Trading Strategy
Lawson Arrington (2026)

Overview

Traditional moving-average–based momentum strategies often lag at turning points and are sensitive to short-lived price noise. MAD-Velocity addresses this limitation by:

Measuring signed distance from trend (MAD)

Modeling the velocity and acceleration of that distance

Discretizing MAD into standardized regimes

Applying rule-based, long-only trading logic grounded in regime transitions

The result is a fully specified, mechanically executable strategy that captures persistent momentum while managing downside risk.

Key Features

Moving Average Distance (MAD)
Standardized deviation of price from its rolling moving average

MAD Velocity & Acceleration
First- and second-order derivatives used to identify momentum inflection points

Regime-Based Framework
MAD discretized into σ-based regimes and modeled via a Markov process

Deterministic Trading Rules
Long-only entries and exits defined explicitly by regime transitions

Reproducible Backtests & Simulations
No parameter fitting, no machine learning, no optimization artifacts

Data

Equity Data:
Daily OHLCV data sourced via yfinance

Universe:
Continuous S&P 500 constituents (2010–2020), reconstructed using historical membership data
Source: https://github.com/fja05680/sp500

Processed Outputs:
Backtest results, simulations, and derived datasets are hosted in a public S3 bucket:
https://mad-velocity-backtest-data.s3.us-east-1.amazonaws.com


Results Summary

Consistent performance across a broad cross-section of S&P 500 equities

Median individual-security equity approximately doubles over 2010–2020

Portfolio simulations achieve ~11–12% CAGR with controlled drawdowns

Forward simulations demonstrate scalability to concentrated, leveraged deployment

High signal-to-profit ratios with moderate trade frequency

Full empirical results are documented in the accompanying paper.

Reproducibility

All experiments are:

Deterministic

Fully specified

Free of look-ahead bias

Reproducible using publicly available data

No proprietary data or closed-source dependencies are required.

Disclaimer

This project is provided for research and educational purposes only.
It does not constitute financial advice, investment recommendations, or an offer to trade securities.
Past performance does not guarantee future results.

MAD-Velocity is an independent, self-directed research project developed exclusively on personal time and is not affiliated with or endorsed by any employer or government entity.



