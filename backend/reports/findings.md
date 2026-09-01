# NEXYGEN EDA Findings

## Trend

- **Scope 1**: decreased 6.3% from 2020 to 2025
- **Scope 2**: decreased 37.6% from 2020 to 2025

## Seasonality

- **Scope 1**: peaks in month 6, troughs in month 12 -- strong seasonality (peak-to-trough swing = 18.1% of the mean)
- **Scope 2**: peaks in month 6, troughs in month 12 -- negligible seasonality (peak-to-trough swing = 1.4% of the mean)

## Anomalies

Detected on the residual after removing a 30-day rolling-median local baseline (not raw z-score against the whole series), so trend drift isn't mistaken for a local anomaly.
- 133 day(s) flagged with |residual z-score| > 2.0
- **Scope 1**: 58 anomalous day(s), e.g. 2020-02-16 (z=-2.35)
- **Scope 2**: 75 anomalous day(s), e.g. 2020-01-11 (z=-2.94)

## Driver Analysis

Random Forest feature importance (aggregated to parent driver), fit on Asset_Type, Location, Emission_Type, Energy_Type:
- **Asset_Type**: 0.588
- **Location**: 0.249
- **Emission_Type**: 0.162
- **Energy_Type**: 0.000
- Excluded (zero variance in this dataset): Operational_Status
- Model R^2 on training data: 0.532 (in-sample fit quality of the importance-ranking model itself, not a forecast metric)

Highest mean-emissions asset type: **ControlRoom** (1.604 tCO2e/row average).