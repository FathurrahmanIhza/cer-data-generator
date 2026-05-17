# CER Data Generator

A Python-based simulation and analysis tool for Consumer Energy Resources (CER), focusing on rooftop PV, battery storage, and Virtual Power Plant (VPP) operation under market conditions.

This project is designed to generate and analyse residential energy telemetry data, evaluate battery dispatch behaviour, and assess the operational and financial impacts of VPP participation.

---

## Features

- Rooftop PV generation simulation
- Household load profile modelling
- Battery charge/discharge operation
- Grid import/export calculation
- Wholesale electricity price integration
- VPP dispatch simulation
- VPP export revenue analysis
- Additional import cost estimation after VPP dispatch
- Monthly and yearly electricity bill comparison
- Self-consumption and self-sufficiency analysis
- Battery dispatch limitation analysis
- Heatmap and operational visualisation tools

---

## Project Structure

```bash
cer-data-generator/
│
├── dataset/               # Input datasets
├── modules/               # Core simulation modules
├── .devcontainer/         # Dev container configuration
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Simulation Scope

The model simulates:

- Solar PV generation
- Household electricity demand
- Battery energy storage operation
- Grid interaction
- VPP dispatch participation based on wholesale price thresholds

The framework can be used to evaluate:

- Battery operational behaviour
- Financial savings
- Electricity bill reduction
- Grid import/export patterns
- VPP operational impacts on household energy autonomy

---

## Main Analysis Outputs

### Battery Operation Analysis

- Battery SOC behaviour
- Charge/discharge operation
- VPP dispatch events
- Dispatch limitation detection

### Financial Analysis

- Actual household electricity bill
- PV + battery without VPP approximation
- Solar-only scenario
- Grid-only scenario
- VPP operational revenue
- Additional import costs after dispatch

### Visualisations

- Monthly electricity bill comparison
- Yearly electricity bill comparison
- VPP dispatch heatmaps
- Battery operation time-series plots
- Requested vs actual VPP dispatch scatter plots
- Self-consumption and self-sufficiency analysis

---
