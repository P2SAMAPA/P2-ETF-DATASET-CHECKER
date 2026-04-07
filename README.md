# 📊 P2 ETF Dataset Monitor

Automated monitoring system for 13 Hugging Face ETF datasets with mixed formats (Parquet/CSV) and folder structures.

## 🎯 Features

- ✅ **Manual GitHub Actions workflow** - Run checks on demand
- 📊 **Streamlit Dashboard** - Real-time visualization of dataset health
- 🔍 **Smart File Detection** - Handles both root and `/data` folder structures
- 📁 **Multi-Format Support** - Parquet and CSV files
- ⏰ **Data Freshness Checks** - Validates recent data availability
- 📈 **Historical Trends** - Track dataset health over time
- 🚨 **Issue Detection** - Missing columns, rows, files, and stale data

## 📁 Repository Structure
P2-ETF-DATASET-CHECKER/
├── .github/
│   └── workflows/
│       └── check-datasets.yml      # GitHub Actions workflow (manual trigger)
├── .streamlit/
│   └── config.toml                 # Streamlit theme config
├── data/                           # Check results (auto-generated)
│   ├── latest_check.json           # Most recent results
│   └── check_YYYYMMDD_HHMMSS.json  # Historical snapshots
├── app.py                          # Streamlit dashboard
├── check_datasets.py               # Validation script
├── datasets_config.json            # Dataset configurations
├── requirements.txt                # Python dependencies
└── README.md                       # This file


## 🚀 Setup

### 1. Repository Setup
```bash
git clone https://github.com/P2SAMAPA/P2-ETF-DATASET-CHECKER.git
cd P2-ETF-DATASET-CHECKER
pip install -r requirements.txt
