#!/usr/bin/env python3
"""
HF Dataset Checker for P2SAMAPA ETF Datasets
Validates 13 datasets with mixed formats (Parquet/CSV) and folder structures
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Dataset configurations - 13 P2SAMAPA ETF datasets
DATASETS_CONFIG = [
    {
        "name": "P2SAMAPA/p2-etf-regime-predictor",
        "files": ["regime_predictor_data.parquet"],
        "expected_columns": ["date", "ticker", "regime", "features"],
        "date_column": "date",
        "min_rows": 100,
        "type": "parquet"
    },
    {
        "name": "P2SAMAPA/P2-ETF-DQN-ENGINE-DATASET",
        "files": ["dqn_data.parquet", "data/dqn_data.parquet"],
        "expected_columns": ["state", "action", "reward", "next_state"],
        "date_column": None,
        "min_rows": 1000,
        "type": "parquet"
    },
    {
        "name": "P2SAMAPA/p2-etf-deepwave-dl",
        "files": ["deepwave_data.parquet", "data/deepwave_data.parquet"],
        "expected_columns": ["date", "ticker", "wave_features", "target"],
        "date_column": "date",
        "min_rows": 500,
        "type": "parquet"
    },
    {
        "name": "P2SAMAPA/my-etf-data",
        "files": ["etf_data.csv", "data/etf_data.csv", "etf_data.parquet"],
        "expected_columns": ["date", "ticker", "open", "high", "low", "close", "volume"],
        "date_column": "date",
        "min_rows": 1000,
        "type": "mixed"
    },
    {
        "name": "P2SAMAPA/etf-entropy-dataset",
        "files": ["entropy_data.parquet", "data/entropy_data.parquet"],
        "expected_columns": ["date", "ticker", "entropy_value", "window"],
        "date_column": "date",
        "min_rows": 100,
        "type": "parquet"
    },
    {
        "name": "P2SAMAPA/p2-etf-hurst-data",
        "files": ["hurst_data.parquet", "data/hurst_data.parquet"],
        "expected_columns": ["date", "ticker", "hurst_exponent", "rs_value"],
        "date_column": "date",
        "min_rows": 100,
        "type": "parquet"
    },
    {
        "name": "P2SAMAPA/p2-etf-merton-ann-data",
        "files": ["merton_data.parquet", "data/merton_data.parquet"],
        "expected_columns": ["date", "ticker", "jump_intensity", "jump_size", "diffusion"],
        "date_column": "date",
        "min_rows": 100,
        "type": "parquet"
    },
    {
        "name": "P2SAMAPA/etf-dlinear-cross-data",
        "files": ["dlinear_data.parquet", "data/dlinear_data.parquet"],
        "expected_columns": ["date", "ticker", "seasonal", "trend", "residual"],
        "date_column": "date",
        "min_rows": 500,
        "type": "parquet"
    },
    {
        "name": "P2SAMAPA/p2-etf-trendfolios-replication-data",
        "files": ["trendfolios_data.parquet", "data/trendfolios_data.parquet"],
        "expected_columns": ["date", "ticker", "momentum_score", "allocation"],
        "date_column": "date",
        "min_rows": 200,
        "type": "parquet"
    },
    {
        "name": "P2SAMAPA/etf_trend_data",
        "files": ["trend_data.csv", "data/trend_data.csv", "trend_data.parquet"],
        "expected_columns": ["date", "ticker", "trend_signal", "strength"],
        "date_column": "date",
        "min_rows": 1000,
        "type": "mixed"
    },
    {
        "name": "P2SAMAPA/fi-etf-macro-signal-master-data",
        "files": ["macro_data.parquet", "data/macro_data.parquet", "macro_data.csv"],
        "expected_columns": ["date", "signal_type", "value", "etf_ticker"],
        "date_column": "date",
        "min_rows": 500,
        "type": "mixed"
    },
    {
        "name": "P2SAMAPA/p2-etf-deepm-data",
        "files": ["deepm_data.parquet", "data/deepm_data.parquet"],
        "expected_columns": ["date", "ticker", "deep_features", "prediction"],
        "date_column": "date",
        "min_rows": 500,
        "type": "parquet"
    },
    {
        "name": "P2SAMAPA/p2-etf-momentum-maxima",
        "files": ["momentum_maxima.parquet", "data/momentum_maxima.parquet"],
        "expected_columns": ["date", "ticker", "momentum", "local_maxima", "window"],
        "date_column": "date",
        "min_rows": 100,
        "type": "parquet"
    }
]

def check_freshness(df, date_column, max_days=7):
    """Check if dataset has recent data (within last 7 days by default)"""
    if date_column not in df.columns:
        return False, f"Date column '{date_column}' not found", None
    
    try:
        # Try to parse date column
        dates = pd.to_datetime(df[date_column], errors='coerce')
        max_date = dates.max()
        min_date = dates.min()
        
        if pd.isna(max_date):
            return False, "No valid dates found", None
        
        days_old = (datetime.now() - max_date).days
        
        if days_old > max_days:
            return False, f"Data is {days_old} days old (max allowed: {max_days})", max_date
        
        return True, f"Fresh (last date: {max_date.strftime('%Y-%m-%d')})", max_date
        
    except Exception as e:
        return False, f"Date parsing error: {str(e)}", None

def check_dataset(config, filter_name=None):
    """Check a single dataset"""
    name = config["name"]
    
    if filter_name and name != filter_name:
        return None
    
    print(f"\n🔍 Checking {name}...")
    
    result = {
        "name": name,
        "checked_at": datetime.utcnow().isoformat(),
        "status": "unknown",
        "health": "unknown",
        "issues": [],
        "warnings": [],
        "stats": {},
        "files_found": [],
        "files_missing": []
    }
    
    api = HfApi()
    repo_files = []
    
    try:
        # List all files in the repo
        repo_files = list_repo_files(name, token=os.environ.get("HF_TOKEN"))
        result["repo_files"] = repo_files
    except Exception as e:
        result["status"] = "error"
        result["issues"].append(f"Cannot access repository: {str(e)}")
        result["health"] = "critical"
        return result
    
    # Find which files from config actually exist
    target_files = []
    for expected_file in config["files"]:
        if expected_file in repo_files:
            target_files.append(expected_file)
            result["files_found"].append(expected_file)
    
    if not target_files:
        # Try to find any parquet or csv files
        auto_detected = [f for f in repo_files if f.endswith(('.parquet', '.csv'))]
        if auto_detected:
            target_files = auto_detected[:1]  # Take first found
            result["warnings"].append(f"Using auto-detected file: {target_files[0]}")
            result["files_found"] = target_files
        else:
            result["status"] = "error"
            result["issues"].append(f"No data files found. Expected one of: {config['files']}")
            result["health"] = "critical"
            return result
    
    result["files_missing"] = list(set(config["files"]) - set(result["files_found"]))
    if result["files_missing"]:
        result["warnings"].append(f"Some expected files missing: {result['files_missing']}")
    
    # Download and validate the first available file
    target_file = target_files[0]
    local_path = None
    
    try:
        local_path = hf_hub_download(
            repo_id=name,
            filename=target_file,
            token=os.environ.get("HF_TOKEN"),
            local_dir="temp_downloads"
        )
        
        # Read file based on extension
        if target_file.endswith('.parquet'):
            df = pd.read_parquet(local_path)
        elif target_file.endswith('.csv'):
            df = pd.read_csv(local_path)
        else:
            raise ValueError(f"Unsupported file format: {target_file}")
        
        # Basic stats
        result["stats"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "file_size_mb": round(os.path.getsize(local_path) / (1024 * 1024), 2),
            "file_format": "parquet" if target_file.endswith('.parquet') else "csv"
        }
        
        # Check row count
        if len(df) < config.get("min_rows", 0):
            result["issues"].append(
                f"Row count ({len(df)}) below minimum ({config['min_rows']})"
            )
        
        # Check columns
        if "expected_columns" in config:
            missing_cols = set(config["expected_columns"]) - set(df.columns)
            extra_cols = set(df.columns) - set(config["expected_columns"])
            
            if missing_cols:
                result["issues"].append(f"Missing columns: {list(missing_cols)}")
            if extra_cols:
                result["warnings"].append(f"Extra columns found: {list(extra_cols)}")
        
        # Check for empty data
        if df.empty:
            result["issues"].append("Dataset is completely empty")
        
        # Check for null values
        null_counts = df.isnull().sum()
        high_null_cols = null_counts[null_counts > len(df) * 0.1]  # >10% null
        if len(high_null_cols) > 0:
            result["warnings"].append(
                f"Columns with >10% null values: {dict(high_null_cols)}"
            )
        
        # Check data freshness if enabled and date column exists
        check_fresh = os.environ.get("CHECK_FRESHNESS", "true").lower() == "true"
        if check_fresh and config.get("date_column"):
            is_fresh, freshness_msg, last_date = check_freshness(
                df, config["date_column"]
            )
            result["stats"]["last_date"] = str(last_date) if last_date else None
            result["stats"]["data_freshness"] = freshness_msg
            
            if not is_fresh:
                result["issues"].append(f"Data freshness issue: {freshness_msg}")
        
        # Determine health
        if result["issues"]:
            result["health"] = "critical" if any("error" in i.lower() or "empty" in i.lower() for i in result["issues"]) else "warning"
            result["status"] = "issues_found"
        else:
            result["health"] = "healthy"
            result["status"] = "ok"
            
        # Check for duplicates
        if len(df) > 0 and df.duplicated().any():
            dup_count = df.duplicated().sum()
            result["warnings"].append(f"Found {dup_count} duplicate rows")
        
        print(f"  ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
    except Exception as e:
        result["status"] = "error"
        result["issues"].append(f"Failed to process file {target_file}: {str(e)}")
        result["health"] = "critical"
        print(f"  ❌ Error: {str(e)}")
    
    finally:
        # Cleanup temp file
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except:
                pass
    
    return result

def main():
    """Main execution"""
    filter_name = os.environ.get("DATASET_FILTER", "").strip()
    
    print("=" * 60)
    print("P2SAMAPA ETF Dataset Checker")
    print(f"Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if filter_name:
        print(f"Filter: Checking only {filter_name}")
    print("=" * 60)
    
    results = []
    
    for config in DATASETS_CONFIG:
        result = check_dataset(config, filter_name if filter_name else None)
        if result:
            results.append(result)
    
    # Summary
    healthy = sum(1 for r in results if r["health"] == "healthy")
    warning = sum(1 for r in results if r["health"] == "warning")
    critical = sum(1 for r in results if r["health"] == "critical")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Healthy:   {healthy}")
    print(f"  Warning:   {warning}")
    print(f"  Critical:  {critical}")
    print("=" * 60)
    
    # Save results
    output = {
        "check_time": datetime.utcnow().isoformat(),
        "total_datasets": len(results),
        "summary": {
            "healthy": healthy,
            "warning": warning,
            "critical": critical
        },
        "datasets": results
    }
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    # Save latest
    with open("data/latest_check.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    # Save historical copy
    history_file = f"data/check_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to:")
    print(f"  - data/latest_check.json")
    print(f"  - {history_file}")
    
    # Exit with error code if any critical issues
    if critical > 0:
        print(f"\n❌ {critical} dataset(s) have critical issues!")
        sys.exit(1)
    elif warning > 0:
        print(f"\n⚠️  {warning} dataset(s) have warnings.")
        sys.exit(0)
    else:
        print(f"\n✅ All datasets healthy!")
        sys.exit(0)

if __name__ == "__main__":
    main()
