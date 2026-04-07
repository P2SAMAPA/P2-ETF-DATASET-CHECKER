#!/usr/bin/env python3
"""
HF Dataset Checker for P2SAMAPA ETF Datasets - DEBUG VERSION
"""

import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
import warnings
warnings.filterwarnings('ignore')

# Import NYSE calendar
try:
    import exchange_calendars as xcals
    from pandas_market_calendars import get_calendar
    NYSE_CALENDAR_AVAILABLE = True
except ImportError:
    NYSE_CALENDAR_AVAILABLE = False
    print("Warning: exchange-calendars not installed")

DATASETS_CONFIG = [
    {"name": "P2SAMAPA/p2-etf-regime-predictor", "date_column": "date", "min_rows": 100},
    {"name": "P2SAMAPA/P2-ETF-DQN-ENGINE-DATASET", "date_column": None, "min_rows": 1000},
    {"name": "P2SAMAPA/p2-etf-deepwave-dl", "date_column": "date", "min_rows": 500},
    {"name": "P2SAMAPA/my-etf-data", "date_column": "date", "min_rows": 1000},
    {"name": "P2SAMAPA/etf-entropy-dataset", "date_column": "date", "min_rows": 100},
    {"name": "P2SAMAPA/p2-etf-hurst-data", "date_column": "date", "min_rows": 100},
    {"name": "P2SAMAPA/p2-etf-merton-ann-data", "date_column": "date", "min_rows": 100},
    {"name": "P2SAMAPA/etf-dlinear-cross-data", "date_column": "date", "min_rows": 500},
    {"name": "P2SAMAPA/p2-etf-trendfolios-replication-data", "date_column": "date", "min_rows": 200},
    {"name": "P2SAMAPA/etf_trend_data", "date_column": "date", "min_rows": 1000},
    {"name": "P2SAMAPA/fi-etf-macro-signal-master-data", "date_column": "date", "min_rows": 500},
    {"name": "P2SAMAPA/p2-etf-deepm-data", "date_column": "date", "min_rows": 500},
    {"name": "P2SAMAPA/p2-etf-momentum-maxima", "date_column": "date", "min_rows": 100},
]

class NYSEMarketCalendar:
    def __init__(self):
        self.calendar = None
        if NYSE_CALENDAR_AVAILABLE:
            try:
                self.calendar = get_calendar('NYSE')
            except:
                try:
                    self.calendar = xcals.get_calendar("XNYS")
                except:
                    pass
    
    def get_last_trading_day(self, date=None):
        if date is None:
            date = pd.Timestamp.now()
        if self.calendar is None:
            return date
        check_date = date
        for _ in range(10):
            try:
                schedule = self.calendar.schedule(start_date=check_date, end_date=check_date)
                if len(schedule) > 0:
                    return check_date
            except:
                pass
            check_date -= timedelta(days=1)
        return date

nyse_calendar = NYSEMarketCalendar()

def check_dataset(config):
    """Check a single dataset with full error details"""
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"🔍 Checking: {name}")
    print('='*60)
    
    result = {
        "name": name,
        "checked_at": datetime.utcnow().isoformat(),
        "status": "unknown",
        "health": "unknown",
        "issues": [],
        "warnings": [],
        "stats": {},
        "files_found": [],
        "debug_info": {}
    }
    
    try:
        # Step 1: List files
        print("Step 1: Listing repository files...")
        try:
            repo_files = list_repo_files(name)
            print(f"  ✓ Found {len(repo_files)} total files")
            result["debug_info"]["total_files"] = len(repo_files)
            result["debug_info"]["sample_files"] = repo_files[:5]
        except Exception as e:
            print(f"  ✗ ERROR listing files: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            result["issues"].append(f"Cannot list files: {str(e)}")
            result["health"] = "critical"
            return result
        
        # Step 2: Find data files
        print("Step 2: Looking for data files...")
        data_files = [f for f in repo_files if f.endswith(('.parquet', '.csv', '.json'))]
        data_files = sorted(data_files, key=lambda x: (0 if x.endswith('.parquet') else 1 if x.endswith('.csv') else 2))
        
        print(f"  Found {len(data_files)} data files: {data_files[:3]}")
        result["files_found"] = data_files
        result["debug_info"]["data_files_count"] = len(data_files)
        
        if not data_files:
            print("  ✗ ERROR: No data files found!")
            result["issues"].append("No .parquet, .csv, or .json files found")
            result["health"] = "critical"
            return result
        
        # Step 3: Download file
        target_file = data_files[0]
        print(f"Step 3: Downloading {target_file}...")
        
        try:
            local_path = hf_hub_download(
                repo_id=name,
                filename=target_file,
                local_dir="temp_downloads",
                local_dir_use_symlinks=False
            )
            print(f"  ✓ Downloaded to: {local_path}")
            result["debug_info"]["downloaded_file"] = local_path
        except Exception as e:
            print(f"  ✗ ERROR downloading: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            result["issues"].append(f"Download failed: {str(e)}")
            result["health"] = "critical"
            return result
        
        # Step 4: Read file
        print("Step 4: Reading file...")
        file_ext = Path(target_file).suffix
        
        try:
            if file_ext == '.parquet':
                df = pd.read_parquet(local_path)
            elif file_ext == '.csv':
                df = pd.read_csv(local_path)
            elif file_ext == '.json':
                try:
                    df = pd.read_json(local_path, lines=True)
                except:
                    df = pd.read_json(local_path)
            else:
                raise ValueError(f"Unknown extension: {file_ext}")
            
            print(f"  ✓ Loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"  Columns: {list(df.columns)[:5]}...")
            
        except Exception as e:
            print(f"  ✗ ERROR reading file: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            result["issues"].append(f"Cannot read file: {str(e)}")
            result["health"] = "critical"
            return result
        
        # Step 5: Basic validation
        print("Step 5: Validating data...")
        result["stats"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "file_format": file_ext.replace('.', ''),
            "source_file": target_file
        }
        
        if df.empty:
            print("  ✗ WARNING: Dataset is empty!")
            result["issues"].append("Dataset is empty")
        
        if len(df) < config.get("min_rows", 0):
            print(f"  ✗ WARNING: Row count {len(df)} < minimum {config['min_rows']}")
            result["issues"].append(f"Row count ({len(df)}) below minimum ({config['min_rows']})")
        
        # Step 6: Date column check
        if config.get("date_column"):
            print(f"Step 6: Checking date column '{config['date_column']}'...")
            if config["date_column"] not in df.columns:
                print(f"  ✗ WARNING: Date column not found!")
                result["warnings"].append(f"Date column '{config['date_column']}' not found")
                result["stats"]["available_columns"] = list(df.columns)
            else:
                try:
                    dates = pd.to_datetime(df[config["date_column"]], errors='coerce')
                    max_date = dates.max()
                    print(f"  ✓ Date range: {dates.min()} to {max_date}")
                    result["stats"]["last_date"] = str(max_date)
                    
                    # Check freshness
                    last_trading = nyse_calendar.get_last_trading_day()
                    days_diff = (last_trading - max_date).days
                    print(f"  Last trading day: {last_trading}, Days behind: {days_diff}")
                    
                except Exception as e:
                    print(f"  ✗ ERROR parsing dates: {str(e)}")
                    result["warnings"].append(f"Date parsing error: {str(e)}")
        
        # Determine final health
        if result["issues"]:
            result["health"] = "warning"  # Downgrade to warning if we at least got the file
            result["status"] = "issues_found"
        else:
            result["health"] = "healthy"
            result["status"] = "ok"
            print("  ✓✓✓ Dataset is HEALTHY")
        
    except Exception as e:
        print(f"✗✗✗ UNEXPECTED ERROR: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        result["issues"].append(f"Unexpected error: {str(e)}")
        result["health"] = "critical"
    
    return result

def main():
    print("="*70)
    print("P2SAMAPA ETF Dataset Checker - DEBUG MODE")
    print(f"Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*70)
    
    results = []
    
    for config in DATASETS_CONFIG:
        result = check_dataset(config)
        results.append(result)
    
    # Summary
    healthy = sum(1 for r in results if r["health"] == "healthy")
    warning = sum(1 for r in results if r["health"] == "warning")
    critical = sum(1 for r in results if r["health"] == "critical")
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"  Healthy:   {healthy}")
    print(f"  Warning:   {warning}")
    print(f"  Critical:  {critical}")
    
    # Show critical ones
    if critical > 0:
        print("\n❌ CRITICAL DATASETS:")
        for r in results:
            if r["health"] == "critical":
                print(f"  • {r['name']}")
                for issue in r.get("issues", []):
                    print(f"    - {issue}")
    
    # Save results
    Path("data").mkdir(exist_ok=True)
    output = {
        "check_time": datetime.utcnow().isoformat(),
        "total_datasets": len(results),
        "summary": {"healthy": healthy, "warning": warning, "critical": critical},
        "datasets": results
    }
    
    with open("data/latest_check.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to data/latest_check.json")
    
    if critical > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
