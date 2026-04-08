#!/usr/bin/env python3
"""P2 ETF Dataset Checker for P2SAMAPA ETF Datasets with NYSE Market Calendar Support
Validates 13 datasets considering US market holidays and weekends

Fixed: Skip corrupted parquet files and try alternative files in the dataset
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
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

# Minimum file size to be considered a valid parquet file (bytes)
MIN_PARQUET_SIZE = 500


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


def is_valid_parquet(file_path):
    """Check if a file is a valid parquet file by checking magic bytes and size"""
    try:
        file_size = os.path.getsize(file_path)
        if file_size < MIN_PARQUET_SIZE:
            return False, f"File too small ({file_size} bytes)"

        with open(file_path, 'rb') as f:
            # FIX: Parquet magic bytes are b'PAR1', not b'PARP'
            header = f.read(4)
            f.seek(-4, 2)  # Seek to 4 bytes before EOF
            footer = f.read(4)

            if header != b'PAR1' or footer != b'PAR1':
                return False, "Missing parquet magic bytes"

        return True, "Valid"
    except Exception as e:
        return False, f"Error checking file: {str(e)}"


def load_data_file(name, data_files):
    """Try to load a data file, skipping corrupted parquet files.
    Returns (df, target_file, local_path) so caller can clean up."""
    for target_file in data_files:
        print(f"  Trying: {target_file}")

        local_path = None  # FIX: always initialise before the try block
        try:
            # Download
            local_path = hf_hub_download(
                repo_id=name,
                filename=target_file,
                repo_type="dataset",
                local_dir="temp_downloads",
                local_dir_use_symlinks=False
            )

            # Check parquet files for corruption
            if target_file.endswith('.parquet'):
                is_valid, reason = is_valid_parquet(local_path)
                if not is_valid:
                    print(f"  ⚠ Skipping {target_file}: {reason}")
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    local_path = None
                    continue

            # Read
            file_ext = Path(target_file).suffix
            if file_ext == '.parquet':
                df = pd.read_parquet(local_path)
            elif file_ext == '.csv':
                df = pd.read_csv(local_path)
            else:
                try:
                    df = pd.read_json(local_path, lines=True)
                except:
                    df = pd.read_json(local_path)

            print(f"  ✓ Loaded: {len(df)} rows, {len(df.columns)} columns")
            # FIX: return local_path so the caller can clean it up
            return df, target_file, local_path

        except Exception as e:
            print(f"  ⚠ Skipping {target_file}: {str(e)}")
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except:
                    pass
            local_path = None
            continue

    return None, None, None  # FIX: return three values to match new signature


def check_dataset(config):
    """Check a single dataset"""
    name = config["name"]
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
    }

    try:
        print("  Listing files...")
        repo_files = list_repo_files(name, repo_type="dataset")
        print(f"  ✓ Found {len(repo_files)} files")

        # Find data files
        data_files = [f for f in repo_files if f.endswith(('.parquet', '.csv', '.json'))]
        data_files = sorted(data_files, key=lambda x: (0 if x.endswith('.parquet') else 1 if x.endswith('.csv') else 2))

        if not data_files:
            result["health"] = "critical"
            result["issues"].append("No data files found")
            return result

        result["files_found"] = data_files

        # FIX: unpack three return values (df, target_file, local_path)
        df, target_file, local_path = load_data_file(name, data_files)

        if df is None:
            result["health"] = "critical"
            result["issues"].append("All data files failed to load (corrupted or invalid)")
            return result

        # Stats
        result["stats"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "file_format": Path(target_file).suffix.replace('.', ''),
            "source_file": target_file
        }

        # Validation
        if df.empty:
            result["issues"].append("Dataset is empty")

        if len(df) < config.get("min_rows", 0):
            result["issues"].append(f"Row count ({len(df)}) below minimum ({config['min_rows']})")

        # Date check
        if config.get("date_column") and config["date_column"] in df.columns:
            dates = pd.to_datetime(df[config["date_column"]], errors='coerce')
            max_date = dates.max()
            result["stats"]["last_date"] = str(max_date)

            # Freshness
            last_trading = nyse_calendar.get_last_trading_day()
            if pd.notna(max_date):
                days_behind = (last_trading - max_date).days
                result["stats"]["trading_days_behind"] = max(0, days_behind)

        # Health
        if result["issues"]:
            result["health"] = "warning"
            result["status"] = "issues_found"
        else:
            result["health"] = "healthy"
            result["status"] = "ok"

        # FIX: local_path is now properly available here from load_data_file's return value
        if local_path and os.path.exists(local_path):
            os.remove(local_path)

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        result["health"] = "critical"
        result["issues"].append(str(e))

    return result


def main():
    print("=" * 60)
    print("P2SAMAPA ETF Dataset Checker")
    print(f"Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    results = [check_dataset(config) for config in DATASETS_CONFIG]

    healthy = sum(1 for r in results if r["health"] == "healthy")
    warning = sum(1 for r in results if r["health"] == "warning")
    critical = sum(1 for r in results if r["health"] == "critical")

    print(f"\nResults: {healthy} healthy, {warning} warning, {critical} critical")

    # Save
    Path("data").mkdir(exist_ok=True)
    output = {
        "check_time": datetime.utcnow().isoformat(),
        "total_datasets": len(results),
        "summary": {"healthy": healthy, "warning": warning, "critical": critical},
        "datasets": results
    }
    with open("data/latest_check.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Saved to data/latest_check.json")
    sys.exit(0 if critical == 0 else 1)


if __name__ == "__main__":
    main()
