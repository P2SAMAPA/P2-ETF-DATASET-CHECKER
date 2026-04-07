#!/usr/bin/env python3
"""
HF Dataset Checker for P2SAMAPA ETF Datasets with NYSE Market Calendar Support
Validates 13 datasets considering US market holidays and weekends
Auto-discovers files (JSON, Parquet, CSV) in any folder structure
"""

import json
import os
import sys
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
    print("Warning: exchange-calendars not installed. Install with: pip install exchange-calendars pandas-market-calendars")

# Dataset configurations - 13 P2SAMAPA ETF datasets
# Files are now auto-discovered, but we keep expected columns for validation
DATASETS_CONFIG = [
    {
        "name": "P2SAMAPA/p2-etf-regime-predictor",
        "expected_columns": ["date", "ticker", "regime", "features"],
        "date_column": "date",
        "min_rows": 100,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/P2-ETF-DQN-ENGINE-DATASET",
        "expected_columns": ["state", "action", "reward", "next_state"],
        "date_column": None,
        "min_rows": 1000,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/p2-etf-deepwave-dl",
        "expected_columns": ["date", "ticker", "wave_features", "target"],
        "date_column": "date",
        "min_rows": 500,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/my-etf-data",
        "expected_columns": ["date", "ticker", "open", "high", "low", "close", "volume"],
        "date_column": "date",
        "min_rows": 1000,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/etf-entropy-dataset",
        "expected_columns": ["date", "ticker", "entropy_value", "window"],
        "date_column": "date",
        "min_rows": 100,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/p2-etf-hurst-data",
        "expected_columns": ["date", "ticker", "hurst_exponent", "rs_value"],
        "date_column": "date",
        "min_rows": 100,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/p2-etf-merton-ann-data",
        "expected_columns": ["date", "ticker", "jump_intensity", "jump_size", "diffusion"],
        "date_column": "date",
        "min_rows": 100,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/etf-dlinear-cross-data",
        "expected_columns": ["date", "ticker", "seasonal", "trend", "residual"],
        "date_column": "date",
        "min_rows": 500,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/p2-etf-trendfolios-replication-data",
        "expected_columns": ["date", "ticker", "momentum_score", "allocation"],
        "date_column": "date",
        "min_rows": 200,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/etf_trend_data",
        "expected_columns": ["date", "ticker", "open", "high", "low", "close", "volume"],
        "date_column": "date",
        "min_rows": 1000,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/fi-etf-macro-signal-master-data",
        "expected_columns": ["date", "signal_type", "value", "etf_ticker"],
        "date_column": "date",
        "min_rows": 500,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/p2-etf-deepm-data",
        "expected_columns": ["date", "ticker", "deep_features", "prediction"],
        "date_column": "date",
        "min_rows": 500,
        "file_types": [".parquet", ".csv", ".json"]
    },
    {
        "name": "P2SAMAPA/p2-etf-momentum-maxima",
        "expected_columns": ["date", "ticker", "momentum", "local_maxima", "window"],
        "date_column": "date",
        "min_rows": 100,
        "file_types": [".parquet", ".csv", ".json"]
    }
]

class NYSEMarketCalendar:
    """NYSE Market Calendar handler for US ETF data validation"""
    
    def __init__(self):
        self.calendar = None
        self.schedule = None
        self._init_calendar()
    
    def _init_calendar(self):
        """Initialize NYSE calendar"""
        if not NYSE_CALENDAR_AVAILABLE:
            return
        
        try:
            # Try pandas-market-calendars first
            self.calendar = get_calendar('NYSE')
        except:
            try:
                # Fallback to exchange-calendars
                self.calendar = xcals.get_calendar("XNYS")
            except:
                print("Warning: Could not initialize NYSE calendar")
                self.calendar = None
    
    def is_trading_day(self, date):
        """Check if given date is a NYSE trading day"""
        if self.calendar is None:
            # Fallback: just check weekday (0=Monday, 6=Sunday)
            return date.weekday() < 5
        
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        # Convert to pandas Timestamp if needed
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        
        # Check if date is in valid trading days
        try:
            # Get schedule for the date
            schedule = self.calendar.schedule(start_date=date, end_date=date)
            return len(schedule) > 0
        except:
            # Fallback to weekday check
            return date.weekday() < 5
    
    def get_last_trading_day(self, date=None):
        """Get the most recent trading day before or on given date"""
        if date is None:
            date = pd.Timestamp.now()
        elif isinstance(date, str):
            date = pd.Timestamp(date)
        
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        
        # Go back day by day until we find a trading day
        check_date = date
        max_days_back = 10  # Don't go back more than 10 days
        
        for _ in range(max_days_back):
            if self.is_trading_day(check_date):
                return check_date
            check_date -= timedelta(days=1)
        
        return date  # Return original if no trading day found
    
    def get_expected_trading_days(self, start_date, end_date):
        """Get all expected trading days between start and end date"""
        if self.calendar is None:
            # Fallback: generate business days excluding weekends
            dates = pd.bdate_range(start=start_date, end=end_date)
            return dates
        
        try:
            schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)
            return schedule.index
        except:
            # Fallback
            return pd.bdate_range(start=start_date, end=end_date)
    
    def get_trading_days_ago(self, n_days, from_date=None):
        """Get the date n trading days ago from from_date"""
        if from_date is None:
            from_date = pd.Timestamp.now()
        elif isinstance(from_date, str):
            from_date = pd.Timestamp(from_date)
        
        if not isinstance(from_date, pd.Timestamp):
            from_date = pd.Timestamp(from_date)
        
        # Get last n trading days
        # Start from a date far enough in the past
        start_lookup = from_date - timedelta(days=n_days * 2 + 10)
        
        try:
            trading_days = self.get_expected_trading_days(start_lookup, from_date)
            # Filter to dates <= from_date
            valid_days = trading_days[trading_days <= from_date]
            
            if len(valid_days) >= n_days:
                return valid_days[-n_days]
            else:
                return from_date - timedelta(days=n_days)  # Fallback
        except:
            # Simple fallback
            return from_date - timedelta(days=n_days)

# Initialize global calendar
nyse_calendar = NYSEMarketCalendar()

def discover_files(repo_files, file_types):
    """Auto-discover data files in the repository"""
    data_files = []
    
    for file in repo_files:
        # Skip hidden files, README, and non-data files
        if file.startswith('.') or file.startswith('_'):
            continue
        if 'README' in file or 'LICENSE' in file or '.git' in file:
            continue
        
        # Check if file matches any of our target extensions
        for ext in file_types:
            if file.endswith(ext):
                data_files.append(file)
                break
    
    # Sort by priority: parquet > csv > json (prefer structured formats)
    def priority(f):
        if f.endswith('.parquet'):
            return 0
        elif f.endswith('.csv'):
            return 1
        elif f.endswith('.json'):
            return 2
        else:
            return 3
    
    return sorted(data_files, key=priority)

def read_data_file(local_path, file_ext):
    """Read data file based on extension"""
    if file_ext == '.parquet':
        return pd.read_parquet(local_path)
    elif file_ext == '.csv':
        return pd.read_csv(local_path)
    elif file_ext == '.json':
        # Try different JSON formats
        try:
            # Try reading as JSON lines first
            return pd.read_json(local_path, lines=True)
        except:
            try:
                # Try reading as regular JSON
                return pd.read_json(local_path)
            except:
                # Try reading as records
                import json
                with open(local_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Try to normalize nested JSON
                    return pd.json_normalize(data)
                else:
                    raise ValueError(f"Unsupported JSON structure in {local_path}")
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

def check_freshness_with_calendar(df, date_column, max_trading_days=5):
    """
    Check data freshness using NYSE trading calendar
    Accounts for weekends and US market holidays
    """
    if date_column not in df.columns:
        return False, f"Date column '{date_column}' not found", None, None
    
    try:
        # Parse dates
        dates = pd.to_datetime(df[date_column], errors='coerce')
        max_date = dates.max()
        min_date = dates.min()
        
        if pd.isna(max_date):
            return False, "No valid dates found", None, None
        
        # Get today's date and last trading day
        today = pd.Timestamp.now().normalize()
        last_trading_day = nyse_calendar.get_last_trading_day(today)
        
        # Calculate trading days difference
        trading_days_diff = 0
        check_date = max_date.normalize()
        
        # Count trading days between max_date and last_trading_day
        while check_date < last_trading_day:
            check_date += timedelta(days=1)
            if nyse_calendar.is_trading_day(check_date):
                trading_days_diff += 1
        
        # Also check for missing recent trading days in dataset
        recent_trading_days = nyse_calendar.get_expected_trading_days(max_date, last_trading_day)
        dataset_dates = set(dates.dropna().normalize().unique())
        
        missing_trading_days = []
        for trading_day in recent_trading_days:
            if trading_day.normalize() not in dataset_dates:
                missing_trading_days.append(trading_day.strftime('%Y-%m-%d'))
        
        # Determine status
        is_fresh = trading_days_diff <= max_trading_days
        
        if trading_days_diff == 0:
            freshness_msg = f"Up to date (last: {max_date.strftime('%Y-%m-%d')}, last trading day: {last_trading_day.strftime('%Y-%m-%d')})"
        else:
            freshness_msg = f"{trading_days_diff} trading day(s) behind (last: {max_date.strftime('%Y-%m-%d')}, expected: {last_trading_day.strftime('%Y-%m-%d')})"
        
        # Add missing days info if any
        if missing_trading_days and len(missing_trading_days) <= 5:
            freshness_msg += f" | Missing: {', '.join(missing_trading_days)}"
        elif missing_trading_days:
            freshness_msg += f" | Missing {len(missing_trading_days)} trading days"
        
        return is_fresh, freshness_msg, max_date, missing_trading_days
        
    except Exception as e:
        return False, f"Date parsing error: {str(e)}", None, None

def check_dataset(config, filter_name=None):
    """Check a single dataset with NYSE calendar awareness"""
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
        "files_missing": [],
        "market_calendar": {
            "using_nyse_calendar": NYSE_CALENDAR_AVAILABLE and nyse_calendar.calendar is not None,
            "last_trading_day": nyse_calendar.get_last_trading_day().strftime('%Y-%m-%d') if nyse_calendar.calendar else None
        }
    }
    
    api = HfApi()
    repo_files = []
    
    try:
        # List all files in the repo
        repo_files = list_repo_files(name, token=os.environ.get("HF_TOKEN"))
        result["repo_files"] = repo_files
        print(f"  📁 Found {len(repo_files)} files in repo")
    except Exception as e:
        result["status"] = "error"
        result["issues"].append(f"Cannot access repository: {str(e)}")
        result["health"] = "critical"
        return result
    
    # Auto-discover data files
    file_types = config.get("file_types", [".parquet", ".csv", ".json"])
    data_files = discover_files(repo_files, file_types)
    
    if not data_files:
        result["status"] = "error"
        result["issues"].append(f"No data files found (looked for: {file_types}). Files in repo: {repo_files[:10]}...")
        result["health"] = "critical"
        return result
    
    result["files_found"] = data_files
    print(f"  📄 Found data files: {data_files[:3]}{'...' if len(data_files) > 3 else ''}")
    
    # Download and validate the first available file (prioritized by format)
    target_file = data_files[0]
    file_ext = Path(target_file).suffix
    local_path = None
    
    try:
        print(f"  ⬇️  Downloading {target_file}...")
        local_path = hf_hub_download(
            repo_id=name,
            filename=target_file,
            token=os.environ.get("HF_TOKEN"),
            local_dir="temp_downloads",
            local_dir_use_symlinks=False
        )
        
        print(f"  📖 Reading {file_ext} file...")
        df = read_data_file(local_path, file_ext)
        
        # Basic stats
        result["stats"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "file_size_mb": round(os.path.getsize(local_path) / (1024 * 1024), 2),
            "file_format": file_ext.replace('.', ''),
            "source_file": target_file
        }
        
        # Check row count
        if len(df) < config.get("min_rows", 0):
            result["issues"].append(
                f"Row count ({len(df)}) below minimum ({config['min_rows']})"
            )
        
        # Check columns (flexible matching)
        if "expected_columns" in config:
            expected = set(config["expected_columns"])
            actual = set(df.columns)
            
            missing_cols = expected - actual
            extra_cols = actual - expected
            
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
                f"Columns with >10% null values: {dict(high_null_cols.head(5))}"
            )
        
        # Check data freshness using NYSE calendar if enabled and date column exists
        check_fresh = os.environ.get("CHECK_FRESHNESS", "true").lower() == "true"
        if check_fresh and config.get("date_column"):
            is_fresh, freshness_msg, last_date, missing_days = check_freshness_with_calendar(
                df, config["date_column"], max_trading_days=5
            )
            result["stats"]["last_date"] = str(last_date) if last_date else None
            result["stats"]["data_freshness"] = freshness_msg
            result["stats"]["missing_trading_days"] = missing_days if missing_days else []
            result["stats"]["trading_days_behind"] = len(missing_days) if missing_days else 0
            
            if not is_fresh:
                result["issues"].append(f"Data freshness issue: {freshness_msg}")
            elif missing_days and len(missing_days) > 0:
                result["warnings"].append(f"Missing trading days: {missing_days[:3]}...")
        
        # Determine health
        if result["issues"]:
            critical_issues = [i for i in result["issues"] if any(x in i.lower() for x in ["error", "empty", "cannot access", "no data files"])]
            result["health"] = "critical" if critical_issues else "warning"
            result["status"] = "issues_found"
        else:
            result["health"] = "healthy"
            result["status"] = "ok"
            
        # Check for duplicates
        if len(df) > 0 and df.duplicated().any():
            dup_count = df.duplicated().sum()
            result["warnings"].append(f"Found {dup_count} duplicate rows")
        
        print(f"  ✅ Loaded {len(df)} rows, {len(df.columns)} columns from {target_file}")
        
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
                # Clean up parent dirs if empty
                parent = Path(local_path).parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except Exception as e:
                pass
    
    return result

def main():
    """Main execution"""
    filter_name = os.environ.get("DATASET_FILTER", "").strip()
    
    print("=" * 70)
    print("P2SAMAPA ETF Dataset Checker")
    print(f"Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Show calendar info
    if NYSE_CALENDAR_AVAILABLE and nyse_calendar.calendar:
        last_trading = nyse_calendar.get_last_trading_day()
        print(f"NYSE Calendar: Active (Last trading day: {last_trading.strftime('%Y-%m-%d')})")
    else:
        print("NYSE Calendar: Not available (using weekday fallback)")
    
    if filter_name:
        print(f"Filter: Checking only {filter_name}")
    print("=" * 70)
    
    results = []
    
    for config in DATASETS_CONFIG:
        result = check_dataset(config, filter_name if filter_name else None)
        if result:
            results.append(result)
    
    # Summary
    healthy = sum(1 for r in results if r["health"] == "healthy")
    warning = sum(1 for r in results if r["health"] == "warning")
    critical = sum(1 for r in results if r["health"] == "critical")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Healthy:   {healthy}")
    print(f"  Warning:   {warning}")
    print(f"  Critical:  {critical}")
    print("=" * 70)
    
    # Save results
    output = {
        "check_time": datetime.utcnow().isoformat(),
        "total_datasets": len(results),
        "summary": {
            "healthy": healthy,
            "warning": warning,
            "critical": critical
        },
        "market_calendar": {
            "type": "NYSE",
            "last_trading_day": nyse_calendar.get_last_trading_day().strftime('%Y-%m-%d') if nyse_calendar.calendar else None,
            "active": NYSE_CALENDAR_AVAILABLE and nyse_calendar.calendar is not None
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
    
    # Show calendar-aware summary
    if nyse_calendar.calendar:
        print(f"\n📅 NYSE Calendar Note:")
        print(f"   Last trading day: {nyse_calendar.get_last_trading_day().strftime('%A, %B %d, %Y')}")
        next_trading = nyse_calendar.get_last_trading_day() + timedelta(days=1)
        while not nyse_calendar.is_trading_day(next_trading):
            next_trading += timedelta(days=1)
        print(f"   Next trading day: {next_trading.strftime('%A, %B %d, %Y')}")
    
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
