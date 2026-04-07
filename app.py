import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import os

# Page config
st.set_page_config(
    page_title="P2 ETF Dataset Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .healthy { color: #00C851; font-weight: bold; font-size: 1.1rem; }
    .warning { color: #ffbb33; font-weight: bold; font-size: 1.1rem; }
    .critical { color: #ff4444; font-weight: bold; font-size: 1.1rem; }
    .dataset-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #ddd;
    }
    .dataset-card.healthy { border-left-color: #00C851; }
    .dataset-card.warning { border-left-color: #ffbb33; }
    .dataset-card.critical { border-left-color: #ff4444; }
    .stProgress > div > div > div > div {
        background-color: #FF6B35;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load latest check results"""
    data_file = Path("data/latest_check.json")
    if data_file.exists():
        with open(data_file, 'r') as f:
            return json.load(f)
    return None

def load_history():
    """Load historical check data"""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    
    history_files = sorted(data_dir.glob("check_*.json"), reverse=True)
    history = []
    
    for f in history_files[:20]:  # Last 20 checks
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                history.append(data)
        except:
            continue
    
    return list(reversed(history))  # Oldest first

def get_health_icon(health):
    icons = {
        'healthy': '✅',
        'warning': '⚠️',
        'critical': '❌',
        'unknown': '❓'
    }
    return icons.get(health, '❓')

def get_health_color(health):
    colors = {
        'healthy': '#00C851',
        'warning': '#ffbb33',
        'critical': '#ff4444',
        'unknown': '#9e9e9e'
    }
    return colors.get(health, '#9e9e9e')

def format_number(num):
    """Format large numbers"""
    if num is None or num == 'N/A':
        return 'N/A'
    if isinstance(num, (int, float)):
        if num >= 1_000_000:
            return f"{num/1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        return str(int(num))
    return str(num)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 P2 ETF Dataset Monitor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time monitoring of 13 Hugging Face ETF datasets</p>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    if not data:
        st.error("⚠️ No check data found. Please run the GitHub Actions workflow first.")
        st.info("Go to: Actions → 'Check HF Datasets Status' → Run workflow")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Filters & Controls")
        
        # Refresh
        if st.button("🔄 Refresh Dashboard", use_container_width=True, type="primary"):
            st.rerun()
        
        st.divider()
        
        # Health filter
        health_filter = st.multiselect(
            "Filter by Health Status",
            ["healthy", "warning", "critical"],
            default=["healthy", "warning", "critical"]
        )
        
        # Format filter
        format_filter = st.multiselect(
            "Filter by Format",
            ["parquet", "csv", "mixed"],
            default=["parquet", "csv", "mixed"]
        )
        
        st.divider()
        
        # Last check info
        check_time = datetime.fromisoformat(data['check_time'].replace('Z', '+00:00'))
        time_diff = datetime.utcnow() - check_time.replace(tzinfo=None)
        
        st.info(f"**Last Check:**\n{check_time.strftime('%Y-%m-%d %H:%M UTC')}\n({time_diff.total_seconds()/3600:.1f} hours ago)")
        
        # Summary in sidebar
        st.subheader("Quick Stats")
        summary = data.get('summary', {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Healthy", summary.get('healthy', 0))
            st.metric("⚠️ Warning", summary.get('warning', 0))
        with col2:
            st.metric("❌ Critical", summary.get('critical', 0))
            st.metric("📊 Total", data.get('total_datasets', 0))
        
        st.divider()
        st.markdown("[Run Manual Check](https://github.com/P2SAMAPA/P2-ETF-DATASET-CHECKER/actions)")
    
    # Main content
    datasets = data.get('datasets', [])
    
    # Apply filters
    filtered_datasets = [
        d for d in datasets 
        if d.get('health') in health_filter
        and d.get('stats', {}).get('file_format', 'unknown') in format_filter
    ]
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total = len(datasets)
    healthy = sum(1 for d in datasets if d.get('health') == 'healthy')
    warning = sum(1 for d in datasets if d.get('health') == 'warning')
    critical = sum(1 for d in datasets if d.get('health') == 'critical')
    
    with col1:
        st.metric("📊 Total", total)
    with col2:
        delta = f"{healthy/total*100:.0f}%" if total > 0 else "0%"
        st.metric("✅ Healthy", healthy, delta=delta)
    with col3:
        st.metric("⚠️ Warning", warning)
    with col4:
        st.metric("❌ Critical", critical)
    with col5:
        total_rows = sum(d.get('stats', {}).get('total_rows', 0) or 0 for d in datasets)
        st.metric("📈 Total Rows", format_number(total_rows))
    
    st.divider()
    
    # Charts row
    col_chart, col_table = st.columns([1, 2])
    
    with col_chart:
        st.subheader("Health Distribution")
        
        health_counts = pd.DataFrame([
            {'Status': 'Healthy', 'Count': healthy, 'color': '#00C851'},
            {'Status': 'Warning', 'Count': warning, 'color': '#ffbb33'},
            {'Status': 'Critical', 'Count': critical, 'color': '#ff4444'}
        ])
        
        if health_counts['Count'].sum() > 0:
            fig = px.pie(
                health_counts,
                values='Count',
                names='Status',
                color='Status',
                color_discrete_map={
                    'Healthy': '#00C851',
                    'Warning': '#ffbb33',
                    'Critical': '#ff4444'
                },
                hole=0.5
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True, height=300)
        else:
            st.info("No data to display")
    
    with col_table:
        st.subheader("Dataset Overview")
        
        overview_data = []
        for ds in filtered_datasets:
            stats = ds.get('stats', {})
            overview_data.append({
                'Dataset': ds['name'].split('/')[-1],  # Short name
                'Health': f"{get_health_icon(ds.get('health'))} {ds.get('health', 'unknown').upper()}",
                'Format': stats.get('file_format', 'unknown'),
                'Rows': format_number(stats.get('total_rows')),
                'Size': f"{stats.get('file_size_mb', 0)} MB",
                'Cols': stats.get('total_columns', 'N/A'),
                'Last Date': stats.get('last_date', 'N/A')[:10] if stats.get('last_date') else 'N/A'
            })
        
        df_overview = pd.DataFrame(overview_data)
        st.dataframe(
            df_overview,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Health": st.column_config.Column(width="medium"),
                "Dataset": st.column_config.Column(width="large")
            }
        )
    
    st.divider()
    
    # Detailed view
    st.subheader("📋 Detailed Dataset View")
    
    # Group by health for better organization
    for health_status in ['critical', 'warning', 'healthy']:
        status_datasets = [d for d in filtered_datasets if d.get('health') == health_status]
        
        if status_datasets:
            st.markdown(f"### {get_health_icon(health_status)} {health_status.upper()} ({len(status_datasets)})")
            
            for ds in status_datasets:
                with st.expander(
                    f"**{ds['name']}** — {ds.get('stats', {}).get('file_format', 'unknown').upper()} | "
                    f"{format_number(ds.get('stats', {}).get('total_rows', 0))} rows",
                    expanded=(health_status == 'critical')
                ):
                    display_dataset_details(ds)

    # Historical trends
    st.divider()
    st.subheader("📈 Historical Trends")
    
    history = load_history()
    if len(history) > 1:
        display_trends(history)
    else:
        st.info("Need more historical data to show trends. Run the checker multiple times.")

def display_dataset_details(ds):
    """Display detailed info for a dataset"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    stats = ds.get('stats', {})
    
    with col1:
        # Health status
        health = ds.get('health', 'unknown')
        st.markdown(f"**Status:** <span class='{health}'>{health.upper()}</span>", unsafe_allow_html=True)
        
        # Issues
        issues = ds.get('issues', [])
        if issues:
            st.error("**🚨 Issues:**")
            for issue in issues:
                st.write(f"• {issue}")
        
        # Warnings
        warnings = ds.get('warnings', [])
        if warnings:
            st.warning("**⚠️ Warnings:**")
            for warning in warnings:
                st.write(f"• {warning}")
        
        if not issues and not warnings:
            st.success("✅ No issues detected")
    
    with col2:
        st.write("**📊 Statistics:**")
        st.write(f"• Rows: {format_number(stats.get('total_rows'))}")
        st.write(f"• Columns: {stats.get('total_columns', 'N/A')}")
        st.write(f"• File Size: {stats.get('file_size_mb', 0)} MB")
        st.write(f"• Format: {stats.get('file_format', 'unknown')}")
        
        freshness = stats.get('data_freshness', 'N/A')
        if freshness != 'N/A':
            st.write(f"• Freshness: {freshness}")
    
    with col3:
        st.write("**📁 Files:**")
        st.write(f"• Found: {len(ds.get('files_found', []))}")
        for f in ds.get('files_found', [])[:3]:  # Show max 3
            st.write(f"  - `{f}`")
        
        missing = ds.get('files_missing', [])
        if missing:
            st.write(f"• Missing: {len(missing)}")
        
        st.write(f"• Checked: {ds.get('checked_at', 'N/A')[:10]}")
    
    # Column details
    with st.expander("View Columns"):
        cols = stats.get('columns', [])
        if cols:
            col_df = pd.DataFrame({'Column Name': cols, 'Index': range(len(cols))})
            st.dataframe(col_df, hide_index=True, use_container_width=True)
        else:
            st.write("No column information available")

def display_trends(history):
    """Display historical trend charts"""
    if not history:
        return
    
    # Prepare trend data
    trend_data = []
    for check in history:
        time_point = datetime.fromisoformat(check['check_time'].replace('Z', '+00:00'))
        summary = check.get('summary', {})
        
        trend_data.append({
            'time': time_point,
            'healthy': summary.get('healthy', 0),
            'warning': summary.get('warning', 0),
            'critical': summary.get('critical', 0),
            'total': check.get('total_datasets', 0)
        })
    
    df_trend = pd.DataFrame(trend_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Health over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_trend['time'], y=df_trend['healthy'],
            mode='lines+markers', name='Healthy',
            line=dict(color='#00C851', width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=df_trend['time'], y=df_trend['warning'],
            mode='lines+markers', name='Warning',
            line=dict(color='#ffbb33', width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=df_trend['time'], y=df_trend['critical'],
            mode='lines+markers', name='Critical',
            line=dict(color='#ff4444', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Health Status Over Time",
            xaxis_title="Check Time",
            yaxis_title="Number of Datasets",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=350,
            margin=dict(t=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Total datasets tracked
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_trend['time'],
            y=df_trend['total'],
            marker_color='#667eea',
            name='Total Datasets'
        ))
        
        fig.update_layout(
            title="Total Datasets Checked",
            xaxis_title="Check Time",
            yaxis_title="Count",
            height=350,
            showlegend=False,
            margin=dict(t=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show history table
    with st.expander("View Check History"):
        history_df = pd.DataFrame([
            {
                'Time': h['check_time'][:16].replace('T', ' '),
                'Healthy': h.get('summary', {}).get('healthy', 0),
                'Warning': h.get('summary', {}).get('warning', 0),
                'Critical': h.get('summary', {}).get('critical', 0),
                'Total': h.get('total_datasets', 0)
            }
            for h in reversed(history[-10:])  # Last 10
        ])
        st.dataframe(history_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()
