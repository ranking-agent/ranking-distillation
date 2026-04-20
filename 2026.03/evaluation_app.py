import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import re

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Surrogate Model Eval", layout="wide")

# --- TRAPI Node Normalization ---
@st.cache_data(show_spinner=False)
def fetch_node_name(curie):
    url = "https://nodenormalization-sri.renci.org/get_normalized_nodes"
    try:
        response = requests.post(url, json={"curies": [curie]}, timeout=10)
        response.raise_for_status()
        data = response.json().get(curie)
        if data and 'id' in data:
            return data['id'].get('label', curie)
    except Exception:
        pass
    return curie

def get_human_readable_name(filename):
    pattern = r'([A-Z][a-zA-Z0-9]+)[_:]([A-Z0-9]+)'
    matches = re.findall(pattern, filename)
    if not matches:
        return filename
    names = [fetch_node_name(f"{prefix}:{identifier}") for prefix, identifier in matches]
    return " -> ".join(names) if names else filename

# --- Data Loading ---
@st.cache_data
def load_run_data(run_dir_path):
    run_dir = Path(run_dir_path)
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    
    pred_files = list(run_dir.glob("preds_fold_*.parquet"))
    if not pred_files:
        return None
        
    dfs = []
    for f in pred_files:
        df = pd.read_parquet(f)
        y_true = df['y_true'].values
        y_prob = df['y_prob'].values
        n_pos = np.sum(y_true == 1)
        
        if n_pos > 0:
            df['fold_auc'] = roc_auc_score(y_true, y_prob)
            df['fold_ap'] = average_precision_score(y_true, y_prob)
            # Random Precision (RP) = Positive Class Prevalence
            df['fold_rp'] = n_pos / len(y_true)
            df['n_pos'] = n_pos
            dfs.append(df)
            
    return pd.concat(dfs, ignore_index=True) if dfs else None

# --- Sidebar ---
st.sidebar.title("🧬 Pipeline Explorer")
run_dir_input = st.sidebar.text_input(
    "Run Directory Path", 
    placeholder="/path/to/timestamped_run_folder"
)

df_all = load_run_data(run_dir_input)

if df_all is None:
    st.warning("Please enter a valid Run Directory path containing prediction files.")
    st.stop()

tab1, tab2 = st.tabs(["📊 Performance Plots", "🔍 Raw Data Simulator"])

# ==========================================
# TAB 1: PERFORMANCE PLOTS
# ==========================================
with tab1:
    st.header("Interactive Performance Metrics")
    
    unique_jobs = df_all['source_file'].unique()
    
    fig_roc = go.Figure()
    fig_pr = make_subplots(rows=2, cols=2, 
                           subplot_titles=("Precision vs Absolute Volume", "Precision vs Fraction", 
                                           "Recall vs Absolute Volume", "Recall vs Fraction"))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, job in enumerate(unique_jobs):
        job_df = df_all[df_all['source_file'] == job]
        y_true = job_df['y_true'].values
        y_prob = job_df['y_prob'].values
        
        hr_name = get_human_readable_name(job)
        auc = job_df['fold_auc'].iloc[0]
        ap = job_df['fold_ap'].iloc[0]
        rp = job_df['fold_rp'].iloc[0]
        n_pos = job_df['n_pos'].iloc[0]
        c = colors[i % len(colors)]
        
        # ROC Trace
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                     name=f"{hr_name} (AUC: {auc:.2f})", 
                                     line=dict(color=c)))
        
        # Volumetric PR Calculations
        idx = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[idx]
        volumes = np.arange(1, len(y_true_sorted) + 1)
        fractions = volumes / len(y_true_sorted)
        cum_pos = np.cumsum(y_true_sorted)
        
        precision_vol = cum_pos / volumes
        recall_vol = cum_pos / n_pos
        
        # Legend with AP/RP context
        legend_label = f"{hr_name} (AP/RP: {ap:.2f}/{rp:.2f})"
        
        # Add traces to subplots
        fig_pr.add_trace(go.Scatter(x=volumes, y=precision_vol, name=legend_label, legendgroup=job, line=dict(color=c)), row=1, col=1)
        fig_pr.add_trace(go.Scatter(x=fractions, y=precision_vol, legendgroup=job, showlegend=False, line=dict(color=c)), row=1, col=2)
        fig_pr.add_trace(go.Scatter(x=volumes, y=recall_vol, legendgroup=job, showlegend=False, line=dict(color=c)), row=2, col=1)
        fig_pr.add_trace(go.Scatter(x=fractions, y=recall_vol, legendgroup=job, showlegend=False, line=dict(color=c)), row=2, col=2)

    # ROC Baseline: Light Gray for Dark Mode
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                 line=dict(color='lightgray', dash='dash', width=1), 
                                 name='Random Baseline', hoverinfo='skip'))
    
    fig_roc.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", 
                          yaxis_title="True Positive Rate", height=600, template="plotly_dark")
    
    fig_pr.update_xaxes(title_text="Total Results Chosen", range=[0, 2000], row=2, col=1)
    fig_pr.update_xaxes(title_text="Fraction of Dataset", range=[0, 1.0], row=2, col=2)
    fig_pr.update_yaxes(title_text="Precision", range=[0, 1.05], row=1, col=1)
    fig_pr.update_yaxes(title_text="Recall", range=[0, 1.05], row=2, col=1)
    fig_pr.update_layout(height=800, hovermode="x unified", template="plotly_dark")
    
    st.plotly_chart(fig_roc, width="stretch")
    st.plotly_chart(fig_pr, width="stretch")

# ==========================================
# TAB 2: RAW DATA SIMULATOR
# ==========================================
with tab2:
    st.header("Search Result Simulator")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        dataset_choice = st.selectbox("Select Dataset Fold", df_all['source_file'].unique())
        top_n = st.slider("Display Top N Results", min_value=10, max_value=2000, value=200, step=10)
    
    display_df = df_all[df_all['source_file'] == dataset_choice].copy()
    display_df = display_df.sort_values(by="y_prob", ascending=False).head(top_n)
    
    # Map tier if not explicitly present in the data dump
    if 'tier' not in display_df.columns:
        display_df['tier'] = display_df['y_true'].apply(lambda x: "Target" if x == 1 else "Other")

    # Handle cases where 'path' or 'explanation' might be missing from input parquets
    for col in ["path", "explanation"]:
        if col not in display_df.columns:
            display_df[col] = "N/A"

    st.subheader(f"Top {top_n} Results for: {get_human_readable_name(dataset_choice)}")
    
    cols_to_show = ["y_prob", "tier", "path", "explanation"]
        
    st.dataframe(
        display_df[cols_to_show],
        width="stretch",
        column_config={
            "y_prob": st.column_config.ProgressColumn("Model Prob", format="%.3f", min_value=0, max_value=1),
            "tier": st.column_config.TextColumn("Tier", width="small"),
            "path": st.column_config.TextColumn("Path", width="large"),
            "explanation": st.column_config.TextColumn("Explanation", width="large")
        },
        hide_index=True
    )
