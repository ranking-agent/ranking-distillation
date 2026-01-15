# %%
import pandas as pd
import numpy as np
import time
import re
import os
from pathlib import Path
import plotly.express as px
from google import genai
from google.genai import types
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.palettes import Category20
from bokeh.transform import factor_cmap

from query_compression import convert_results_to_sentences


# --- 1. Setup & Embedding Generation ---
# Replace with your API key or set it in your environment
client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
MODEL_ID = "text-embedding-004" # Latest stable embedding model


# --- 0. Data Loading & Filename Matching ---
def match_orig_csv(job_path, orig_dir):
    """
    Cleaner way to match ranking results back to raw query results.
    Uses regex to extract the core query name from the long Gemini output filename.
    """
    
    # Extract the core query name. 
    # Example: '2025-12-31-gemini-ranking-MONDO_002..._paths-gemini-3...'
    # We look for the part between '-ranking-' and the next '-gemini-' or '-bs-'
    match = re.search(r"(?:-ranking-|_)(.*?)(?:-gemini|_ranked)", job_path.name)
    if not match:
        raise ValueError(f"Could not parse query name from filename: {job_path.name}")
    
    query_core_name = match.group(1)
    
    # Find the original CSV in the raw queries directory that contains this core name
    orig_csvs = list(Path(orig_dir).glob(f"*{query_core_name}*.csv"))
    
    if not orig_csvs:
        raise FileNotFoundError(f"No original CSV found matching: {query_core_name}")
        
    return orig_csvs[0]


def get_embeddings_with_retry(texts, batch_size=50):
    """
    Generates embeddings using Gemini with exponential backoff for rate limits.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        retries = 0
        while retries < 5:
            try:
                # Task type 'REGRESSION' is often best for predicting continuous scores like MRR
                result = client.models.embed_content(
                    model=MODEL_ID,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                all_embeddings.extend([e.values for e in result.embeddings])
                break
            except Exception as e:
                wait_time = (2 ** retries)
                print(f"Error at batch {i}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retries += 1
        else:
            raise RuntimeError(f"Failed to embed batch starting at index {i} after 5 retries.")
    return np.array(all_embeddings)


def process_datasets(dfs):
    """
    df_list: List of 6 DataFrames. Each must have 'query_sentence', 'median', and 'dataset_id'.
    """
    full_data_list = []
    
    for i, df in enumerate(dfs):
        print(f"Processing Dataset {i+1}/6...")
        # Generate embeddings
        sentences = df['query_sentence'].tolist()
        embeddings = get_embeddings_with_retry(sentences)
        
        # Store metadata and features
        df['embeddings'] = list(embeddings)
        df['dataset_idx'] = i # Useful for Leave-One-Dataset-Out CV
        full_data_list.append(df)
        
    return pd.concat(full_data_list, ignore_index=True)


# --- 2. Leave-One-Dataset-Out Cross Validation ---
def run_lodo_cv(combined_df):
    """
    Fits a Random Forest to predict 'median' (MRR) using embeddings.
    Runs 6 folds (leaving one whole pathfinder dataset out each time).
    """
    dataset_indices = combined_df['dataset_idx'].unique()
    results = []

    # Features (X) and Target (y)
    X = np.stack(combined_df['embeddings'].values)
    y = combined_df['median'].values

    for test_idx in dataset_indices:
        # Split
        train_mask = combined_df['dataset_idx'] != test_idx
        test_mask = combined_df['dataset_idx'] == test_idx
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Model
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Predict
        preds = rf.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results.append({'fold': test_idx, 'mse': mse, 'r2': r2})
        print(f"Fold {test_idx} (Left Out): MSE={mse:.4f}, R2={r2:.4f}")

    return pd.DataFrame(results)


def create_tsne_viz(df, output_filename=None):
    """
    Projects embeddings for a SINGLE dataset to 2D and creates an interactive Bokeh plot.
    If output_filename is provided (e.g., 'viz.html'), it saves a standalone HTML file.
    """
    print("Fitting t-SNE for single dataset visualization...")
    X = np.stack(df['embeddings'].values)
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    projections = tsne.fit_transform(X)
    
    df = df.copy()
    df['tsne_1'] = projections[:, 0]
    df['tsne_2'] = projections[:, 1]
    
    # Configure output file if requested
    if output_filename:
        # mode='inline' embeds BokehJS so the file works offline
        output_file(filename=output_filename, title="KG Insight Semantic Map", mode='inline')
    
    source = ColumnDataSource(df)
    factors = sorted(df['categories'].unique().tolist())
    palette = Category20[len(factors)] if len(factors) <= 20 else Category20[20] * (len(factors)//20 + 1)
    color_map = factor_cmap(field_name='categories', palette=palette[:len(factors)], factors=factors)
    
    p = figure(title="Pathfinder Result Semantic Clusters", width=900, height=700,
               tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom")
    p.circle('tsne_1', 'tsne_2', size=8, source=source, color=color_map, 
             fill_alpha=0.6, line_color="white", legend_field="categories")
    
    hover = HoverTool(tooltips=[("Insight (MRR)", "@median{0.000}"), ("Category", "@categories"), ("Sentence", "@query_sentence")])
    p.add_tools(hover)
    p.legend.location = "top_left"; p.legend.click_policy = "hide"
    
    if output_filename:
        print(f"Saving visualization to {output_filename}...")
        save(p)
    else:
        show(p)


def get_stratified_samples(
    df, 
    rank_col='reciprocal_rank', 
    top_n=20,
    mid_range=(0.45, 0.55), 
    mid_samples=10,
    bot_range=(0.90, 1.0), 
    bot_samples=10
):
    """
    Generalised sampler for model evaluation.
    
    Args:
        df: Input DataFrame.
        rank_col: Column to sort by.
        top_n: Absolute number of top rows to return.
        mid_range: Tuple of (start_percent, end_percent) for the middle zone.
        mid_samples: Number of rows to spread-sample from the middle zone.
        bot_range: Tuple of (start_percent, end_percent) for the bottom zone.
        bot_samples: Number of rows to spread-sample from the bottom zone.
    """
    df_sorted = df.sort_values(by=rank_col, ascending=True).reset_index(drop=True)
    n = len(df_sorted)

    def get_spread_samples(target_df, num_samples):
        if len(target_df) <= num_samples:
            return target_df
        # Create evenly spaced indices across the provided slice
        indices = np.linspace(0, len(target_df) - 1, num_samples).astype(int)
        return target_df.iloc[indices]

    # 1. Top Subset (Standard Head)
    top_df = df_sorted.head(top_n)

    # 2. Middle Subset (Spread)
    mid_start, mid_end = int(n * mid_range[0]), int(n * mid_range[1])
    mid_zone = df_sorted.iloc[mid_start:mid_end]
    mid_df = get_spread_samples(mid_zone, mid_samples)

    # 3. Bottom Subset (Spread)
    bot_start, bot_end = int(n * bot_range[0]), int(n * bot_range[1])
    bot_zone = df_sorted.iloc[bot_start:bot_end]
    bot_df = get_spread_samples(bot_zone, bot_samples)

    return top_df, mid_df, bot_df


# %%
if __name__ == "__main__":
    
    # Base NAS dir
    TEST_SETS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/")

    # Original results from Neo4j
    ORIG_QUERIES_DIR = TEST_SETS_DIR / "rare_disease_queries"
    test_csvs = list(ORIG_QUERIES_DIR.glob("*.csv"))
    print("Raw query results files:")
    print(*test_csvs, sep="\n")

    # Results from Gemini
    # RESULTS_DIR = TEST_SETS_DIR / "ranked_results" / "flash"
    RESULTS_DIR = TEST_SETS_DIR / "ranked_results"
    ranking_outputs = list(RESULTS_DIR.glob("*.parquet"))
    print("\nRanked outputs avaialble:")
    print(*ranking_outputs, sep="\n")

    df_list = []
    for pth in ranking_outputs:
        print("\nLoading:", pth.name)
        orig_dataset = match_orig_csv(pth, ORIG_QUERIES_DIR)

        # Read in original query results
        df_orig = pd.read_csv(orig_dataset, sep="\t")
        df_orig = df_orig.reset_index(names="orig_query_index")
        df_orig = convert_results_to_sentences(df_orig).set_index("orig_query_index", drop=True)

        # Read in ranking job results
        df = pd.read_parquet(pth, engine="fastparquet")
        df['reciprocal_rank'] = df.groupby("job_id")["rank"].transform(lambda x: x / x.max())
        df = df.rename(columns={"index": "orig_query_index"})

        # Group by original result and aggregate reciprocal rank (basically, generate mean reciprocal rank)
        df_mrr = df.groupby('orig_query_index').agg({
            'reciprocal_rank': 'median',
            'explanation': 'first', # Grab the first explanation found
        }).reset_index().sort_values("reciprocal_rank")
        df_mrr = df_mrr.join(df_orig, on="orig_query_index")
        df_list.append(df_mrr)


# %%
for dfq in df_list:
    print(dfq['reciprocal_rank'].iloc[0])
    print(dfq['query_sentence'].iloc[0])
    print(dfq['explanation'].iloc[0])
    


# %%
master_df = process_datasets(df_list)
master_df.to_parquet("data/rare_diseases_with_embeddings.parquet", engine="pyarrow", compression="snappy")

# %%
master_df = pd.read_parquet("data/rare_diseases_with_embeddings.parquet", engine="pyarrow")

# %%
# cv_stats = run_lodo_cv(master_df)

# %%
df1 = master_df[master_df['dataset_idx']==0]
create_tsne_viz(df1, output_filename="figures/tsne_1.html")

# %%
# Generate top / middle / bottom cuts for google sheets
df1 = master_df[master_df['dataset_idx']==2]
top, mid, bot = get_stratified_samples(df1)

disp_cols = [
    "path",
    "reciprocal_rank",
    "explanation",
]
print(top[disp_cols].to_csv(sep='\t', index=True))
print(mid[disp_cols].to_csv(sep='\t', index=True))
print(bot[disp_cols].to_csv(sep='\t', index=True))


# %%
# Base NAS dir
TEST_SETS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/")

# Original results from Neo4j
ORIG_QUERIES_DIR = TEST_SETS_DIR / "rare_disease_queries"
test_csvs = list(ORIG_QUERIES_DIR.glob("*.csv"))
print("Raw query results files:")
print(*test_csvs, sep="\n")

# Results from pre-Flash
RESULTS_DIR = TEST_SETS_DIR / "ranked_results"
ranking_outputs = list(RESULTS_DIR.glob("*.parquet"))
print("\nRanked outputs avaialble:")
print(*ranking_outputs, sep="\n")

df_list_old = []
orig_results_dfs_old = []
for ii, pth in enumerate(ranking_outputs):
    print("\nLoading:", pth.name)
    try:
        orig_dataset = match_orig_csv(pth, ORIG_QUERIES_DIR)
    except FileNotFoundError:
        df_list_old.append(pd.DataFrame())
        orig_results_dfs_old.append(pd.DataFrame())
        continue

    # Read in original query results
    df_orig = pd.read_csv(orig_dataset, sep="\t")
    df_orig = df_orig.reset_index(names="orig_query_index")
    df_orig = convert_results_to_sentences(df_orig).set_index("orig_query_index", drop=True)
    orig_results_dfs_old.append(df_orig)

    # Read in ranking job results
    df = pd.read_parquet(pth, engine="fastparquet")
    df['reciprocal_rank'] = df.groupby("job_id")["rank"].transform(lambda x: x / x.max())
    df = df.rename(columns={"index": "orig_query_index"})

    # Group by original result and aggregate reciprocal rank (basically, generate mean reciprocal rank)
    df_mrr = df.groupby('orig_query_index').agg({
        'reciprocal_rank': 'median',
        'explanation': 'first', # Grab the first explanation found
    }).reset_index().sort_values("reciprocal_rank")
    df_mrr = df_mrr.join(df_orig, on="orig_query_index")
    df_mrr['dataset_idx'] = ii
    df_list_old.append(df_mrr)

master_df_old = pd.concat(df_list_old, ignore_index=True)


# %%
# Generate top / middle / bottom cuts for google sheets
ds_idx = 9
df2 = df_list_old[ds_idx]
df2_orig = orig_results_dfs_old[ds_idx]
top, mid, bot = get_stratified_samples(df2)

disp_cols = [
    "path",
    "orig_query_index",
    "reciprocal_rank",
    "explanation",
    "query_sentence"
]
print(top[disp_cols].to_csv(sep='\t', index=True))
print(mid[disp_cols].to_csv(sep='\t', index=True))
print(bot[disp_cols].to_csv(sep='\t', index=True))

# %%
