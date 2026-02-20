# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths (adjust to your environment)
RESULTS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/training_queries/ranking_outputs")

# 1. Load Data
parquet_files = [f for f in RESULTS_DIR.glob("*.parquet") if "-v9-" in f.name]
dfs = [pd.read_parquet(f).assign(source_file=f.name) for f in parquet_files]
full_df = pd.concat(dfs, ignore_index=True).dropna(subset=['tier', 'query_sentence'])
full_df['tier'] = full_df['tier'].astype(int)
full_df['is_tier_1'] = (full_df['tier'] == 1).astype(int)
full_df['is_top_tier'] = (full_df['tier'] <= 2).astype(int)


# %%
def analyze_tiers_by_category(df):
    # 1. Prepare count data for sorting and visualization
    # We group by both category and tier to get counts
    counts_df = df.groupby(['categories', 'tier']).size().reset_index(name='count')
    
    # 2. Determine sort order (Most Tier 1 results on the left)
    # We pivot to get tiers as columns to easily sort by Tier 1 values
    pivot_df = counts_df.pivot(index='categories', columns='tier', values='count').fillna(0)
    
    # Ensure tiers 1 and 2 exist in columns even if they aren't in the data
    for t in [1, 2]:
        if t not in pivot_df.columns:
            pivot_df[t] = 0
            
    # Sort by Tier 1 (descending), then Tier 2
    pivot_df = pivot_df.sort_values(by=[1, 2], ascending=False)
    category_order = pivot_df.index.tolist()
    
    # 3. Create the Bar Chart
    plt.figure(figsize=(14, 7))
    sns.countplot(
        data=df, 
        x='categories', 
        hue='tier', 
        order=category_order,
        palette='viridis' # Distinct colors for tiers
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.title("Pathfinder Results: Tier Distribution per Category")
    plt.xlabel("Category Path")
    plt.ylabel("Result Count")
    plt.legend(title='Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('tier_distribution.png')
    
    # 4. Identify and print categories with zero results in Tiers 1 or 2
    # This filters for categories where BOTH Tier 1 and Tier 2 counts are 0
    low_tier_only = pivot_df[(pivot_df[1] == 0) & (pivot_df[2] == 0)].index.tolist()
    
    print("Categories with zero results in Tiers 1 or 2:")
    if not low_tier_only:
        print("None - all categories have at least one Tier 1 or Tier 2 result.")
    else:
        for cat in low_tier_only:
            print(f"- {cat}")

# %%
analyze_tiers_by_category(full_df)

# %%
def calculate_low_tier_category_fraction(df):
    # 1. Group by category and tier to find which categories have Tiers 1 or 2
    cat_tier_counts = df.groupby(['categories', 'tier']).size().unstack(fill_value=0)
    
    # Ensure columns 1 and 2 exist in the check
    high_tiers = [1, 2]
    existing_high_tiers = [t for t in high_tiers if t in cat_tier_counts.columns]
    
    # 2. Identify categories where the sum of Tier 1 and Tier 2 results is 0
    if not existing_high_tiers:
        # If no Tier 1 or 2 exists in the whole DF, then 100% of the DF is in this state
        low_tier_categories = cat_tier_counts.index.tolist()
    else:
        # Sum counts across Tier 1 and 2; if 0, it's a "low-tier only" category
        low_tier_mask = cat_tier_counts[existing_high_tiers].sum(axis=1) == 0
        low_tier_categories = cat_tier_counts[low_tier_mask].index.tolist()
    
    # 3. Calculate the fraction of the total rows belonging to these categories
    num_low_tier_rows = df[df['categories'].isin(low_tier_categories)].shape[0]
    total_rows = len(df)
    
    fraction = num_low_tier_rows / total_rows if total_rows > 0 else 0
    
    print(f"Number of 'Tier 3/4 only' categories: {len(low_tier_categories)}")
    print(f"Fraction of overall DataFrame: {fraction:.2%} ({num_low_tier_rows} / {total_rows} rows)")
    
    return fraction

# %%
fraction = calculate_low_tier_category_fraction(full_df)
# %%
