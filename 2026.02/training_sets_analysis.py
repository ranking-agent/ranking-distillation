# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from query_compression import convert_results_to_sentences


# %%
# Base NAS dir
TEST_SETS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/")

# Original results from Neo4j
ORIG_QUERIES_DIR = TEST_SETS_DIR / "training_queries"
test_csvs = list(ORIG_QUERIES_DIR.glob("*.csv"))
print("Raw query results files:")
print(*test_csvs, sep="\n")

# Results from Gemini
RESULTS_DIR = ORIG_QUERIES_DIR / "ranking_outputs"
ranking_outputs = list(RESULTS_DIR.glob("*.parquet"))
print("\nRanked outputs avaialble:")
print(*ranking_outputs, sep="\n")

# Pick a specific job result file
# job_file = Path("2026-02-05-gemini-labels-CHEBI_13719_to_MONDO_0005575_3_hop_w_direction_paths-gemini-3-flash-preview-prompt-v9-nr-1-bs-200-df-1-sb-0-gb-categories-rs-42.parquet")
# job_file = Path("2026-02-05-gemini-labels-CHEBI_5118_to_MONDO_0100233_3_hop_w_direction_paths-gemini-3-flash-preview-prompt-v9-nr-1-bs-200-df-1-sb-0-gb-categories-rs-42.parquet")
# job_file = Path("2026-02-05-gemini-labels-asthma_results_subset-gemini-3-flash-preview-prompt-v9-nr-1-bs-500-df-1-sb-0-gb-categories-rs-42.parquet")
job_file = Path("2026-02-12-gemini-labels-CHEBI_9139_to_MONDO_0004975_3_hop_w_direction_paths-gemini-3-flash-preview-prompt-v9-nr-1-bs-500-df-20-sb-0-gb-categories-rs-42.parquet")
# job_file = Path()
selected_results = RESULTS_DIR / job_file
print("\nSelected:", selected_results.name)


# %%
# Read in ranking job results
df = pd.read_parquet(selected_results, engine="fastparquet")
df = df.rename(columns={"index": "orig_query_index"})
print(df.shape)
df.sample(3)


# %%
# Pull all job results for a given original query result
for TIER in [1, 2, 3, 4]:
    result_disp_cols = [
        "orig_query_index",
        "path",
        "tier",
        "explanation"
    ]
    tier_sample = df[df['tier']==TIER].sample(1).iloc[0]
    print("TIER:", TIER)
    print("Path:", tier_sample['path'])
    print("Query as Sentence: ", tier_sample['query_sentence'])
    print("\nRanking explanation(s):")
    print(tier_sample['explanation'])
    print("------------\n")



# %%
df.groupby('tier')['path'].count()


# %%
df = df.sort_values('tier')
plt.figure(figsize=(20,10))
ax = sns.countplot(data=df, x='categories', hue='tier', palette='colorblind')
plt.xticks(rotation=45, ha='right')
ax.set_yscale("log")
plt.tight_layout()



# %%
# Spreadsheet-pastable output
for ii, row in df.groupby('tier', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 20))).iterrows():
    print(f"{row['path']}\t{row['tier']}\t{row['query_sentence']}\t{row['explanation']}")


# %%
def run_test_node_report(df, root_node_curie=None):
    """
    Detects the dataset via the first node curie, prints a tier-based occurrence report,
    and returns a DataFrame of hits in Tiers 3 & 4.
    """
    # Registry of test sets
    test_registry = {
        "CHEBI:13719": { # Acetylsalicylate -> Colorectal Cancer
            "COX2": ["NCBIGene:4513"],
            "NFkB": ["NCBIGene:4791", "NCBIGene:4790"],
            "resolvins": ["CHEBI:132120", "UMLS:C4288292"],
            "prostaglandin PGE2": ["CHEBI:26333", "CHEBI:176067", "CHEBI:15551"],
            "hMSH2": ["NCBIGene:4436"],
            "hMLH1": ["NCBIGene:4292"]
        },
        "CHEBI:5118": { # Fluoxetine -> Long Covid
            "IFNG": ["NCBIGene:3458"],
            "SIGMAR1": ["NCBIGene:10280"],
            "IRE1": ["NCBIGene:856478", "NCBIGene:2081"],
            "vagus nerve": ["UBERON:0001759"]
        },
        "CHEBI:45783": { # Imatinib -> Asthma
            "KIT": ["NCBIGene:3815"],
            "Histamine": ["CHEBI:18295", "PR:000049994"],
            "SCF-1": ["NCBIGene:4254"],
            "Mast Cell": ["CL:0000097"]
        }
    }

    # 1. Detect root node if not provided (pulling from first node of first path)
    if not root_node_curie:
        root_node_curie = df['node_curies'].iloc[0][0]
    
    expected_nodes = test_registry.get(root_node_curie)
    
    if not expected_nodes:
        print(f"No test set found for root node: {root_node_curie}")
        return pd.DataFrame()

    # Dynamic header using node names
    query_string = f"{df['node_names'].iloc[0][0]} --> {df['node_names'].iloc[0][-1]}"
    print(f"### Test Node Report: {query_string} ({root_node_curie}) ###\n")
    
    # Track indices for the returned DataFrame
    matched_indices = {}

    # 2. Generate report
    for name, curies in expected_nodes.items():
        mask = df['node_curies'].apply(lambda nc: any(c in nc for c in curies))
        
        # Capture indices of matches to map them back to names later
        for idx in df[mask].index:
            matched_indices.setdefault(idx, []).append(name)
        
        total = mask.sum()
        tier_counts = df[mask].groupby('tier').size().to_dict()
        tier_str = ", ".join([f"T{t}: {count}" for t, count in sorted(tier_counts.items())])
        
        print(f"{name:<20} | Total: {total:<4} | {tier_str}")

    # 3. Filter for results in Tiers 3 & 4 for error analysis
    low_tier_df = df[df['tier'].isin([3, 4])].copy()
    low_tier_df['matched_test_nodes'] = low_tier_df.index.map(lambda x: matched_indices.get(x, []))
    
    # Only return rows that actually contained a test node
    low_tier_df = low_tier_df[low_tier_df['matched_test_nodes'].map(len) > 0]

    return low_tier_df


# %%
# Print low-tier results containing test nodes
low_tier_hits = run_test_node_report(df)
for ii, row in low_tier_hits.iterrows():
    print(f"{row['path']}\t{row['tier']}\t{row['explanation']}")


# %%
# Custom sampling of high-tier results with specific test node
test_node_names = ["Histamine"]
for ii, row in df[df['node_names'].apply(lambda ll: all([tnn in ll for tnn in test_node_names]))].groupby('tier').get_group(1).sample(3).iterrows():
    print(f"{row['path']}\t{row['tier']}\t{row['explanation']}")


# %%
