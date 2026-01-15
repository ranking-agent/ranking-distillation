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
ORIG_QUERIES_DIR = TEST_SETS_DIR / "rare_disease_queries"
test_csvs = list(ORIG_QUERIES_DIR.glob("*.csv"))
print("Raw query results files:")
print(*test_csvs, sep="\n")

# Results from Gemini
RESULTS_DIR = TEST_SETS_DIR / "ranked_results" / "flash"
ranking_outputs = list(RESULTS_DIR.glob("*.parquet"))
print("\nRanked outputs avaialble:")
print(*ranking_outputs, sep="\n")

# Pick a specific job result file
# job_file = Path("2026-01-13-gemini-ranking-MONDO_0021020_to_NCBIGene_54658_3_hop_w_direction_paths-gemini-3-flash-preview-nr-1-bs-200-df-1-sb-0-gb-categories-rs-42.parquet")
# job_file = Path("2026-01-13-gemini-ranking-MONDO_0021020_to_NCBIGene_54658_3_hop_w_direction_paths-gemini-3-flash-preview-nr-1-bs-50-df-1-sb-0-gb-categories-rs-42.parquet")
job_file = Path("2026-01-13-gemini-labels-CHEBI_45783_to_MONDO_0004979_3_hop_w_direction_paths-gemini-3-flash-preview-prompt-v5-nr-1-bs-100-df-10-sb-0-gb-categories-rs-42.parquet")
selected_results = RESULTS_DIR / job_file
print("\nSelected:", selected_results.name)

# orig_dataset = [pp for pp in test_csvs if str(pp.name)[:25] == str(selected_results.name)[26:51]][0]
# print("\nMatched original dataset:", orig_dataset)


# %%
# Read in original query results
# df_orig = pd.read_csv(orig_dataset, sep="\t")
# df_orig = df_orig.reset_index(names="orig_query_index")
# df_orig = convert_results_to_sentences(df_orig).set_index("orig_query_index", drop=True)
# df_orig.sample(3)

# %%
# Read in ranking job results
df = pd.read_parquet(selected_results, engine="fastparquet")
df = df.rename(columns={"index": "orig_query_index"})
df = convert_results_to_sentences(df)
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
df.groupby('tier').count()

# %%
test_nodes = [
    "KIT",
    "Histamine",
    "SCF-1",
    "Mast Cell"
]

for test_node in test_nodes:
    print(test_node)
    
    total_count = sum([test_node in nn for nn in df['node_names']])
    print(total_count)

    tier_counts = df.groupby('tier').apply(lambda dfg: sum([test_node in nn for nn in dfg['node_names']]))
    print("Tier Counts:")
    print(tier_counts)
    print("\n")


# %%
test_curies = [
    "NCBIGene:3815",
    "CHEBI:18295",
    "PR:000049994",
    "NCBIGene:4254",
    "CL:0000097",
]
for test_curie in test_curies:
    print(test_curie)
    
    total_count = sum([test_curie in nc for nc in df['node_curies']])
    print(total_count)

    tier_counts = df.groupby('tier').apply(lambda dfg: sum([test_curie in nc for nc in dfg['node_curies']]))
    print("Tier Counts:")
    print(tier_counts)
    print("\n")

# %%
