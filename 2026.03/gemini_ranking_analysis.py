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
# TEST_SETS_DIR = Path("/home/npersson/data/pathfinder_test_sets/")

# Original results from Neo4j
ORIG_QUERIES_DIR = TEST_SETS_DIR / "gandalf_responses_predicates_with_inverse"
test_csvs = list(ORIG_QUERIES_DIR.glob("*.csv"))
print("Raw query results files:")
print(*test_csvs, sep="\n")

# Results from Gemini
RESULTS_DIR = ORIG_QUERIES_DIR / "ranking_outputs_flash_3p1_10k"
ranking_outputs = list(RESULTS_DIR.glob("*.parquet"))
print("\nRanked outputs avaialble:")
print(*ranking_outputs, sep="\n")


# %%
# Pick a specific job result file
selected_results = ranking_outputs[1]
# selected_results = RESULTS_DIR / job_file
print("\nSelected:", selected_results.name)


# %%
# Read in ranking job results
df = pd.read_parquet(selected_results, engine="fastparquet")
df = df.rename(columns={"index": "orig_query_index"})
df.sample(3)


# %%
df.groupby('tier')['path'].count().plot(kind='bar')
plt.title('Count of Paths by Tier')
plt.xlabel('Tier')
plt.ylabel('Count')


# %%
df = df.sort_values('tier')
plt.figure(figsize=(18,12))
sns.countplot(data=df, x='categories', hue='tier', palette='colorblind')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


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
disp_cols = [
    # 'orig_query_index',
    'path',
    'tier',
    'query_sentence',
    'explanation',
]
print(df.groupby('tier').apply(lambda x: x.sample(min(len(x), 15))).reset_index(drop=True)[disp_cols].to_csv(sep='\t', index=False))


# %%
test_curies = [
    "NCBIGene:3815", # KIT
    "CHEBI:18295", # Histamine
    "PR:000049994", # Histamine
    "NCBIGene:4254", # SCF-1
    "CL:0000097", # Mast Cell
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
