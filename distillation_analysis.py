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
RESULTS_DIR = TEST_SETS_DIR / "ranked_results"
ranking_outputs = list(RESULTS_DIR.glob("*.parquet"))
print("\nRanked outputs avaialble:")
print(*ranking_outputs, sep="\n")

# Pick a specific job result file
# job_file = Path("MONDO_0021020_to_NCBIGene_54658_3_hop_w_direction_paths_ranked_batch_ddid0rt60yoomydk794iu6abjq30964z8x0k.parquet")
# job_file = Path("MONDO_0021020_to_NCBIGene_54658_3_hop_w_direction_paths_ranked_batch_grpuxy6whxotvg6e66q53a9lo421c8cv3nqd.parquet")
# job_file = Path ("MONDO_0021020_to_NCBIGene_54658_3_hop_w_direction_paths_ranked_batch_dsc62t5ssgwfd9cm8hhw1eoiyl9t95m3zaur.parquet")
job_file = Path ("MONDO_0021020_to_NCBIGene_54658_3_hop_w_direction_paths_ranked_batch_qzrhp409uijfk9pjbinttf9xklenchbi57fd.parquet")
selected_results = RESULTS_DIR / job_file
print("\nSelected:", selected_results.name)

orig_dataset = [pp for pp in test_csvs if str(pp.name)[:25] == str(selected_results.name)[:25]][0]
print("\nMatched original dataset:", orig_dataset)


# %%
# Read in original query results
df_orig = pd.read_csv(orig_dataset, sep="\t")
df_orig = df_orig.reset_index(names="orig_query_index")
df_orig = convert_results_to_sentences(df_orig).set_index("orig_query_index", drop=True)

# Read in ranking job results
df = pd.read_parquet(selected_results, engine="fastparquet")
df['reciprocal_rank'] = df.groupby("job_id")["rank"].transform(lambda x: x / x.max())
df = df.rename(columns={"index": "orig_query_index"})
df.sample(8)


# %%
# Group by original result and aggregate reciprocal rank (basically, generate mean reciprocal rank)
df_mrr = df.groupby('orig_query_index')['reciprocal_rank'].agg(['mean', 'median', 'min', 'max']).reset_index().sort_values("median")
df_mrr = df_mrr.join(df_orig, on="orig_query_index")
mrr_disp_cols = [
    "orig_query_index",
    "path",
    "categories",
    "median",
    "metapaths",
]
df_mrr[mrr_disp_cols].head(20)
# df_mrr.to_csv(RESULTS_DIR / )

# %%
# Pull all job results for a given original query result
for RANK_IDX in [0, 1, 2, 3, -3, -2, -1]:
    result_disp_cols = [
        "orig_query_index",
        "path",
        "rank",
        "explanation"
    ]
    one_result_set = df[df['orig_query_index']==df_mrr['orig_query_index'].iloc[RANK_IDX]][result_disp_cols]
    print("Rank:", RANK_IDX, "| MRR: %.5f" % df_mrr['median'].iloc[RANK_IDX])
    print("Path:", df_mrr['path'].iloc[RANK_IDX])
    print("Query as Sentence: ", df_mrr['query_sentence'].iloc[RANK_IDX])
    print("\nRanking explanation(s):")
    print(*one_result_set['explanation'], sep="\n")
    print("------------\n")


# %%
cat_counts = df_orig['categories'].value_counts(normalize=True)
df_cats = df_mrr.groupby("categories")['median'].agg(["median", "min", "max"]).reset_index().sort_values("median")
df_cats['weight'] = df['categories'].apply(lambda cc: np.round(cat_counts.loc[cc]*100, 3))
df_cats = df_cats.set_index('categories')
sorted_categories = df_cats.index
df['categories'] = pd.Categorical(df['categories'], categories=sorted_categories, ordered=True)
df = df.sort_values('categories')


# %%
df_cats.head(10)


# %%
df_cats.tail(8)


# %%
plt.figure(figsize=(18, 6))
ax = sns.boxplot(x='categories', y='reciprocal_rank', data=df, color='lightblue', showfliers=False)
ax = sns.stripplot(x='categories', y='reciprocal_rank', data=df, color='black', alpha=0.3, jitter=True, ax=ax)

plt.title('MRR Grouped by Category')
plt.xlabel('Node Metapaths')
plt.ylabel('Reciprocal Rank')
plt.xticks(rotation=60, ha='right')

plt.savefig(f'figures/{job_file.stem}.png')


# %%