# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# %%
TEST_SETS_DIR = Path("/mnt/nas0_data0/projects/translator/users/npersson/pathfinder_test_sets/")
RESULTS_DIR = TEST_SETS_DIR / "ranked_results"
test_csvs = list(TEST_SETS_DIR.glob("*.csv"))
ranking_outputs = list(RESULTS_DIR.glob("*.parquet"))
print(*ranking_outputs, sep="\n")
selected_results = ranking_outputs[-1]
print("\nSelected:", selected_results.name)
orig_dataset = [pp for pp in test_csvs if str(pp.name)[:25] == str(selected_results.name)[:25]][0]
print("\nMatched original dataset:", orig_dataset)


# %%
df_orig = pd.read_csv(orig_dataset, sep="\t")

df = pd.read_parquet(selected_results, engine="fastparquet")
df['rank_frac'] = df.groupby("job_id")["rank"].transform(lambda x: 1 - x / x.max())
df.sample(8)


# %%
cat_counts = df_orig['categories'].value_counts(normalize=True)
df_cats = df.groupby("categories")['rank'].agg(["median", "min", "max"]).reset_index().sort_values("median")
df_cats['weight'] = df['categories'].apply(lambda cc: np.round(cat_counts.loc[cc]*100, 3))
df_cats = df_cats.set_index('categories')
sorted_categories = df_cats.index
df['categories'] = pd.Categorical(df['categories'], categories=sorted_categories, ordered=True)
df = df.sort_values('categories')


# %%
df_cats.head(8)


# %%
df_cats.tail(8)


# %%
plt.figure(figsize=(18, 6))
ax = sns.boxplot(x='categories', y='rank_frac', data=df, color='lightblue', showfliers=False)
ax = sns.stripplot(x='categories', y='rank_frac', data=df, color='black', alpha=0.3, jitter=True, ax=ax)

plt.title('Node Metapath Member Rankings within Chunk (Warafarin -> Cancer)')
plt.xlabel('Node Metapaths')
plt.ylabel('Fractional Rank within Cohort')
plt.xticks(rotation=60, ha='right')

plt.savefig('figures/node_metapath_ranking.png')


# %%