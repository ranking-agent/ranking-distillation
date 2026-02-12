# %%
import pandas as pd
from pathlib import Path
from query_compression import convert_results_to_sentences

# %%
# Load Data
TEST_SET_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets")
# QUERIES_DIR = TEST_SET_DIR / "rare_disease_queries"
# test_csvs = [
#     "MONDO_0021020_to_NCBIGene_54658_3_hop_w_direction_paths.csv", # Crigler-Najjar
#     # "MONDO_0009897_to_NCBIGene_2632_3_hop_w_direction_paths.csv",
#     # "MONDO_0007931_to_NCBIGene_7439_3_hop_w_direction_paths.csv",
#     # "MONDO_0009672_to_NCBIGene_6606_3_hop_w_direction_paths.csv",
#     # "MONDO_0013166_to_NCBIGene_18_3_hop_w_direction_paths.csv",
#     # "MONDO_0011308_to_NCBIGene_617_3_hop_w_direction_paths.csv",
#     # "MONDO_0019353_to_NCBIGene_24_3_hop_w_direction_paths.csv",
#     # "MONDO_0010130_to_NCBIGene_1806_3_hop_w_direction_paths.csv",
#     # "MONDO_0003947_to_NCBIGene_959_3_hop_w_direction_paths.csv",
#     # "MONDO_0009653_to_NCBIGene_57192_3_hop_w_direction_paths.csv",
#     # "MONDO_0008224_to_NCBIGene_6329_3_hop_w_direction_paths.csv",
#     # "MONDO_0008692_to_NCBIGene_4547_3_hop_w_direction_paths.csv", # 17MB
# ]
QUERIES_DIR = TEST_SET_DIR / "training_queries"
test_csvs = [
    "CHEBI_45783_to_MONDO_0004979_3_hop_w_direction_paths.csv", # imatinib -> asthma
    "CHEBI_13719_to_MONDO_0005575_3_hop_w_direction_paths.csv", # acetylsalicylate -> colorectal cancer
    "CHEBI_5118_to_MONDO_0100233_3_hop_w_direction_paths.csv", # Fluoxetine -> Long COVID
]

# %%
for 


# %%
pth = QUERIES_DIR / test_csvs[2]
print(f"Loading data from: {pth}")
df = pd.read_csv(pth, sep="\t")


# %%
# Subset to ensure we have all data to pass test plus a lot of extra noise
test_curies = [
    "NCBIGene:3815", # KIT
    "CHEBI:18295", # histamine
    "PR:000049994", # histamine
    "NCBIGene:4254", # SCF-1
    "CL:0000097", # Mast Cell
]

contains_test_curie = lambda pc: any([tc in pc for tc in test_curies])

df['has_test_curies'] = df['path_curies'].apply(contains_test_curie)


# %%
df_passing = df[df['has_test_curies']].copy()
df_not_passing = df[~df['has_test_curies']].copy()
df_strat = df_not_passing.groupby('categories').sample(frac=0.1)
df_hack = pd.concat([df_passing, df_strat])


# %%
df_hack = convert_results_to_sentences(df_hack)

# %%
df_hack.to_csv("~/data/pathfinder_test_sets/hackathon/asthma_results_subset.csv", sep="\t")
# %%
