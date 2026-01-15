# %%
from pathlib import Path
import pandas as pd
from query_compression import convert_results_to_sentences

# Load Data
TEST_SET_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/")
QUERIES_DIR = TEST_SET_DIR / "rare_disease_queries"
test_csvs = [
    "MONDO_0021020_to_NCBIGene_54658_3_hop_w_direction_paths.csv", # Crigler-Najjar
    "MONDO_0009897_to_NCBIGene_2632_3_hop_w_direction_paths.csv",
    "MONDO_0007931_to_NCBIGene_7439_3_hop_w_direction_paths.csv",
    "MONDO_0009672_to_NCBIGene_6606_3_hop_w_direction_paths.csv",
    "MONDO_0013166_to_NCBIGene_18_3_hop_w_direction_paths.csv",
    "MONDO_0011308_to_NCBIGene_617_3_hop_w_direction_paths.csv",
    "MONDO_0019353_to_NCBIGene_24_3_hop_w_direction_paths.csv",
    "MONDO_0010130_to_NCBIGene_1806_3_hop_w_direction_paths.csv",
    "MONDO_0003947_to_NCBIGene_959_3_hop_w_direction_paths.csv",
    "MONDO_0009653_to_NCBIGene_57192_3_hop_w_direction_paths.csv",
    "MONDO_0008224_to_NCBIGene_6329_3_hop_w_direction_paths.csv",
    "MONDO_0008692_to_NCBIGene_4547_3_hop_w_direction_paths.csv", # 17MB
]

for csv in test_csvs:
    pth = QUERIES_DIR / csv
    df = pd.read_csv(pth, sep="\t")
    df_sent = convert_results_to_sentences(df)
    for idx in [0, 1, -2, -1]:
        print("Path:", df_sent['path'].iloc[idx])
        print("Query as Sentence: ", df_sent['query_sentence'].iloc[idx])
        print("------------\n")

# %%
