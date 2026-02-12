# %%
from pathlib import Path
import pandas as pd
from query_compression import convert_results_to_sentences

# %%
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
TEST_SETS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/")
ORIG_QUERIES_DIR = TEST_SETS_DIR / "training_queries"
test_csvs = list(ORIG_QUERIES_DIR.glob("*.csv"))
print("Raw query results files:")
print(*test_csvs, sep="\n")

# Pick a specific job result file
job_file = Path("CHEBI_13719_to_MONDO_0005575_3_hop_w_direction_paths.csv")
# job_file = Path("CHEBI_5118_to_MONDO_0100233_3_hop_w_direction_paths.csv")
# job_file = Path("asthma_results_subset.csv")
raw_data_path = ORIG_QUERIES_DIR / job_file

df = pd.read_csv(raw_data_path, sep='\t')

# %%
for csv in test_csvs:
    pth = ORIG_QUERIES_DIR / csv
    print(pth)
    df = pd.read_csv(pth, sep="\t")
    print(f"{csv.stem} has shape {df.shape}")
    df_sent = convert_results_to_sentences(df.sample(1))
    for ii, row in df_sent.iterrows():
        print("Path:", row['path'])
        print("Query as Sentence: ", row['query_sentence'])
        print("------------\n")

# %%
