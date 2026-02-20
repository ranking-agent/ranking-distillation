# %%
from pathlib import Path
import pandas as pd

# Load Data
TEST_SET_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets")

QUERIES_DIR = TEST_SET_DIR / "gandalf_responses_related"
test_csvs = list(QUERIES_DIR.glob("*.tsv"))
print(*test_csvs, sep="\n")


# %%
for cc in test_csvs:
    df = pd.read_csv(cc, sep="\t")
    print(df.shape)
    if df.shape[0]>0:
        print(df['path'].iloc[0])

# %%
df = pd.read_csv(test_csvs[2], sep="\t")
df
# %%
