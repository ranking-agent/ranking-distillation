# %%
import os
from pathlib import Path
import pandas as pd
from google import genai
from dotenv import load_dotenv
load_dotenv("/home/npersson/.env")

# %%
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

# response = client.models.generate_content(
#     model="gemini-2.5-pro", contents="Explain how AI works in a few words"
# )
# print(response.text)

CONTEXT = """
I'm working on a module called "pathfinder" for NIH biomedical data Translator. Translator is backed by a knowledge graph assembled from NIH data sources, primarily linking drugs, genes and diseases. Pathfinder is a UI feature where the user specifies two endpoints (drug, disease, gene, etc.), and it's supposed to surface the "most interesting" paths between the two.      

Here's example results from a query linking two nodes. Currently, the tool returns an exhaustive list of all paths between the two nodes with a max of 3 hops. Each of these results is from a different "category" (= meta-path of node types).      

Rank these in terms of their level of novelty / "interestingness" (as opposed to being obvious, trivial, etc.) and generate a brief explanation of the ranking. Return valid JSON with fields "rank", "index", "path", "explanation", and nothing else. Every result must be a member of the returned JSON.

"""


# %%
TEST_SET_DIR = Path("/mnt/nas0_data0/projects/translator/users/npersson/pathfinder_test_sets/")
test_csvs = list(TEST_SET_DIR.glob("*.csv"))
print(*test_csvs, sep="\n")


# %%
pth = test_csvs[2]
df = pd.read_csv(pth, sep="\t")
# df[['node1', 'node2', 'node3', 'node4']] = df['categories'].str.split("-->", expand=True)
df.describe()


# %%
df_catsort = df.sort_values('categories')
print("Num unique node metapaths:", df_catsort['categories'].nunique())
# Show categories with the most results
df_catsort.groupby('categories')['categories'].count().sort_values(ascending=False)


# %%
cat_samples = []
for cat, df_cat in df_catsort.groupby('categories'):
    cat_samples.append(df_cat.sample(1))

df_test_cats = pd.concat(cat_samples)
json_data = df_test_cats.to_json(orient='records')
df_test_cats.reset_index().to_json('data/test_cats.json', orient='records', indent=True)

# %%
prompt = CONTEXT + json_data
response = client.models.generate_content(
    # model="gemini-2.5-pro",
    model="gemini-3-pro-preview",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        # "response_json_schema": Recipe.model_json_schema(),
    }
)
print(response.text)


# %%