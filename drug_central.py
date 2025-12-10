import polars as pl

DRUG_CENTRAL_PATH = "/mnt/nas0/projects/translator/users/npersson/drug_central.tsv"


def load_drug_central_df(
        file_path=DRUG_CENTRAL_PATH
    ):
    try:
        df = pl.read_csv(
            DRUG_CENTRAL_PATH,
            separator='\t',
            # columns=["_id", "chebi", "drug_umls", "disease_umls", "drug_name", "disease_name"]
        )
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        # Handle error (e.g., file not found, incorrect separator)
        exit()


def get_top_treatments(
    df,
    disease_umls: str,
    top_n: int = 10
    ):
    """
    Finds the top N drugs for a given disease using Polars.
    """
    try:
        result = (
            df.filter(pl.col("disease_umls") == disease_umls)
            .group_by("drug_umls")
            .len()
            .sort("len", descending=True)
            .head(top_n)
        )
        return result['drug_umls'].to_list()
    except Exception as e:
        print(f"Error during query: {e}")
        return None


if __name__ == "__main__":

    # --- 1. One-Time Setup ---
    # Load the TSV into a Polars DataFrame.
    # This is the "startup cost." It will be fast, but not instant.
    df = load_drug_central_df()

    # --- 3. Run Your Queries ---
    disease_input = "UMLS:C0022568"  # Example disease
    top_drugs = get_top_treatments(df, disease_input, top_n=5)

    if top_drugs is not None:
        print(f"\nTop 5 drugs for '{disease_input}':")
        print(top_drugs)