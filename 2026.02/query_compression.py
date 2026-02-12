# %%
from pathlib import Path
import pandas as pd


def query_result_to_sentence(
    rd,
    pred_fields=[
        'first_hop_predicates',
        'second_hop_predicates',
        'third_hop_predicates',
    ],
):
    # Build a sentence like:
    # <Node> (a <node type>), either <pred 1> or <pred 2> <Node> (a <node type>), which <pred 3> <Node>... etc.
    query_str = ""
    for ii, (nn, nt, pf) in enumerate(zip(rd['node_names'][:-1], rd['node_types'][:-1], pred_fields)):
        
        # Crigler-Najjar syndrome type 1 (a Disease) 
        ntc = nt.replace('biolink:', '')
        if ii == 0:
            query_str += f"{nn} (a {ntc}) "
        else:
            # ... studied to treat coffee (a MolecularMixture), which ...
            query_str += f"{nn} (a {ntc}), which "
        
        preds = rd[pf]
        if len(preds) == 1:
            pred = preds[0]
            pred_str = pred.replace('biolink:', '').replace('_', ' ')
            # ... has phenotype ...
            query_str += f"{pred_str} "

        else:
            # ... either treats or affects ...
            query_str += f"either ["
            for pred in preds[:-1]:
                pred_str = pred.replace('biolink:', '').replace('_', ' ')
                query_str += f"{pred_str} or "
            query_str += f"{preds[-1].replace('biolink:', '').replace('_', ' ')}] "

    query_str += f"{rd['node_names'][-1]} (a {rd['node_types'][-1].replace('biolink:', '')})."

    return query_str


def convert_results_to_sentences(
    df_orig
):
    new_res = []
    pred_fields = [
        'first_hop_predicates',
        'second_hop_predicates',
        'third_hop_predicates',
    ]
    for ii, sr in df_orig.iterrows():
        rd = {}
        for kk, vv in sr.to_dict().items():
            try:
                rd[kk] = eval(vv)
            except:
                rd[kk] = vv
        rd['node_names'] = rd['path'].split(" -> ")
        rd['node_curies'] = rd['path_curies'].split(' --> ')
        # Drop anything with circuitous paths
        if rd['node_curies'][0] in rd['node_curies'][1:] or rd['node_curies'][-1] in rd['node_curies'][:-1]:
            continue
        rd['node_types'] = rd['categories'].split(' --> ')
        for pf in pred_fields:
            rd[pf] = list(rd[pf])
        rd['query_sentence'] = query_result_to_sentence(rd)
        # print(rd['query_sentence'])
        new_res.append(rd)

    return pd.DataFrame(new_res)


# %%
if __name__ == "__main__":

    # Base NAS dir
    TEST_SETS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/")
    ORIG_QUERIES_DIR = TEST_SETS_DIR / "rare_disease_queries"
    test_csvs = list(ORIG_QUERIES_DIR.glob("*.csv"))
    print("Raw query results files:")
    print(*test_csvs, sep="\n")

    orig_dataset = test_csvs[0]
    print("\nSelected:\n", orig_dataset)

    # Read in original query results
    df_orig = pd.read_csv(orig_dataset, sep="\t")
    df_orig = df_orig.reset_index(names="query_index")
    new_res = convert_results_to_sentences(df_orig)
    df_sent = new_res
    df_sent

# %%
