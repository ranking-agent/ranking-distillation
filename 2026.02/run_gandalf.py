import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import httpx
import pandas as pd

shepherd_url = "https://shepherd.renci.org"
# shepherd_url = "http://localhost:5439"


def generate_query(curie1: str, curie2: str):
    """Given a curie, return a TRAPI message."""
    return {
        "workflow": [
            {
                "id": "aragorn.lookup"
            }
        ],
        "message": {
            "query_graph": {
                "nodes": {
                "n0": {
                    "ids": [curie1]
                },
                "n1": {},
                "n2": {},
                "n3": {
                    "ids": [curie2]
                }
                },
                "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:related_to"]
                },
                "e1": {
                    "subject": "n1",
                    "object": "n2",
                    "predicates": ["biolink:related_to"]
                },
                "e2": {
                    "subject": "n2",
                    "object": "n3",
                    "predicates": ["biolink:related_to"]
                }
                }
            }
        },
        "parameters": {
            "gandalf": True
        }
    }


def single_lookup(curie1: str, curie2: str, target: str, save_dir: Path = Path(".")):
    """Run a single query lookup synchronously."""
    query = generate_query(curie2, curie1)
    start_time = datetime.now()
    try:
        with httpx.Client(timeout=600000) as client:
            response = client.post(
                f"{shepherd_url}/{target}/query",
                json=query,
            )
            response.raise_for_status()
            response = response.json()
            num_results = len((response.get("message") or {}).get("results") or [])
    except Exception as e:
        num_results = 0
        response = {
            "Error": str(e),
        }

    stop_time = datetime.now()
    print(f"{curie1}->{curie2} took {stop_time - start_time} seconds and gave {num_results} results")
    save_path = save_dir / f"pathfinder_{('_').join(curie1.split(':'))}_{('_').join(curie2.split(':'))}_response.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(response, f)
    print("Saved to", save_path)
    return save_path


def parse_trapi_to_paths(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    kg = data['message']['knowledge_graph']
    results = data['message'].get('results', [])
    
    # 1. Lookups for quick access
    node_names = {k: v.get('name', k) for k, v in kg['nodes'].items()}
    node_cats = {k: v.get('categories', ['biolink:NamedThing'])[0] for k, v in kg['nodes'].items()}
    
    # Map edges: {edge_id: (subject, predicate, object)}
    edge_map = {}
    for eid, info in kg['edges'].items():
        edge_map[eid] = {
            'sub': info['subject'],
            'obj': info['object'],
            'pred': info['predicates'][0] if info.get('predicates') else 'biolink:related_to'
        }

    path_data = []

    # 2. Reconstruct paths from results
    for res in results:
        # TRAPI results map query nodes (n0, n1...) to KG nodes
        nb = res['node_bindings']
        eb = res['edge_bindings']
        
        # Get Curies for the path (assumes path order n0 -> n1 -> n2 -> n3)
        try:
            c0, c1, c2, c3 = nb['n0'][0]['id'], nb['n1'][0]['id'], nb['n2'][0]['id'], nb['n3'][0]['id']
            
            # Get Predicates for each hop
            # (Finding which edge connects n0 to n1, etc.)
            p1 = {edge_map[e['id']]['pred'] for e in eb['e0']}
            p2 = {edge_map[e['id']]['pred'] for e in eb['e1']}
            p3 = {edge_map[e['id']]['pred'] for e in eb['e2']}
            
            # Formatting strings for your specific CSV columns
            path_str = f"{node_names[c0]} -> {node_names[c1]} -> {node_names[c2]} -> {node_names[c3]}"
            categories_str = f"{node_cats[c0]} --> {node_cats[c1]} --> {node_cats[c2]} --> {node_cats[c3]}"
            curies_str = f"{c0} --> {c1} --> {c2} --> {c3}"
            
            # Calculate metapaths (one for each combination of predicates if multiple exist)
            metapaths = []
            for pred1 in p1:
                for pred2 in p2:
                    for pred3 in p3:
                        metapaths.append(f"{node_cats[c0]} ---{pred1}--> {node_cats[c1]} ---{pred2}--> {node_cats[c2]} ---{pred3}--> {node_cats[c3]}")

            path_data.append({
                'path': path_str,
                'num_paths': len(metapaths),
                'categories': categories_str,
                'first_hop_predicates': p1,
                'second_hop_predicates': p2,
                'third_hop_predicates': p3,
                'has_gene': 'biolink:Gene' in categories_str,
                'metapaths': metapaths,
                'path_curies': curies_str
            })
        except KeyError:
            continue # Skip results that don't fit the 4-node path pattern

    return pd.DataFrame(path_data)


pathfinder_tests = [
    ("MONDO:0004979", "CHEBI:45783"),
    ("CHEBI:27881", "NCBIGene:2739"),
    ("MONDO:0005011", "MONDO:0005180"),
    ("NCBIGene:54716", "MONDO:0100096"),
    ("NCBIGene:3458", "MONDO:0100096"),
    ("NCBIGene:3458", "CHEBI:16828"),
    ("CHEBI:9139", "MONDO:0004975"),
    ("CHEBI:13719", "MONDO:0005575"),
    ("CHEBI:5118", "MONDO:0100233"),
    ("MONDO:0005180", "MONDO:0005105"),
    ("CHEBI:10033", "MONDO:0004992"),
    ("CHEBI:28364", "MONDO:0005311"),
    ("NCBIGene:5328", "NCBIGene:4982"),
    ("CHEBI:7465", "MONDO:0008218"),
    ("MONDO:0019632", "MONDO:0005340"),
    ("GO:0006914", "MONDO:0005265"),
    ("CHEBI:15647", "UNII:31YO63LBSN"),
    ("UNII:7SE5582Q2P", "DOID:4480"),
    ("CHEBI:3750", "MONDO:0013209"),
    ("CHEBI:83766", "MONDO:0008170"),
    ("CHEBI:50924", "MONDO:0007256"),
]


def main():
    """Run the given query and time it."""
    start = time.time()
    save_dir = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/gandalf_outputs")
    for test in tqdm(pathfinder_tests[1:]):
        print(f"\nRunning {test}...")
        save_path = single_lookup(test[0], test[1], "aragorn", save_dir=save_dir)
        df_test = parse_trapi_to_paths(save_path)
    print(f"All queries took {time.time() - start} seconds")


if __name__ == "__main__":
    main()
 