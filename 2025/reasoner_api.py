import httpx
import json
from typing import List, Dict, Any, Optional
from reasoner_pydantic import Query, Message, QueryGraph, QNode, QEdge, Operation

def find_connections(
    start_curie: str,
    start_category: str,
    target_category: str,
    top_k: int = 20,
    edge_predicates: Optional[List[str]] = None,
    start_node_is_subject: bool = True,
    pretty_print_query: bool = False,
) -> List[Dict[str, Any]]:
    """
    Queries a TRAPI endpoint to find the top K nodes connected to a start node.

    Args:
        start_curie: The CURIE of the node you know (e.g., "MONDO:0005180").
        start_category: The biolink category of the start node (e.g., "biolink:Disease").
        target_category: The biolink category of the node(s) you want to find (e.g., "biolink:Drug").
        top_k: The maximum number of results to return.
        edge_predicates: Optional list of predicates. If None (default), the ARA will
                         perform an abstract, inferred query.
        start_node_is_subject: If True, the query is (start)-[edge]->(target).
                               If False, the query is (target)-[edge]->(start).
        pretty_print_query: If True, prints the formatted JSON query to the console
                            before sending.                               
    """
    
    # Define the "start" (bound) and "target" (unbound) nodes
    start_node_key = "n0_start"
    target_node_key = "n1_target"
    
    q_nodes = {
        start_node_key: QNode(ids=[start_curie], categories=[start_category]),
        target_node_key: QNode(categories=[target_category])
    }

    # Define the edge subject and object based on direction
    edge_subject = start_node_key if start_node_is_subject else target_node_key
    edge_object = target_node_key if start_node_is_subject else start_node_key

    # Create the QEdge. Only add predicates if they are provided.
    edge_params = {
        "subject": edge_subject,
        "object": edge_object
    }
    if edge_predicates:
        edge_params["predicates"] = edge_predicates
        print(f"Running SPECIFIC query with predicates: {edge_predicates}")
    else:
        print(f"Running GENERAL (inferred) query with no predicates.")

    q_edges = {
        "e01": QEdge(**edge_params)
    }

    # Build the full QueryGraph
    qgraph = QueryGraph(nodes=q_nodes, edges=q_edges)

    # Define the workflow to get only the top results
    workflow = [
        Operation(
            id="filter_results_top_n",
            parameters={"max_results": top_k}
        )
    ]

    # Build the full Query object
    query_payload = Query(
        message=Message(query_graph=qgraph),
        workflow=workflow
    )

    # Convert to a Python dictionary, excluding unset fields
    query_dict = query_payload.model_dump(exclude_unset=True)
    query_json = query_payload.model_dump_json(exclude_unset=True)

    # --- 4. Pretty Print (if requested) ---
    if pretty_print_query:
        print("\n--- 🖨️  Pretty-Printed TRAPI Query ---")
        print(json.dumps(query_dict, indent=2))
        print("---------------------------------------\n")

    # Send the request to the RoboKop API endpoint
    robokop_url = "https://aragorn.renci.org/robokop/query?answer_coalesce_type=all"
    
    print(f"Sending query to RoboKop for {start_curie}...")
    
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(robokop_url, content=query_json)
            
        if response.status_code != 200:
            print(f"Error: API returned status {response.status_code}")
            print(response.text)
            return []

        # Parse the response
        response_data = response.json()
        response_message = Message.model_validate(response_data['message'])

        print("Successfully retrieved results.")

        # Extract the ground truth nodes
        ground_truths = []
        if not response_message.results:
            print("No results found.")
            return []

        kg = response_message.knowledge_graph
        for result in response_message.results:
            # Get the CURIE for the bound target node
            target_binding = result.node_bindings[target_node_key][0]
            target_curie = target_binding.id
            
            if target_curie in kg.nodes:
                target_name = kg.nodes[target_curie].name
                ground_truths.append({
                    "curie": target_curie,
                    "name": target_name
                })
            
        return ground_truths

    except httpx.RequestError as e:
        print(f"An error occurred while requesting {e.request.url!r}: {e}")
        return []


# --- Run the example ---
if __name__ == "__main__":

    print("--- ☕ Running the Foolproof Caffeine Test ---")
    
    caffeine_targets = find_connections(
        start_curie="CHEMBL.COMPOUND:CHEMBL113",
        start_category="biolink:Drug",
        target_category="biolink:Gene",
        edge_predicates=None, # General query
        start_node_is_subject=True, # (Caffeine) -> (Gene)
        top_k=5,
        pretty_print_query=True,
    )
    
    if caffeine_targets:
        print(f"\n--- Top {len(caffeine_targets)} Caffeine-Related Genes ---")
        for i, gene in enumerate(caffeine_targets):
            print(f"{i+1}. {gene['name']} ({gene['curie']})")
    else:
        print("\n--- TEST FAILED: No results returned for Caffeine. ---")
        print("This suggests a problem with the RoboKop server or a network issue.")

    diseases = [
        "UMLS:C0042164",
        "UMLS:C0729587",
        "UMLS:C0029967",
        "UMLS:C0042345",
        "UMLS:C0271066",
        "UMLS:C1562543",
        "UMLS:C0271084",
        "UMLS:C0149871",
        "UMLS:C0034065",
        "UMLS:C0020443",
        "UMLS:C0011854",
        "UMLS:C0011860",
        "UMLS:C0238080",
        "UMLS:C0041327",
    ] # Parkinson's disease
    
    # ---
    # OPTION 1: The GENERAL (Inferred) Query - This should work!
    # ---
    # This asks: "RoboKop, what Drugs are related to Parkinson's? Infer the relationship."
    # We set start_node_is_subject=False because we want (Drug)-[...]->(Disease)
    # So, the "start" node (Disease) is the OBJECT.
    
    # for disease in diseases:
    #     print("--- 1. Running GENERAL Inferred Query ---")
    #     general_treatments = find_connections(
    #         start_curie=disease,
    #         start_category="biolink:Disease",
    #         target_category="biolink:Drug",
    #         edge_predicates=None, # <-- THE KEY
    #         start_node_is_subject=False, # (Drug) is subject, (Disease) is object
    #         top_k=20
    #     )
        
    #     if general_treatments:
    #         print(f"\n--- Top {len(general_treatments)} Inferred Treatments for {disease} ---")
    #         for i, drug in enumerate(general_treatments):
    #             print(f"{i+1}. {drug['name']} ({drug['curie']})")
        
    #     print("\n" + "="*80 + "\n")

    #     # ---
    #     # OPTION 2: The INVERSE Predicate Query - Good to check
    #     # ---
    #     # This asks: "Find literal 'treated_by' edges"
    #     # Query: (Disease)-[treated_by]->(Drug)
        
    #     print("--- 2. Running SPECIFIC Inverse Query ('treated_by') ---")
    #     inverse_treatments = find_connections(
    #         start_curie=disease,
    #         start_category="biolink:Disease",
    #         target_category="biolink:Drug",
    #         edge_predicates=["biolink:treated_by"], # <-- Specific
    #         start_node_is_subject=True, # (Disease) is subject, (Drug) is object
    #         top_k=20
    #     )

    #     if inverse_treatments:
    #         print(f"\n--- Top {len(inverse_treatments)} 'treated_by' Treatments for {disease} ---")
    #         for i, drug in enumerate(inverse_treatments):
    #             print(f"{i+1}. {drug['name']} ({drug['curie']})")