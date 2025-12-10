import httpx
import json
from typing import Optional, List, Dict, Any

def get_normalized_curie(
    input_curie: str, 
    preferred_prefix: Optional[str] = None,
    normalizer_url="https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes",
    verbose=False,
) -> Optional[str]:
    """
    Queries the Translator Node Normalization API to find the
    canonical CURIE for a concept.

    Args:
        input_curie: The CURIE to normalize (e.g., "UMLS:C0030567").
        preferred_prefix: If provided, will try to find an equivalent CURIE
                          with this prefix if the canonical one doesn't match.
                          Note: Case-sensitive (e.g., "MONDO").

    Returns:
        The canonical (or preferred) CURIE (e.g., "MONDO:0005180"),
        or None if no equivalent is found or an error occurs.
    """
    
    # This is the standard public endpoint for the Node Normalizer
    
    # The API takes the CURIEs as a 'curie' query parameter
    params = {"curie": input_curie}
    
    if verbose:
        print(f"Querying Node Normalizer for: {input_curie}")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(normalizer_url, params=params)
            
        if response.status_code != 200:
            print(f"Error: API returned status {response.status_code}")
            print(response.text)
            return None

        # --- 1. Parse the Response ---
        response_data = response.json()

        if not response_data or input_curie not in response_data or response_data[input_curie] is None:
            print(f"Error: CURIE '{input_curie}' not found by Node Normalizer.")
            return None
            
        node_data = response_data[input_curie]
        
        # --- 2. Get the Canonical (Preferred) ID ---
        # The 'id' field holds the preferred CURIE for this concept.
        canonical_id_obj = node_data.get("id", {})
        canonical_curie = canonical_id_obj.get("identifier")
        canonical_id_label = canonical_id_obj.get("label")
        
        if not canonical_curie:
            print(f"Error: Node data found, but it has no canonical identifier.")
            return None
        
        # --- 3. Handle Preferred Prefix (if provided) ---
        if preferred_prefix:
            # Check if canonical already matches the preference
            if canonical_curie.startswith(preferred_prefix + ":"):
                if verbose:
                    print(f"Found canonical match with preferred prefix: {canonical_curie}")
                return canonical_curie
            
            # If not, search equivalents for the preferred prefix
            equivalents = node_data.get("equivalent_identifiers", [])
            for eq_id_obj in equivalents:
                equivalent_curie = eq_id_obj.get("identifier")
                if equivalent_curie and equivalent_curie.startswith(preferred_prefix + ":"):
                    if verbose:
                        print(f"Found equivalent match with preferred prefix: {equivalent_curie}")
                    return equivalent_curie
            
            print(f"Warning: Preferred prefix '{preferred_prefix}' not found.")
            # Fallback to returning the canonical CURIE anyway
            print(f"Returning canonical CURIE instead: {canonical_curie}")
            return canonical_curie

        # --- 4. Return Canonical ID (default behavior) ---
        if verbose:
            print(f"Found canonical CURIE: {canonical_curie}")
        return canonical_curie, canonical_id_label

    except httpx.RequestError as e:
        print(f"An error occurred while requesting {e.request.url!r}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON response from server.")
        return None

# --- Run Examples ---
if __name__ == "__main__":
    
    umls_curie = "UMLS:C0030567" # Parkinson's Disease

    # --- Example 1: Get canonical CURIE (default) ---
    print("--- Example 1: Get Canonical ID for Parkinson's ---")
    canonical = get_normalized_curie(umls_curie)
    if canonical:
        print(f"  SUCCESS: {umls_curie}'s canonical ID is {canonical}\n")
    else:
        print(f"  FAILED: Could not normalize {umls_curie}.\n")

    # --- Example 2: Get with a preferred prefix (that matches canonical) ---
    print("--- Example 2: Get with 'MONDO' prefix ---")
    mondo_curie = get_normalized_curie(umls_curie, preferred_prefix="MONDO")
    if mondo_curie:
        print(f"  SUCCESS: Found preferred MONDO: {mondo_curie}\n")
    else:
        print(f"  FAILED: Could not find MONDO for {umls_curie}.\n")

    # --- Example 3: Get with a different preferred prefix (non-canonical) ---
    print("--- Example 3: Get with 'MESH' prefix ---")
    mesh_curie = get_normalized_curie(umls_curie, preferred_prefix="MESH")
    if mesh_curie:
        print(f"  SUCCESS: Found preferred MESH: {mesh_curie}\n")
    else:
        print(f"  FAILED: Could not find MESH for {umls_curie}.\n")
        
    # --- Example 4: A "failed" lookup ---
    print("--- Example 4: Fake CURIE ---")
    fake_curie = "UMLS:NOT_A_REAL_ID"
    failed_curie = get_normalized_curie(fake_curie)
    if not failed_curie:
        print(f"  SUCCESS: Correctly failed to find {fake_curie}\n")