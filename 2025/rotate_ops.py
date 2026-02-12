import numpy as np
import psycopg2 # Using standard postgres driver
import json # <-- Added this import


# --- 1. Database Helper Functions (Updated for your Schema) ---

def get_db_connection():
    """
    Placeholder for your database connection logic.
    Returns a psycopg2 connection object.
    """
    conn = None
    try:
        # Replace with your actual database credentials
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None


def reconstruct_complex_vector(db_vector):
    """
    Utility function to convert a flat (2k) real vector 
    from the DB into a (k) complex numpy array.
    
    Handles DB vector being a JSON string.
    """
    if db_vector is None:
        return None
    
    # --- THIS BLOCK IS NEW/MODIFIED ---
    # If the db_vector is a string (e.g., JSON), parse it first
    if isinstance(db_vector, str):
        try:
            db_list = json.loads(db_vector)
        except json.JSONDecodeError:
            print(f"Error: Could not decode embedding string: {db_vector[:50]}...")
            return None
    elif isinstance(db_vector, list):
        # If it's already a list (e.g., from a native psql array)
        db_list = db_vector
    else:
        # Try to handle other types, like tuples
        db_list = list(db_vector)
    # --- END MODIFIED BLOCK ---

    # Ensure it's a numpy array
    db_vector_np = np.array(db_list)
    
    # Calculate the original complex dimension (k)
    k = len(db_vector_np) // 2
    
    # Split the vector into its real and imaginary parts
    real_parts = db_vector_np[:k]
    imag_parts = db_vector_np[k:]
    
    # Reconstruct the complex vector
    return real_parts + 1j * imag_parts


def get_node_embedding(conn, curie):
    """
    Fetches a node embedding from the 'node_embeddings' table by its CURIE.
    Returns a complex numpy array.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT embedding FROM node_embeddings WHERE curie = %s", 
            (curie,)
        )
        row = cursor.fetchone()
        if row:
            # row[0] is the flat, concatenated embedding (as a string)
            return reconstruct_complex_vector(row[0])
    print(f"Warning: No node found for CURIE: {curie}")
    return None


def get_edge_embedding(
    conn,
    predicate,
    return_complex=True,
    ):
    """
    Fetches an edge embedding from the 'edge_embeddings' table by its predicate.
    
    NEW LOGIC: Assumes the DB stores a k-dim *phase* (theta) vector.
    It computes the complex vector from this phase.
    """
    with conn.cursor() as cursor:
        # Fetches the first embedding that matches the predicate
        cursor.execute(
            "SELECT embedding FROM edge_embeddings WHERE predicate = %s LIMIT 1",
            (predicate,)
        )
        row = cursor.fetchone()
        if not row:
            print(f"Warning: No edge found for predicate: {predicate}")
            return None
            
        # --- THIS BLOCK IS NEW/MODIFIED ---
        db_vector_str = row[0]
        
        # 1. Parse the JSON string to get the phase vector
        if isinstance(db_vector_str, str):
            try:
                phase_list = json.loads(db_vector_str)
            except json.JSONDecodeError:
                print(f"Error: Could not decode edge embedding string: {db_vector_str[:50]}...")
                return None
        else:
            phase_list = list(db_vector_str)
            
        phase_vector = np.array(phase_list) # This is a k-dim vector of angles (theta)
        
        if return_complex:
            # 2. Compute the complex vector
            # r = cos(theta) + i * sin(theta)
            real_parts = np.cos(phase_vector)
            imag_parts = np.sin(phase_vector)
            return real_parts + 1j * imag_parts
        else:
            return phase_vector


def find_closest_nodes_pgvector(conn, target_vector_complex, k=5):
    """ Finds the K-nearest neighbors using the pgvector index. """
    print(f"Finding top {k} closest nodes using pgvector...")
    
    # Deconstruct the complex vector into a 2k-dim real vector
    real_parts = target_vector_complex.real
    imag_parts = target_vector_complex.imag
    db_vector_query = np.concatenate([real_parts, imag_parts])
    db_vector_str = f"[{','.join(map(str, db_vector_query))}]"

    # This is the standard k-NN query
    sql = """
        SELECT curie, embedding <-> %s AS distance
        FROM node_embeddings
        ORDER BY embedding <-> %s  -- Order by the distance
        LIMIT %s
    """
    
    with conn.cursor() as cursor:
        # Note the three parameters:
        cursor.execute(sql, (db_vector_str, db_vector_str, k))
        return cursor.fetchall()
    

def get_ground_truth_treatments(jsonl_file_path, disease_curie):
    """
    Fetches the "ground truth" list of drugs known to treat a disease
    by reading a JSONL file.
    """
    print(f"Fetching ground truth treatments for {disease_curie} from {jsonl_file_path}...")
    ground_truth_set = set()
    
    try:
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                try:
                    triple = json.loads(line)
                    
                    # Check if the triple matches our ground truth criteria
                    if (triple.get('predicate') == 'biolink:treats'
                        and triple.get('object') == disease_curie
                        # and triple.get('subject', '').startswith(drug_prefix)
                        ):
                        
                        ground_truth_set.add(triple['subject'])
                        
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line[:50]}...")
        
        print(f"Found {len(ground_truth_set)} ground truth drugs.")
        return ground_truth_set
        
    except FileNotFoundError:
        print(f"\nCRITICAL ERROR: The triples file was not found at: {jsonl_file_path}")
        return None
    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not read the triples file.")
        print(f"Error: {e}\n")
        return None
    

# --- 2. The Validation Logic ---

def validate_rule(start_disease_curie):
    """
    Tests the rule: 
    (Drug) -> [biolink:targets] -> (Gene) -> [biolink:associated_with] -> (Disease)
    
    We start from the Disease and apply the rule in reverse.
    """
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to DB. Exiting.")
        return

    # 1. Define the rule components
    # We assume 'biolink:targets' is the predicate for (Drug, Gene)
    # and 'biolink:associated_with' is for (Gene, Disease)
    disease_predicate = "biolink:associated_with"
    drug_predicate = "biolink:targets"
    
    # We need a prefix to identify "Drug" nodes for the final search
    # This is an assumption, update it to match your CURIEs
    drug_prefix = "CHEMBL:" 
    
    # 2. Get the starting node and the relation embeddings
    print(f"Validating rule in reverse from: {start_disease_curie}")
    disease_vec = get_node_embedding(conn, start_disease_curie)
    r_assoc_with_vec = get_edge_embedding(conn, disease_predicate)
    r_targets_vec = get_edge_embedding(conn, drug_predicate)

    if disease_vec is None or r_assoc_with_vec is None or r_targets_vec is None:
        print("Error: Could not fetch all required embeddings for the rule.")
        conn.close()
        return

    # 3. Apply the rule in reverse using RotatE logic
    # The inverse of a rotation (relation) is its complex conjugate
    
    # Step 1: Go from Disease to Gene
    # Gene ≈ Disease o inverse(r_associated_with)
    r_assoc_with_inverse = np.conjugate(r_assoc_with_vec)
    predicted_gene_vec = disease_vec * r_assoc_with_inverse # Element-wise product

    # Step 2: Go from Gene to Drug
    # Drug ≈ Gene o inverse(r_targets)
    r_targets_inverse = np.conjugate(r_targets_vec)
    predicted_drug_vec = predicted_gene_vec * r_targets_inverse

    print("Calculated predicted drug vector.")

    # 4. Find the closest actual drug embeddings to our predicted vector
    closest_drugs = find_closest_nodes_slow(
        conn, 
        predicted_drug_vec, 
        node_type_prefix=drug_prefix, 
        k=5
    )

    print("\n--- Validation Results ---")
    print(f"Top 5 candidate drugs for '{start_disease_curie}' based on rule:")
    for curie, distance in closest_drugs:
        print(f"  - {curie} (Distance: {distance:.4f})")
        
    conn.close()

# --- Run the validation ---
if __name__ == "__main__":
    # You would test this with a known disease CURIE from your KG
    # Example using Asthma's MONDO CURIE
    validate_rule(start_disease_curie="MONDO:0004979")