# %%

import sys
sys.path.append('..')

import numpy as np
from rotate_ops import *
from drug_central import *
from node_norm import *

import matplotlib.pyplot as plt


# %%
TRIPLES_FILE_PATH = "/home/npersson/code/translator/aragorn/edges.jsonl"
DRUG_CENTRAL_PATH = "/mnt/nas0/projects/translator/users/npersson/drug_central.tsv"
df_dc = load_drug_central_df()


####################################
### TEST REVERSE ROTATIONS       ###
####################################
# start_disease_umls = 'UMLS:C2585945' # This one HITS
start_disease_umls = df_dc['disease_umls'].sample(1)[0]
start_disease_curie, start_disease_label = get_normalized_curie(start_disease_umls)
print("\nStarting from:", start_disease_umls)
print("Normalized Curie for start disease:", start_disease_curie, "|", start_disease_label)

# Get ground truth from drug central
ground_truth_umls = get_top_treatments(df_dc, start_disease_umls)
normalized_ground_truths = [
    get_normalized_curie(curie) for curie in ground_truth_umls
]
ground_truth_curies = [gt[0] for gt in normalized_ground_truths]
print("\nFound treatments:")
print(*normalized_ground_truths, sep='\n')

# Get the normalized Curie (not UMLS)

# %%
# Use the triples to find subjects that "treat" the specified disease
# ground_truth_set = get_ground_truth_treatments(TRIPLES_FILE_PATH, start_disease_curie)
# print(ground_truth_set)

# Get embeddings for the specified diesease
conn = get_db_connection()
start_disease_emb = get_node_embedding(conn, start_disease_curie)
print("Retrieved disease embedding")
# print(start_disease_emb)


# %%

# 1. Define the rule components
# We assume 'biolink:targets' is the predicate for (Drug, Gene)
# and 'biolink:associated_with' is for (Gene, Disease)
correlated_predicate = "biolink:correlated_with"
drug_predicate = "biolink:treats"

# Limit final drug search just to CHEMBL nodes
drug_prefix = "CHEMBL:"

# Number of neighbors to find from resulting node
K = 20

# 2. Get the starting node and the relation embeddings
print(f"Validating rule in reverse from: {start_disease_curie}")
disease_vec = get_node_embedding(conn, start_disease_curie)
r_assoc_with_vec = get_edge_embedding(conn, correlated_predicate)
r_targets_vec = get_edge_embedding(conn, drug_predicate)

if disease_vec is None or r_assoc_with_vec is None or r_targets_vec is None:
    print("Error: Could not fetch all required embeddings for the rule.")
    conn.close()


# 3. Apply the rule in reverse using RotatE logic
# The inverse of a rotation (relation) is its complex conjugate

# Let's literally just apply "treats" in reverse.
r_treats = get_edge_embedding(conn, "biolink:treats")
r_treats_inverse = np.conjugate(r_treats)
predicted_drug_vec = disease_vec * r_treats_inverse

# # Step 1: Go from Disease to Gene
# # Gene ≈ Disease o inverse(r_associated_with)
# r_assoc_with_inverse = np.conjugate(r_assoc_with_vec)
# predicted_gene_vec = disease_vec * r_assoc_with_inverse # Element-wise product

# # Step 2: Go from Gene to Drug
# # Drug ≈ Gene o inverse(r_targets)
# r_targets_inverse = np.conjugate(r_targets_vec)
# predicted_drug_vec = predicted_gene_vec * r_targets_inverse

print("Calculated predicted drug vector.")

# 4. Find the closest actual drug embeddings to our predicted vector
closest_drugs = find_closest_nodes_pgvector(
    conn, 
    predicted_drug_vec,
    k=K
)

print("\n--- Validation Results ---")
print(f"Top {K} candidate drugs for '{start_disease_curie}' based on rule:")
for curie, distance in closest_drugs:
    print(f"  - {curie} ({get_normalized_curie(curie)[1]}) (Distance: {distance:.4f})")



# %%
# Find the intersection
k_vals = []
recalls = []
for kk in range(1, K+1):
    predicted_list = [curie for curie, distance in closest_drugs[:kk]]
    predicted_set = set(predicted_list)
    hits = list(predicted_set.intersection(ground_truth_curies))

    # Recall@k = (Number of Hits) / (Total Number of Ground Truth Items)
    k_vals.append(kk)
    recalls.append(len(hits))

# print(recalls)
fig = plt.figure()
plt.plot(k_vals, recalls)
plt.xlabel("Top K predictions tested")
plt.ylabel(f"Recall @ K (# Ground Truths Found of {len(ground_truth_curies)})")
plt.xticks(list(range(K+1))); plt.yticks(list(range(max(recalls)+1)))


# %%
####################################
### TEST FORWARD ROTATIONS       ###
####################################
# Ok, let's test a forward treats edge
dcdb_row = df_dc.sample(1)
start_drug_umls = dcdb_row['drug_umls'][0]
target_disease_umls = dcdb_row['disease_umls'][0]
start_drug_curie, start_drug_label = get_normalized_curie(start_drug_umls)
target_disease_curie, target_disease_label = get_normalized_curie(target_disease_umls)
print("\nStarting from:", start_drug_umls)
print("Normalized Curie for start drug:", start_drug_curie, "|", start_drug_label)
print("\nThis drug targets:", target_disease_curie, "|", target_disease_label)


# %%
start_drug_emb = get_node_embedding(conn, start_drug_curie)
predicted_disease_vec = start_drug_emb * r_treats
print("Calculated predicted disease vector.")

closest_diseases = find_closest_nodes_pgvector(
    conn,
    predicted_disease_vec,
    k=K
)

print("\n--- Validation Results ---")
print(f"Top {K} candidate diseases for '{start_drug_curie}' based on rule:")
for curie, distance in closest_diseases:
    if curie == target_disease_curie:
        print("HIT!")
    print(f"  - {curie} ({get_normalized_curie(curie)[1]}) (Distance: {distance:.4f})")


# %%
