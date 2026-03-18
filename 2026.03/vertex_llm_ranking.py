import os
import time
import uuid
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Vertex AI & GCS
from google.cloud import aiplatform, storage

from query_compression import convert_results_to_sentences

# Load env vars
load_dotenv("/home/npersson/.env")

#####################################################
# GLOBAL CONFIG
#####################################################

PROJECT_ID = "gen-lang-client-0636424847"
LOCATION = "global"
STAGING_BUCKET = "gemini-ranking-outputs-global" 

aiplatform.init(project=PROJECT_ID, location=LOCATION)
storage_client = storage.Client(project=PROJECT_ID)

CONTEXT = """
You are an expert biomedical researcher and graph data scientist working on the NIH Translator project. Your task is to evaluate "Pathfinder" knowledge graph paths based on their clinical utility and causal logic.

You will be given a list of paths connecting two biomedical concepts expressed as human-readable sentences. The paths are nodes connected by predicates from the Biolink model. When multiple predicates are available between two nodes, they will be enclosed in square brackets.

Your goal is to classify each path into one of four tiers and provide an explanation for why you assigned it to that tier. You should focus on how the path traverses the nodes (e.g. which predicates are used and in what order) as much as the content of the nodes themselves. When multiple predicates are available, make your selection based on the most specific available predicate for each hop.

### The Scoring Rubric

**Tier 1: Direct, Causal Mechanistic steps with high clinical or research significance**
* **Criteria:** The path represents a path through direct, causal mechanistic steps with high clinical or research relevance. It makes hops that are mechanistic in nature (vs through similarity or correlation) between nodes that are fundamental to disease processes. In general, Tier 1 should be pretty selective and only represent a small fraction of a given dataset.
* **Examples:**
<Drug> inhibits <gene> which participates in <cellular mechanism> which contributes to <disease>
<ChemicalEntity> [biolink:increases_abundance_of] <Protein> [biolink:regulates] <Gene> [biolink:contributes_to] <Disease>
<ChemicalEntity> [biolink:interacts_with] <Gene> [biolink:biologically_interacts_with] <Gene> [biolink:associated_with] <Disease>
<ChemicalEntity> [biolink:prevents] <PhenotypicFeature> [biolink:associated_with] <BiologicalProcess> [biolink:related_to] <Disease>
* **Counter-Examples:**
These counter-examples apply to both Tier 1 and Tier 2. Paths matching these counter-examples should be placed in Tier 3 or 4.
Do not include paths just because one of the nodes is interesting if the other is unrelated. For example, <Drug> studied to treat <common disease> associated with <Gene> related to <Disease>. The common disease is not an explanatory or mechanistic hop.
If it seems like a better path might exist using the node of interest, assume that that path will be found and placed in Tier 1. Not all results are present in each batch being scored.
Do not include paths that make large hops in different directions. For example, <Drug> affects <Gene> which affected by <Other Drug> which treats <Disease>. Hopping back to a drug from a gene indicates lack of direction. Up-rank paths that take linear, mechanistic paths from start to end.
The caveat to this criteria is when finding paths between nodes of similar type, e.g. Disease to Disease. In this case, the path may need to go "in reverse" to find linking nodes (e.g. a gene-gene or symptomatic pair) before going back up to the ending Disease node.
Do not include "negative" results, unless they appear to be highlighting a previously unknown or unexplored interaction. For example, <Drug> causes <symptom / side effect> which adverse event of <other drug> which treats <Disease> is not helpful.
Do not include similarity hops, e.g. <Drug> similar to <other drug> targets <gene> affects <Disease>. These are not mechanistic.

This counter-example applies only to Tier 1.
Many paths include two genes. These should only be included in Tier 1 if the interaction between the genes captures the most fundamental aspects of the disease process, as opposed to smaller sub-mechanisms, such as those between sibbling genes.

**Tier 2: Direct, Causal Mechanism with lower clinical or research significance**
* **Criteria:** Nearly the same criteria as Tier 1, but this tier captures mechanistic paths that are of secondary importance, lower relevance or lower clinical significance than those in Tier 1.
* **Examples:** The same examples from Tier 1 apply to Tier 2 - the distinction is more of a value judgment.
* **Counter-Examples:** Very similar to Tier 1 counter-examples. However, there is increased tolerance in Tier 2 for more granular mechanistic hops, less-specific predicates, and less significant pathways.

**Tier 3: Low-insight / low significance, generic or off-topic hops, indirect paths, negative results, similarity / correlative hops**
* **Criteria:** The path hops mostly through nodes of high similarity, high-degree nodes (e.g. cancer), through vague predicates, or follow meandering paths that are otherwise not mechanistic, not causal, off-topic, or of little clinical significance. In general, it is expected that the plurality or even majority of results will be Tier 3.
* **Examples:** <ChemicalEntity> [biolink:chemically_similar_to] <ChemicalEntity> [biolink:treats] <Disease> [biolink:subclass_of] <Disease>
<Drug> studied to treat <other Disease> associated with <major organ system> associated with <Target Disease>
Counter-examples from Tiers 1 and 2 are also positive examples for Tier 3.

**Tier 4: Trivial / Artifacts**
* **Criteria:** The path contains loops, tautologies, or data artifacts.
* **Example:** <ChemicalEntity> [biolink:treats] <Disease> [biolink:related_to] <Disease> [biolink:has_phenotype] <PhenotypicFeature> (Where the 2nd and 3rd node are the same concept).
<ChemicalEntity> [biolink:treats] <Disease> [biolink:subclass_of] <Disease> [biolink:super_class_of] <Disease> (A redundant loop up and down the hierarchy).

### Output Format

Return a JSON object containing a list of all input results by their original index. Do not repeat the input path text.
Format:
{
  "results": [
    {
      "index": <integer_id>,
      "tier": <integer_1_to_4>,
      "explanation": "<concise_reasoning>"
    }
  ]
}
"""
print("Context:", "\n--------\n", CONTEXT, "\n--------\n")

RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "tier": {"type": "integer"},
                    "explanation": {"type": "string"}
                },
                "required": ["index", "tier", "explanation"]
            }
        }
    }
}

#####################################################
# GCS HELPERS
#####################################################

def upload_to_gcs(local_path, gcs_uri):
    bucket_name = gcs_uri.split('/')[2]
    blob_path = '/'.join(gcs_uri.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

def delete_gcs_prefix(gcs_uri):
    bucket_name = gcs_uri.split('/')[2]
    prefix = '/'.join(gcs_uri.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        blob.delete()

def download_and_parse_results(gcs_output_directory):
    """
    Scans the output directory and handles the nested Gemini JSON structure.
    """
    bucket_name = gcs_output_directory.split('/')[2]
    prefix = '/'.join(gcs_output_directory.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    all_flat_results = []
    
    for blob in blobs:
        # Vertex names output files usually as 'prediction.results-00000-of-00001'
        if "predictions" in blob.name:
            print(f"Processing: {blob.name}")
            content = blob.download_as_text()
            
            for line in content.splitlines():
                if not line.strip(): continue
                
                record = json.loads(line)
                
                # NAVIGATION: Vertex Batch Gemini Output structure
                # Typically: record['response']['candidates'][0]['content']['parts'][0]['text']
                # OR: record['prediction'][...depending on SDK version]
                try:
                    # 1. Try to find the raw text response
                    response_obj = record.get('response') or record.get('prediction')
                    if not response_obj:
                        continue
                        
                    raw_text = response_obj['candidates'][0]['content']['parts'][0]['text']
                    
                    # 2. Parse the inner JSON string generated by the model
                    parsed_json = json.loads(raw_text)
                    
                    if "results" in parsed_json:
                        all_flat_results.extend(parsed_json["results"])
                    else:
                        # If the model returned a single object instead of a list
                        all_flat_results.append(parsed_json)
                        
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"Skipping line due to error: {e}")
                    # Optional: print(line) to see the exact structure
                    
    return all_flat_results

#####################################################
# BATCH GENERATION
#####################################################

def generate_request_entry(batch_df):
    batch_json = batch_df.to_json(orient='records')
    prompt = f"{CONTEXT}\n\nINPUT DATA:\n{batch_json}"
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generation_config": {
                "response_mime_type": "application/json",
                "response_schema": RANKING_SCHEMA
            }
        }
    }

def run_ranking(pth, output_dir, raw_results_dir, output_file_name, **config):
    # 1. RAW DATA PREP
    print(f"Loading data from: {pth}")
    df = pd.read_csv(pth, sep="\t")
    
    if config['DOWNSAMPLE_FACTOR'] == "auto":
        total_results = len(df)
        print(f"Dataset contains {total_results} results. Targeting <20,000")
        downsample = total_results // 20000 + 1 if total_results > 25000 else 1
    else:
        downsample = config['DOWNSAMPLE_FACTOR']

    if 'index' not in df.columns:
        df['index'] = df.index

    if "query_sentence" not in df.columns:
        print("Generating query sentences...")
        df = convert_results_to_sentences(df)

    # 2. BATCH PREP
    batch_entries = []
    for run_i in range(config['NUM_RUNS']):
        df_shuffled = df.sample(frac=1, random_state=(config['RANDOM_SEED'] + run_i)).reset_index(drop=True)
        stop_row = len(df_shuffled) // downsample
        simple_chunks = [df_shuffled.iloc[i:i + config['BATCH_SIZE']] for i in range(0, stop_row, config['BATCH_SIZE'])]
        
        for chunk in simple_chunks:
            chunk_for_job = chunk[['index', 'query_sentence']]
            batch_entries.append(generate_request_entry(chunk_for_job))

    # 3. TOKEN ESTIMATION & PROMPT
    tokens_in_sample = len(batch_entries[0]['request']['contents'][0]['parts'][0]['text']) // 4
    total_est_tokens = len(batch_entries) * tokens_in_sample
    print(f"Generated {len(batch_entries)} batches (~{total_est_tokens} total tokens).")

    user_choice = input(f"Submit to Vertex AI? (y/n): ")
    if user_choice.lower().strip() != "y":
        raise KeyboardInterrupt("Cancelled by user.")

    # 4. UPLOAD & SUBMIT
    job_uuid = str(uuid.uuid4())
    local_input_dir = Path("gemini_inputs")
    local_input_dir.mkdir(parents=True, exist_ok=True)
    local_input = local_input_dir / f"vertex_input_{job_uuid}.jsonl"
    gcs_input = f"gs://{STAGING_BUCKET}/inputs/{job_uuid}.jsonl"
    gcs_out_prefix = f"gs://{STAGING_BUCKET}/outputs/{job_uuid}/"

    with open(local_input, "w") as f:
        for entry in batch_entries:
            f.write(json.dumps(entry) + "\n")
    
    upload_to_gcs(local_input, gcs_input)
    
    print("Submitting Vertex AI Batch Job...")
    batch_job = aiplatform.BatchPredictionJob.create(
        job_display_name=f"pathfinder_{datetime.now().strftime('%Y%m%d_%H%M')}",
        model_name=f"publishers/google/models/{config['MODEL_NAME']}",
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=[gcs_input],
        gcs_destination_prefix=gcs_out_prefix,
    )
    
    batch_job.wait()

    # 5. RETRIEVE & MERGE
    if batch_job.state.name == "JOB_STATE_SUCCEEDED":
        gcs_actual_output = batch_job.output_info.gcs_output_directory
        all_flat_results = download_and_parse_results(gcs_actual_output)
        if all_flat_results:
            results_df = pd.DataFrame(all_flat_results)
            results_df['index'] = results_df['index'].astype(int)
            final_df = results_df.merge(df, on='index', how='left')
            
            output_path = output_dir / output_file_name
            final_df.to_parquet(output_path, index=False)
            print(f"Success! Saved to {output_path}")
            print("Example output:", final_df.sample(1))

        # 6. CLEANUP
        # delete_gcs_prefix(gcs_input)
        # delete_gcs_prefix(gcs_actual_output)
        # if os.path.exists(local_input): os.remove(local_input)
        # print("GCS and local temporary files cleaned.")
    else:
        print(f"Job failed: {batch_job.error}")

if __name__ == "__main__":
    config = {
        "MODEL_NAME": "gemini-3.1-flash-lite-preview",
        "NUM_RUNS": 1,
        "BATCH_SIZE": 500,
        "DOWNSAMPLE_FACTOR": "auto",
        "RANDOM_SEED": 42
    }

    # Load Data
    TEST_SET_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets")
    # QUERIES_DIR = TEST_SET_DIR / "training_queries"
    # test_csvs = [
    #     # "CHEBI_45783_to_MONDO_0004979_3_hop_w_direction_paths.csv", # imatinib -> asthma
    #     # "asthma_results_subset.csv", # imatinib -> asthma with all passing test nodes
    #     # "CHEBI_13719_to_MONDO_0005575_3_hop_w_direction_paths.csv", # acetylsalicylate -> colorectal cancer - not enough paths
    #     # "CHEBI_5118_to_MONDO_0100233_3_hop_w_direction_paths.csv", # Fluoxetine -> Long COVID
    #     # "CHEBI_7465_to_MONDO_0008218_3_hop_w_direction_paths.csv", # Naltrexone -> Hailey-Hailey disease
    #     "CHEBI_9139_to_MONDO_0004975_3_hop_w_direction_paths.csv", # Sildenafil -> Alzheimer's (needs 10:1 downsampling)
    # ]
    QUERIES_DIR = TEST_SET_DIR / "gandalf_responses_predicates_with_inverse"
    test_csvs = [
        "MONDO_0019632_to_MONDO_0005340_3_hop_w_direction_paths.tsv", # Lyme Disease -> alopecia
        "MONDO_0005011_to_MONDO_0005180_3_hop_w_direction_paths.tsv", # Crohns -> Parksinsons
    ]

    OUTPUT_DIR = QUERIES_DIR / "ranking_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RAW_RESULTS_DIR = TEST_SET_DIR / "raw_gemini_batch_outputs"
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    for csv in test_csvs:
        pth = QUERIES_DIR / csv

        # Construct output filename
        today = datetime.today().strftime("%Y-%m-%d")
        output_file_name = (
            f"{today}-gemini-labels-"
            f"{pth.stem}-"
            f"{config['MODEL_NAME']}-"
            f"prompt-v10-"
            f"nr-{config['NUM_RUNS']}-"
            f"bs-{config['BATCH_SIZE']}-"
            f"df-{config['DOWNSAMPLE_FACTOR']}-"
            # f"sb-{config['STRATIFIED_BATCHES_PER_RUN']}-"
            # f"gb-{config['GROUP_BY_COL']}-"
            f"rs-{config['RANDOM_SEED']}"
            ".parquet"
        )

        print("Output file:\n  ", output_file_name)

        run_ranking(
            pth,
            OUTPUT_DIR,
            RAW_RESULTS_DIR,
            output_file_name,
            **config
            )
        