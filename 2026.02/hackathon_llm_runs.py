import os
import time
import uuid
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

from query_compression import convert_results_to_sentences

# Load env vars
load_dotenv("/home/npersson/.env")

#####################################################
# GLOBAL JOB CONFIG
#####################################################

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


# Define the Schema for Structured Output
RANKING_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "tier": {"type": "INTEGER"},
            "explanation": {"type": "STRING"}
        },
        "required": ["index", "tier", "explanation"]
    }
}

#####################################################
# FUNCTIONS
#####################################################

def generate_request_entry(batch_df, batch_id, run_id, strategy):
    """
    Creates a single JSONL entry for the Batch API using Structured Output.
    """
    # Convert batch to JSON string for the prompt
    batch_json = batch_df.to_json(orient='records')
    prompt = f"{CONTEXT}\n\nINPUT DATA:\n{batch_json}"

    # Encode metadata into the key
    key = f"{run_id}|{strategy}|{batch_id}"

    entry = {
        # CRITICAL FIX: The API requires 'custom_id', not 'key'
        "custom_id": key, 
        "request": {
            "contents": [{
                "parts": [{"text": prompt}],
                "role": "user"
            }],
            "generation_config": {
                "response_mime_type": "application/json",
                "response_schema": RANKING_SCHEMA
            }
        }
    }
    return entry


def run_ranking(
    pth,
    output_dir,
    raw_results_dir,
    output_file_name=None,
    MODEL_NAME="gemini-3-flash-preview",
    NUM_RUNS=1,
    BATCH_SIZE=10000,
    DOWNSAMPLE_FACTOR=1, # Total batches will be divided by this number before submission. Set to 1 for all.
    STRATIFIED_BATCHES_PER_RUN=0,
    GROUP_BY_COL="categories",
    JSONL_FILENAME="batch_requests.jsonl",
    SLEEP_INTERVAL=15, # Frequency (in seconds) to poll for job status and print
    RANDOM_SEED=42,
    ):
    """
    Submit a job to the Gemini API to rank the query results
    from pth using the provided settings
    """

    ########################
    # CLIENT SETUP
    ########################

    # Ensure GEMINI_API_KEY is in your environment or .env file
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return
        
    client = genai.Client(api_key=api_key)
    
    ########################
    # RAW DATA PREP
    ########################

    print(f"Loading data from: {pth}")
    df = pd.read_csv(pth, sep="\t")
    if DOWNSAMPLE_FACTOR == "auto":
        total_results = len(df)
        print(f"Dataset contains {total_results} results. Targeting <20,000")
        if total_results > 25000:
            DOWNSAMPLE_FACTOR = total_results // 20000 + 1 # Never yield more than 20k rows for batch jobs
        else:
            DOWNSAMPLE_FACTOR = 1
        print(f"Using downsample factor of {DOWNSAMPLE_FACTOR}")

    
    if 'index' not in df.columns:
        df['index'] = df.index

    # 3. Preparation
    has_category = GROUP_BY_COL in df.columns
    if not has_category:
        print(f"WARNING: Column '{GROUP_BY_COL}' not found. Stratified sampling will be skipped.")

    has_sentences = "query_sentence" in df.columns
    if not has_sentences:
        print("Generating query sentences...")
        df = convert_results_to_sentences(df)

    ########################
    # BATCH PREP
    ########################

    batch_entries = []
    print(f"Preparing batch requests for {NUM_RUNS} runs...")

    for run_i in range(NUM_RUNS):

        # A. STRATIFIED
        if has_category:
            for b in range(STRATIFIED_BATCHES_PER_RUN):
                try:
                    stratified_sample = df.groupby(GROUP_BY_COL).sample(n=1)
                    stratified_sample_for_job = stratified_sample[['index', 'query_sentence']]
                    batch_id = str(uuid.uuid4())
                    entry = generate_request_entry(stratified_sample_for_job, batch_id, run_i, "stratified")
                    batch_entries.append(entry)
                except ValueError as e:
                    print(f"Skipping stratified batch (possibly not enough data): {e}")
        
        # B. SIMPLE SHUFFLE AND CHUNKING
        df_shuffled = df.sample(frac=1, random_state=(RANDOM_SEED + run_i)).reset_index(drop=True)
        total_rows = len(df_shuffled)
        stop_row = total_rows // DOWNSAMPLE_FACTOR

        simple_chunks = [df_shuffled.iloc[i:i + BATCH_SIZE] for i in range(0, stop_row, BATCH_SIZE)]
        print(f"Preparing {len(simple_chunks)} chunks...")
        
        for chunk in simple_chunks:
            batch_id = str(uuid.uuid4())
            chunk_for_job = chunk[['index', 'query_sentence']]
            entry = generate_request_entry(chunk_for_job, batch_id, run_i, "simple_shuffle")
            batch_entries.append(entry)

    tokens_in_sample_entry = len(batch_entries[0]['request']['contents'][0]['parts'][0]['text']) // 4
    print(f"Generated {len(batch_entries)} total batch requests.")
    print(f"About {tokens_in_sample_entry} tokens in the first entry.")
    print(f"In total about {len(batch_entries) * tokens_in_sample_entry} tokens.")

    ### PROMPT USER BEFORE SUBMITTING ###
    user_choice = input(
        f"\nAre you sure you want to submit "
        f"{len(batch_entries)} batches)? (y/n): "
    )
    if user_choice.lower().strip() == "y":
        pass
    elif user_choice.lower().strip() != "y":
        raise KeyboardInterrupt("Decided not to run")

    # 5. Write to JSONL
    print(f"Writing requests to {JSONL_FILENAME}...")
    with open(JSONL_FILENAME, "w") as f:
        for entry in batch_entries:
            f.write(json.dumps(entry) + "\n")

    # 6. Upload File
    print("Uploading file to Gemini File API...")
    batch_file = client.files.upload(
        file=JSONL_FILENAME,
        config={'display_name': f'pathfinder_batch_{int(time.time())}', 'mime_type': 'application/json'}
    )
    print(f"File uploaded: {batch_file.name}")

    # 7. Submit Batch Job
    print("Submitting Batch Job...")
    batch_job = client.batches.create(
        model=MODEL_NAME,
        src=batch_file.name,
        config={
            'display_name': f"pathfinder_run_{datetime.now().strftime('%Y%m%d_%H%M')}",
        }
    )
    job_name = batch_job.name
    print(f"Batch Job Created: {job_name}")
    print("Polling for completion...")

    # 8. Poll for Completion
    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])
    
    while True:
        try:
            batch_job = client.batches.get(name=job_name)
            current_state = batch_job.state.name
            print(f"Current state: {current_state}")
            
            if current_state in completed_states:
                break
            
            time.sleep(SLEEP_INTERVAL) 
        except Exception as e:
            print(f"Error polling job status: {e}. Retrying in {SLEEP_INTERVAL}s...")
            time.sleep(SLEEP_INTERVAL)

    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Job failed with state: {batch_job.state.name}")
        if hasattr(batch_job, 'error'):
            print(f"Error details: {batch_job.error}")
        return

    print("Job Completed Successfully! Processing results...")

    # 9. Retrieve Results
    flat_results = []
    
    if batch_job.dest and batch_job.dest.file_name:
        result_file_name = batch_job.dest.file_name
        print(f"Downloading results from: {result_file_name}")
        
        # Download the file content (returns bytes)
        file_content = client.files.download(file=result_file_name)
        
        # MEMORY SAFEGUARD: Write bytes to disk immediately instead of loading string into RAM
        # This prevents OOM errors on large batch results
        raw_results_path = raw_results_dir / f"raw_results_{job_name.split('/')[-1]}.jsonl"
        print(f"Writing raw results to: {raw_results_path}")
        
        with open(raw_results_path, "wb") as f:
            f.write(file_content)
            
        # Clear the heavy bytes object from memory
        del file_content
        
        # Process the file line-by-line from disk
        print("Parsing results file line-by-line...")
        with open(raw_results_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if not line.strip(): continue
                
                try:
                    res_obj = json.loads(line)
                    
                    # The 'custom_id' connects this result to your specific batch request
                    key = res_obj.get("custom_id") 
                    
                    if not key:
                        # Debugging help if key is missing
                        print(f"WARNING: 'custom_id' missing in result line {line_num}. Keys found: {list(res_obj.keys())}")
                        # Try to parse 'key' if accidentally used in older runs
                        key = res_obj.get("key", "")
                        if not key:
                            continue

                    # Metadata extraction
                    try:
                        run_id, strategy, batch_id = key.split("|")
                        run_id = int(run_id)
                    except ValueError:
                        print(f"Skipping malformed key structure: {key}")
                        continue

                    response_wrapper = res_obj.get("response", {})
                    
                    if "error" in response_wrapper:
                        print(f"Error in batch item {key}: {response_wrapper['error']}")
                        continue
                    
                    # Extract text
                    try:
                        # Access path: response -> candidates[0] -> content -> parts[0] -> text
                        candidates = response_wrapper.get('candidates', [])
                        if not candidates:
                            print(f"No candidates returned for {key}")
                            continue
                            
                        model_text = candidates[0]['content']['parts'][0]['text']
                        
                        # Because we used Structured Output, model_text is valid JSON
                        parsed_json = json.loads(model_text)
                        
                        if isinstance(parsed_json, list):
                            for item in parsed_json:
                                item['job_id'] = batch_id
                                item['run_id'] = run_id
                                item['sampling_strategy'] = strategy
                                item['model_version'] = MODEL_NAME
                                flat_results.append(item)
                        else:
                            print(f"Unexpected JSON structure for {key}: Expected list, got {type(parsed_json)}")
                            
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        print(f"Failed to parse model output for {key}: {e}")
                        # Print snippet for debugging
                        print(f"Snippet: {str(response_wrapper)[:200]}")
                        
                except json.JSONDecodeError:
                    print(f"Failed to decode JSONL line {line_num}")
    else:
        print("Job succeeded but no output file found in destination.")
        return

    print(f"Processing complete. Received {len(flat_results)} ranked items.")

    # 10. Merge & Save
    if flat_results:
        results_df = pd.DataFrame(flat_results)
        
        if 'index' in results_df.columns:
            results_df['index'] = results_df['index'].astype(int)
            
            # Ensure index types match for merge
            if df['index'].dtype != results_df['index'].dtype:
                df['index'] = df['index'].astype(int)

            final_df = results_df.merge(df, on='index', how='left')
            
            ########################
            # SAVE OUTPUTS
            ########################

            if output_file_name is None:
                original_name = pth.stem
                output_file_name = f"{original_name}_ranked_batch_{job_name.split('/')[-1]}.parquet"
            output_path = output_dir / output_file_name
            
            # Fallback to csv if fastparquet/pyarrow not installed, but try parquet first
            try:
                final_df.to_parquet(output_path, index=False)
                print(f"Saved merged results to {output_path}")
            except ImportError:
                csv_path = str(output_path).replace('.parquet', '.csv')
                final_df.to_csv(csv_path, index=False)
                print(f"Saved merged results to {csv_path} (parquet library missing)")
                
        else:
            print("Error: 'index' column missing from API response.")
    else:
        print("No valid results parsed.")


if __name__ == "__main__":
    
    start_time = time.perf_counter()
    # LLM Job Config
    config = {
        "MODEL_NAME": "gemini-3-flash-preview",
        "NUM_RUNS": 1,
        "BATCH_SIZE": 500,
        "DOWNSAMPLE_FACTOR": "auto",
        "STRATIFIED_BATCHES_PER_RUN": 0,
        "GROUP_BY_COL": "categories",
        "JSONL_FILENAME": "batch_requests.jsonl",
        "SLEEP_INTERVAL": 30,
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
        # "MONDO_0005011_to_MONDO_0005180_3_hop_w_direction_paths.tsv", # Crohns -> Parksinsons
        "MONDO_0019632_to_MONDO_0005340_3_hop_w_direction_paths.tsv", # Lyme Disease -> alopecia
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
            f"prompt-v9-"
            f"nr-{config['NUM_RUNS']}-"
            f"bs-{config['BATCH_SIZE']}-"
            f"df-{config['DOWNSAMPLE_FACTOR']}-"
            f"sb-{config['STRATIFIED_BATCHES_PER_RUN']}-"
            f"gb-{config['GROUP_BY_COL']}-"
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
        
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds")