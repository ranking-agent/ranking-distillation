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

You will be given a list of paths connecting two biomedical concepts. Your goal is to classify each path into one of four tiers.

### The Scoring Rubric

**Tier 1: High-Value Causal Mechanism**
* **Criteria:** The path represents a plausible, direct mechanism of action with high clinical or research relevance. It uses specific predicates (e.g., `inhibits`, `regulates`, `has phenotype`) rather than vague associations.
* **Example:** Imatinib inhibits KIT which participates in Mast Cell Activation which contributes to Asthma

**Tier 2: Plausible but Indirect/Loose**
* **Criteria:** The logic is sound, but the path might be an indirect variant of a primary mechanism, or the predicates are slightly weaker (e.g., `associated with` in a crucial step), or the relevance is secondary.

**Tier 3: Low-Insight / Generic Hops**
* **Criteria:** The path hops mostly through nodes of high similarity, high-degree nodes (e.g. cancer) or through vague predicates (e.g. `associated with`)
* **Example:** Imatinib chemically similar to masitinib which related to asthma

**Tier 4: Trivial / Artifacts**
* **Criteria:** The path contains loops, tautologies, or data artifacts.
* **Example:** Disease X related to Disease X.

### Output Format

Return a JSON object containing a list of all input results by their original index. Do not repeat the input path text.
Format:
{
  "results": [
    {
      "index": <integer_id>,
      "tier": <integer_1_to_4>,
      "explanation": "<brief_concise_reasoning>"
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
    MODEL_NAME="gemini-2.5-pro",
    NUM_RUNS=1,
    BATCH_SIZE=10000,
    DOWNSAMPLE_FACTOR=1, # Total batches will be divided by this number before submission. Set to 1 for all.
    STRATIFIED_BATCHES_PER_RUN=0,
    GROUP_BY_COL="categories",
    JSONL_FILENAME="batch_requests.jsonl",
    SLEEP_INTERVAL=60, # Frequency (in seconds) to poll for job status and print
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
    
    if 'index' not in df.columns:
        df['index'] = df.index

    # 3. Preparation
    has_category = GROUP_BY_COL in df.columns
    if not has_category:
        print(f"WARNING: Column '{GROUP_BY_COL}' not found. Stratified sampling will be skipped.")

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
                    stratified_sample_sentences = convert_results_to_sentences(stratified_sample)
                    stratified_sample_sentences_for_job = stratified_sample_sentences[['index', 'query_sentence']]
                    batch_id = str(uuid.uuid4())
                    entry = generate_request_entry(stratified_sample_sentences_for_job, batch_id, run_i, "stratified")
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
            chunk_with_sentences = convert_results_to_sentences(chunk) # Turns query result into human-readable sentence
            chunk_for_job = chunk_with_sentences[['index', 'query_sentence']]
            entry = generate_request_entry(chunk_for_job, batch_id, run_i, "simple_shuffle")
            batch_entries.append(entry)

    tokens_in_sample_entry = len(batch_entries[0]['request']['contents'][0]['parts'][0]['text']) // 4
    print(f"Generated {len(batch_entries)} total batch requests.")
    print(f"About {tokens_in_sample_entry} tokens in the first entry.")

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
    
    # LLM Job Config
    config = {
        "MODEL_NAME": "gemini-3-flash-preview",
        "NUM_RUNS": 1,
        "BATCH_SIZE": 100,
        "DOWNSAMPLE_FACTOR": 10,
        "STRATIFIED_BATCHES_PER_RUN": 0,
        "GROUP_BY_COL": "categories",
        "JSONL_FILENAME": "batch_requests.jsonl",
        "SLEEP_INTERVAL": 30,
        "RANDOM_SEED": 42
    }

    # Load Data
    TEST_SET_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/")
    # QUERIES_DIR = TEST_SET_DIR / "rare_disease_queries"
    # test_csvs = [
    #     "MONDO_0021020_to_NCBIGene_54658_3_hop_w_direction_paths.csv", # Crigler-Najjar
    #     # "MONDO_0009897_to_NCBIGene_2632_3_hop_w_direction_paths.csv",
    #     # "MONDO_0007931_to_NCBIGene_7439_3_hop_w_direction_paths.csv",
    #     # "MONDO_0009672_to_NCBIGene_6606_3_hop_w_direction_paths.csv",
    #     # "MONDO_0013166_to_NCBIGene_18_3_hop_w_direction_paths.csv",
    #     # "MONDO_0011308_to_NCBIGene_617_3_hop_w_direction_paths.csv",
    #     # "MONDO_0019353_to_NCBIGene_24_3_hop_w_direction_paths.csv",
    #     # "MONDO_0010130_to_NCBIGene_1806_3_hop_w_direction_paths.csv",
    #     # "MONDO_0003947_to_NCBIGene_959_3_hop_w_direction_paths.csv",
    #     # "MONDO_0009653_to_NCBIGene_57192_3_hop_w_direction_paths.csv",
    #     # "MONDO_0008224_to_NCBIGene_6329_3_hop_w_direction_paths.csv",
    #     # "MONDO_0008692_to_NCBIGene_4547_3_hop_w_direction_paths.csv", # 17MB
    # ]
    QUERIES_DIR = TEST_SET_DIR / "generic_queries"
    test_csvs = [
        "CHEBI_45783_to_MONDO_0004979_3_hop_w_direction_paths.csv" # imatinib -> asthma
    ]

    OUTPUT_DIR = TEST_SET_DIR / "ranked_results" / "flash"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RAW_RESULTS_DIR = TEST_SET_DIR / "gemini_batch_outputs"
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    for csv in test_csvs:
        pth = QUERIES_DIR / csv

        # Construct output filename
        today = datetime.today().strftime("%Y-%m-%d")
        output_file_name = (
            f"{today}-gemini-labels-"
            f"{pth.stem}-"
            f"{config['MODEL_NAME']}-"
            f"prompt-v5-"
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