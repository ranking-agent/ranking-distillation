import time

from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions, JobState

PROJECT_ID = "gen-lang-client-0636424847"
LOCATION = "us-central1"
STAGING_BUCKET = "gemini-ranking-outputs-global" 

client = genai.Client(http_options=HttpOptions(api_version="v1"))
# TODO(developer): Update and un-comment below line
output_uri = f"gs://{STAGING_BUCKET}/20260701-genai-test"

# See the documentation: https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches.create
job = client.batches.create(
    # To use a tuned model, set the model param to your tuned model using the following format:
    # model="projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}
    model="gemini-3.5-flash",
    # Source link: https://storage.cloud.google.com/cloud-samples-data/batch/prompt_for_batch_gemini_predict.jsonl
    src="gs://cloud-samples-data/batch/prompt_for_batch_gemini_predict.jsonl",
    config=CreateBatchJobConfig(dest=output_uri),
)
print(f"Job name: {job.name}")
print(f"Job state: {job.state}")
# Example response:
# Job name: projects/.../locations/.../batchPredictionJobs/9876453210000000000
# Job state: JOB_STATE_PENDING

# See the documentation: https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob
completed_states = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
}

while job.state not in completed_states:
    time.sleep(30)
    job = client.batches.get(name=job.name)
    print(f"Job state: {job.state}")
# Example response:
# Job state: JOB_STATE_PENDING
# Job state: JOB_STATE_RUNNING
# Job state: JOB_STATE_RUNNING
# ...
# Job state: JOB_STATE_SUCCEEDED