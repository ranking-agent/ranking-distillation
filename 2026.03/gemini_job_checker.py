# %%
import os
from google import genai

# Ensure GEMINI_API_KEY is in your environment or .env file
api_key = os.environ.get('GEMINI_API_KEY')    
client = genai.Client(api_key=api_key)

batch_jobs = client.batches.list()


# %%
# Optional query config:
# batch_jobs = client.batches.list(config={'page_size': 5})

for batch_job in batch_jobs:
    if batch_job.state == genai.types.JobState.JOB_STATE_PENDING:
        print(batch_job)
        client.batches.cancel(name=batch_job.name)
        print(f"\nCancelled {batch_job.name}")
    else:
        print(f"\n{batch_job.name} not Pending??")
        print(f"{batch_job.state}")


# %%
