import requests
import json
from typing import List, Dict, Optional, Any

# Define the constant API endpoint
API_URL = "https://automat.renci.org/robokopkg/cypher"

# Standard headers for this API
HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

def find_drugs_for_disease_id(disease_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Queries the Robokop API to find drugs that treat a specific disease.

    Args:
        disease_id: The MONDO identifier for the disease 
                      (e.g., "MONDO:0004979" for Alzheimer's).

    Returns:
        A list of dictionaries, where each dictionary contains a drug's
        name, ID, and description.
        Returns None if an API or network error occurs.
    """
    
    # 1. Construct the specific Cypher query string
    # We embed the disease_id directly into the query.
    query_string = (
        f"MATCH (a:`biolink:Drug`)-[r]-(b {{id: '{disease_id}'}}) "
        f"WHERE type(r) = 'biolink:treats' RETURN a"
    )

    # 2. Build the JSON payload for the POST request
    payload = {
        "query": query_string
    }

    print(f"Querying Robokop for disease: {disease_id}...")

    try:
        # 3. Make the POST request
        response = requests.post(
            API_URL, 
            headers=HEADERS, 
            json=payload,
            timeout=20
        )

        # Raise an exception for bad HTTP status codes (like 404, 500)
        response.raise_for_status()

        # 4. Parse the JSON response
        data = response.json()

        # 5. Check for API-level errors
        # The API response includes an "errors" key
        if data.get("errors") and len(data["errors"]) > 0:
            print(f"API returned an error: {data['errors']}")
            return None

        # 6. Parse the complex "results" structure
        # --- THIS IS THE UPDATED PARSING LOGIC ---
        results = data.get("results")
        if not results:
            print("No 'results' key found in API response. Returning empty list.")
            return []

        cleaned_drugs = []
        for item in results:  # `item` is {"columns": [...], "data": [...]}
            # Check if 'data' key exists and is a list
            if not isinstance(item.get("data"), list):
                continue
            
            for data_item in item["data"]: # `data_item` is {"row": [...], "meta": []}
                # Check if 'row' exists and has at least one element
                if isinstance(data_item.get("row"), list) and len(data_item["row"]) > 0:
                    # The actual drug data is the first element of the 'row' list
                    drug_data = data_item["row"][0]
                    
                    # Extract the useful fields into a simpler dictionary
                    # Switched to 'description' from 'mrdef' based on your snippet
                    cleaned_drugs.append({
                        "name": drug_data.get("name"),
                        "id": drug_data.get("id"),
                        "description": drug_data.get("description") 
                    })
        # --- END OF UPDATED LOGIC ---

        print(f"Found {len(cleaned_drugs)} treating drugs.")
        return cleaned_drugs

    # Handle various exceptions that can occur
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Response Body: {e.response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}")
    except requests.exceptions.Timeout as e:
        print(f"Request Timed Out: {e}")
    except requests.exceptions.JSONDecodeError:
        print(f"Failed to decode JSON response. Raw text: {response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Return None on any failure
    return None

# --- Example Usage ---
if __name__ == "__main__":
    # MONDO:0004979 is Alzheimer's disease
    alzheimer_id = "MONDO:0004979"
    drugs = find_drugs_for_disease_id(alzheimer_id)

    if drugs:
        print(f"\n--- Drugs found for {alzheimer_id} ---")
        for i, drug in enumerate(drugs):
            print(f"\n{i+1}. {drug['name']} ({drug['id']})")
            # Print a snippet of the description
            desc_snippet = (drug['description'] or "No description.")[:120]
            print(f"   Desc: {desc_snippet}...")
    
    print("\n--- Testing a different disease ---")
    # MONDO:0005149 is Type 2 Diabetes
    diabetes_id = "MONDO:0005149"
    diabetes_drugs = find_drugs_for_disease_id(diabetes_id)
    
    if diabetes_drugs:
        print(f"\n--- Drugs found for {diabetes_id} ---")
        for drug in diabetes_drugs[:5]: # Print first 5
             print(f"- {drug['name']} ({drug['id']})")