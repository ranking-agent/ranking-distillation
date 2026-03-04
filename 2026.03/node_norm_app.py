import streamlit as st
import re
import requests
import pandas as pd

st.set_page_config(page_title="CURIE Node Normalizer", layout="wide")

st.title("🧬 CURIE Node Normalizer")

def extract_curies_ordered(text):
    # Using your improved regex
    pattern = r'([A-Z][a-zA-Z0-9]+)[_:]([A-Z0-9]+)'
    matches = re.findall(pattern, text)
    
    seen = set()
    ordered_curies = []
    
    for prefix, identifier in matches:
        curie = f"{prefix}:{identifier}"
        if curie not in seen:
            ordered_curies.append(curie)
            seen.add(curie)
            
    return ordered_curies

user_input = st.text_area("Input Text", placeholder="Paste filenames here...", height=200)

if user_input:
    found_curies = extract_curies_ordered(user_input)
    
    if not found_curies:
        st.warning("No CURIEs found with the pattern Prefix_ID or Prefix:ID.")
    else:
        with st.spinner("Calling TRAPI Node Norm..."):
            url = "https://nodenormalization-sri.renci.org/get_normalized_nodes"
            try:
                response = requests.post(url, json={"curies": found_curies}, timeout=30)
                response.raise_for_status()
                results = response.json()
            except Exception as e:
                st.error(f"API Error: {e}")
                results = {}

            table_data = []
            for curie in found_curies:
                data = results.get(curie)
                if data and 'id' in data:
                    name = data['id'].get('label', "No label found")
                    # Join categories and limit length for the table
                    category = ", ".join(data.get('type', ["Unknown"]))
                else:
                    name = "Not Found"
                    category = "N/A"
                
                table_data.append({
                    "CURIE": curie,
                    "Human-Readable Name": name,
                    "Category": category
                })
            
            df = pd.DataFrame(table_data)

            # --- Column Configuration ---
            # We set specific widths here to prevent squashing.
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "CURIE": st.column_config.TextColumn(
                        "CURIE",
                        width="medium",  # Approx 200px
                        help="The extracted identifier"
                    ),
                    "Human-Readable Name": st.column_config.TextColumn(
                        "Human-Readable Name",
                        width="large",   # Approx 400px
                        help="The preferred name from Node Norm"
                    ),
                    "Category": st.column_config.TextColumn(
                        "Category",
                        width="small",   # Keeps this thin
                    )
                },
                hide_index=True
            )
            
            # Export
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "curie_results.csv")