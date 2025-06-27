import requests, json, pandas as pd

url = "https://www.cdc.gov/wcms/vizdata/NCEZID_DIDRI/NWSSRegionalLevel.json"

resp = requests.get(url, timeout=10)
resp.raise_for_status()                            # good habit

text = resp.content.decode("utf-8-sig")            # ← removes BOM
rows = json.loads(text)                            # now parses fine

df = pd.DataFrame(rows)
df[["Week_Ending_Date", "National_WVAL"]].to_csv("national_wastewater_wval.tsv", index=False, sep='\t')
