## SGTF King County dataset

Data preparation followed:

1. UW Virology data on SGTF counts from https://github.com/proychou/sgtf

2. Download Seattle Flu Study "S positive" counts from https://backoffice.seattleflu.org/metabase/question/822. Download as CSV and rename to `sfs_counts.csv`.

3. King County cases from https://github.com/nytimes/covid-19-data/blob/master/rolling-averages/us-counties-recent.csv

4. Processed in Mathematica by running the notebook `sgtf-king-county_data-prep.nb`. This results in the export of files: `sgtf-king-county_location-variant-sequence-counts.tsv` and `sgtf-king-county_location-case-counts.tsv`.
