## Omicron US manuscript dataset

Data preparation followed:

1. Nextstrain-curated metadata TSV of open data in the GenBank database was downloaded. Uncompressing and renaming this file resulted in `open_metadata.tsv` via:
```
curl https://data.nextstrain.org/files/ncov/open/metadata.tsv.gz --output metadata.tsv.gz
gzip -d metadata.tsv.gz -c > open_metadata.tsv
```

2. The metadata file was pruned to only relevant columns via:
```
tsv-select -H -f strain,date,country,division,QC_overall_status,Nextstrain_clade open_metadata.tsv > open_metadata_pruned.tsv
```

3. This `open_metadata_pruned.tsv` is processed in Mathematica by running the notebook `omicron-us-manuscript_data-prep.nb`. This results in the export of files: `omicron-us-manuscript_location-variant-sequence-counts.tsv` and `omicron-us-manuscript_location-case-counts.tsv` versioned here. This subsets between Dec 1, 2021 and June 31, 2022. Samples labeled as `bad` in Nextclade QC are removed. There will be dates that are missing sequence counts or case counts. These should be assumed to be 0.
