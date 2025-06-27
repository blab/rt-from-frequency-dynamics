## Clades dataset

1. Download clade-level counts via:
```
curl -fsSL --compressed https://data.nextstrain.org/files/workflows/forecasts-ncov/gisaid/nextstrain_clades/global.tsv.gz --output clade_sequence_counts.tsv.gz
gzip -d clade_sequence_counts.tsv.gz -c > clade_sequence_counts.tsv
```

2. Download wastewater and process prevalence via:
```
python download_wastewater.py
```
resulting in `national_wastewater_wval.tsv`

2. These files are processed in Mathematica by running the notebook `data-prep.nb`. This results in the export of files: 
- `clades_location-variant-sequence-counts.tsv`
- `clades_location-case-counts.tsv`

These files represent heavily derived GISAID data and are equivalent to downloadable results from [outbreak.info](https://outbreak.info), [cov-spectrum.org](https://cov-spectrum.org) and [covariants.org](https://covariants.org). This use is allowable under the [GISAID Terms of Use](https://www.gisaid.org/registration/terms-of-use/).

