## Omicron South Africa BA.4 BA.5

This dataset includes the following variant categories: `other`, `Delta`, `Omicron BA.1`, `Omicron BA.2`, `Omicron BA.4`, and `Omicron BA.5`.

Data preparation followed:

1. Nextstrain-curated metadata TSV of GISAID database was downloaded. Uncompressing and renaming this file resulted in `gisaid_metadata.tsv` via:
```
nextstrain remote download s3://nextstrain-ncov-private/metadata.tsv.gz
gzip -d metadata.tsv.gz -c > gisaid_metadata.tsv
```

2. The metadata file was pruned to only relevant columns via:
```
tsv-select -H -f strain,date,country,division,QC_overall_status,Nextstrain_clade,Nextclade_pango gisaid_metadata.tsv > gisaid_metadata_pruned.tsv
```

3. This `gisaid_metadata_pruned.tsv` is processed in Mathematica by running the notebook `omicron-countries-split_data-prep.nb`. This results in the export of files: `omicron-south-africa-BA4-BA5_location-variant-sequence-counts.tsv` and `omicron-south-africa-BA4-BA5_location-case-counts.tsv` versioned here. Samples labeled as `bad` in Nextclade QC are removed. These files represent heavily derived GISAID data and are equivalent to downloadable results from [outbreak.info](https://outbreak.info), [cov-spectrum.org](https://cov-spectrum.org) and [covariants.org](https://covariants.org). This use is allowable under the [GISAID Terms of Use](https://www.gisaid.org/registration/terms-of-use/).

There will be dates that are missing sequence counts or case counts. These should be assumed to be 0.
