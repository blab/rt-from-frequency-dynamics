## Omicron US dataset

_As of April 7, 2022, these datasets are no longer being updated. Instead all updates are occurring with `omicron-us-split` that partitions "Omicron" into "Omicron BA.1" and "Omicron BA.2"._

Data preparation followed:

1. Nextstrain-curated metadata TSV of GISAID database was downloaded. Uncompressing and renaming this file resulted in `gisaid_metadata.tsv` via:
```
nextstrain remote download s3://nextstrain-ncov-private/metadata.tsv.gz
gzip -d metadata.tsv.gz -c > gisaid_metadata.tsv
```

2. The metadata file was pruned to only relevant columns via:
```
tsv-select -H -f strain,date,country,division,QC_overall_status,Nextstrain_clade gisaid_metadata.tsv > gisaid_metadata_pruned.tsv
```

3. This `gisaid_metadata_pruned.tsv` is processed in Mathematica by running the notebook `omicron-us_data-prep.nb`. This results in the export of files: `omicron-us_location-variant-sequence-counts.tsv` and `omicron-us_location-case-counts.tsv` versioned here. This subsets between Nov 15, 2021 and recent samples. Samples labeled as `bad` in Nextclade QC are removed. These files represent heavily derived GISAID data and are equivalent to downloadable results from [outbreak.info](https://outbreak.info), [cov-spectrum.org](https://cov-spectrum.org) and [covariants.org](https://covariants.org). This use is allowable under the [GISAID Terms of Use](https://www.gisaid.org/registration/terms-of-use/).

As above, there will be dates that are missing sequence counts or case counts. These should be assumed to be 0.
