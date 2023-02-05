## Pango countries dataset

This dataset only includes Pango lineages with >150 sequences in US dataset. Rarer Pango lineages are collapsed into parental lineages, ie BM.1 was collapsed into BA.2.75.3.

### Metadata

1. Nextstrain-curated metadata TSV of GISAID database was downloaded. Uncompressing and renaming this file resulted in `gisaid_metadata.tsv` via:
```
nextstrain remote download s3://nextstrain-ncov-private/metadata.tsv.gz
gzip -d metadata.tsv.gz -c > gisaid_metadata.tsv
```

2. The metadata file was pruned to only relevant columns via:
```
tsv-select -H -f strain,date,country,division,QC_overall_status,Nextclade_pango gisaid_metadata.tsv > gisaid_metadata_pruned.tsv
```

### Pango aliasing

3. Download JSON from the 21L Nextclade build:
```
curl -fsSL https://staging.nextstrain.org/nextclade_sars-cov-2_21L.json -o nextclade_sars-cov-2_21L.json.gz
```

4. Decompress this file:
```
gzip -d nextclade_sars-cov-2_21L.json.gz -c > nextclade_sars-cov-2_21L.json
```

5. Extract relevant tip attributes to TSV:
```
python extract_tip_attributes.py --json nextclade_sars-cov-2_21L.json > pango_aliasing.tsv
```

This TSV looks like
```
seqName	clade	Nextclade_pango	partiallyAliased
BQ.1	22E (Omicron)	BQ.1	BA.5.3.1.1.1.1.1
BQ.1.1	22E (Omicron)	BQ.1.1	BA.5.3.1.1.1.1.1.1
```

### Counts

6. These files are processed in Mathematica by running the notebook `pango-countries_data-prep.nb`. This results in the file `pango_location-variant-sequence-counts.tsv` versioned here. These files represent heavily derived GISAID data and are equivalent to downloadable results from [outbreak.info](https://outbreak.info), [cov-spectrum.org](https://cov-spectrum.org) and [covariants.org](https://covariants.org). This use is allowable under the [GISAID Terms of Use](https://www.gisaid.org/registration/terms-of-use/).

There will be dates that are missing sequence counts. These should be assumed to be 0.
