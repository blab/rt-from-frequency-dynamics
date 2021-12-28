# Data

## Variants US dataset

Data preparation followed:

1. Nextstrain-curated metadata TSV of GenBank database was downloaded Nov 22, 2021 from https://data.nextstrain.org/files/ncov/open/metadata.tsv.gz. Uncompressing and renaming this file resulted in `open_metadata.tsv` with 2,346,967 entries.

2. The metadata file was pruned to only relevant columns via:
```
tsv-select -H -f strain,date,country,division,Nextstrain_clade open_metadata.tsv > open_metadata_pruned.tsv
```

3. This `open_metadata_pruned.tsv` is processed in Mathematica by running the notebook `variants-us_data-prep.nb`. This results in the export of files: `variants-us_location-variant-sequence-counts.tsv` and `variants-us_location-case-counts.tsv` versioned here. This keeps only sequences from the USA that were collected between Jan 1, 2021 and Oct 1, 2021 resulting in 952,091 entries. Additionally, only variants with greater than 2000 sequences and states with greater than 5000 sequences were kept.

`variants-us_location-variant-sequence-counts.tsv` contains:
```
date	location	variant	sequences
2021-01-02	Alabama	other	3
2021-01-03	Alabama	other	3
2021-01-04	Alabama	other	12
...
```

`variants-us_location-case-counts.tsv` contains:
```
date	location	cases
2021-01-01	Alabama	3630
2021-01-02	Alabama	2501
2021-01-03	Alabama	2103
...
```

There will be dates that are missing sequence counts or case counts. These should be assumed to be 0.

### Analysis with Jupyter notebook

Create a conda environment for the analysis and load the Jupyter notebook.

``` bash
conda create -n rt python jupyterlab pandas
conda activate rt
jupyter lab variants-us_data-prep.py.ipynb
```

Run all cells in the Jupyter notebook.

## Omicron countries dataset

Data preparation followed:

1. Nextstrain-curated metadata TSV of GISAID database was downloaded. Uncompressing and renaming this file resulted in `gisaid_metadata.tsv` via:
```
nextstrain remote download s3://nextstrain-ncov-private/metadata.tsv.gz
gzip -d metadata.tsv.gz -c > gisaid_metadata.tsv
```

2. The metadata file was pruned to only relevant columns via:
```
tsv-select -H -f strain,date,country,division,Nextstrain_clade gisaid_metadata.tsv > gisaid_metadata_pruned.tsv
```

3. This `gisaid_metadata_pruned.tsv` is processed in Mathematica by running the notebook `omicron-countries_data-prep.nb`. This results in the export of files: `omicron-countries_location-variant-sequence-counts.tsv` and `omicron-countries_location-case-counts.tsv` versioned here. This subsets between Sep 1, 2021 and recent samples. These files represent heavily derived GISAID data and are equivalent to downloadable results from [outbreak.info](https://outbreak.info), [cov-spectrum.org](https://cov-spectrum.org) and [covariants.org](https://covariants.org). This use is allowable under the [GISAID Terms of Use](https://www.gisaid.org/registration/terms-of-use/).

As above, there will be dates that are missing sequence counts or case counts. These should be assumed to be 0.

## Omicron US dataset

Data preparation followed:

1. Nextstrain-curated metadata TSV of GISAID database was downloaded. Uncompressing and renaming this file resulted in `gisaid_metadata.tsv` via:
```
nextstrain remote download s3://nextstrain-ncov-private/metadata.tsv.gz
gzip -d metadata.tsv.gz -c > gisaid_metadata.tsv
```

2. The metadata file was pruned to only relevant columns via:
```
tsv-select -H -f strain,date,country,division,Nextstrain_clade gisaid_metadata.tsv > gisaid_metadata_pruned.tsv
```

3. This `gisaid_metadata_pruned.tsv` is processed in Mathematica by running the notebook `omicron-us_data-prep.nb`. This results in the export of files: `omicron-us_location-variant-sequence-counts.tsv` and `omicron-us_location-case-counts.tsv` versioned here. This subsets between Oct 1, 2021 and recent samples. These files represent heavily derived GISAID data and are equivalent to downloadable results from [outbreak.info](https://outbreak.info), [cov-spectrum.org](https://cov-spectrum.org) and [covariants.org](https://covariants.org). This use is allowable under the [GISAID Terms of Use](https://www.gisaid.org/registration/terms-of-use/).

As above, there will be dates that are missing sequence counts or case counts. These should be assumed to be 0.

## SGTF King County dataset

Data preparation followed:

1. UW Virology data on SGTF counts from https://github.com/proychou/sgtf

2. Download Seattle Flu Study "S positive" counts from https://backoffice.seattleflu.org/metabase/question/822. Download as CSV and rename to `sfs_counts.csv`.

3. King County cases from https://github.com/nytimes/covid-19-data/blob/master/rolling-averages/us-counties-recent.csv

4. Processed in Mathematica by running the notebook `sgtf-king-county_data-prep.nb`. This results in the export of files: `sgtf-king-county_location-variant-sequence-counts.tsv` and `sgtf-king-county_location-case-counts.tsv`.
