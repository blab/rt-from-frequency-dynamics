## Variants US dataset

Data preparation followed:

1. Nextstrain-curated metadata TSV of open data in the GenBank database was downloaded. Uncompressing and renaming this file resulted in `open_metadata.tsv` via:
```
curl https://data.nextstrain.org/files/ncov/open/metadata.tsv.gz --output metadata.tsv.gz
gzip -d metadata.tsv.gz -c > open_metadata.tsv
```

2. The metadata file was pruned to only relevant columns via:
```
tsv-select -H -f strain,date,country,division,Nextstrain_clade open_metadata.tsv > open_metadata_pruned.tsv
```

3. This `open_metadata_pruned.tsv` is processed in Mathematica by running the notebook `variants-us_data-prep.nb`. This results in the export of files: `variants-us_location-variant-sequence-counts.tsv` and `variants-us_location-case-counts.tsv` versioned here. This keeps only sequences from the USA that were collected between Jan 1, 2021 and Jan 1, 2022 resulting in 1,312,651 entries. Additionally, only variants with greater than 2000 sequences and states with greater than 10,000 sequences were kept.

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
