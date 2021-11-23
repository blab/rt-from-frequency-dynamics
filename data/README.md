# Data

Data preparation was as follows:

1. Nextstrain-curated metadata TSV of GenBank database was downloaded Nov 22, 2021 from https://data.nextstrain.org/files/ncov/open/metadata.tsv.gz. Uncompressing this file results in `metadata.tsv` with 2,346,967 entries.

2. The metadata file was pruned to only relevant columns via:
```
tsv-select -H -f strain,date,country,division,Nextstrain_clade metadata.tsv > metadata_pruned.tsv
```

3. This `metadata_pruned.tsv` is processed in Mathematica by running the notebook `variant-frequencies-case-counts-data-prep.nb`. This results in the export of files: `location-variant-sequence-counts.tsv` and `location-case-counts.tsv` versioned here. This keeps only sequences from the USA that were collected between Jan 1, 2021 and Oct 1, 2021 resulting in 1,130,952 entries. Additionally, only variants with greater than 2000 sequences and states with greater than 5000 sequences were kept.

`location-variant-sequence-counts.tsv` contains:
```
date	location	variant	sequences
2021-01-02	Alabama	other	3
2021-01-03	Alabama	other	3
2021-01-04	Alabama	other	12
...
```

`location-case-counts.tsv` contains:
```
date	location	cases
2021-01-01	Alabama	3630
2021-01-02	Alabama	2501
2021-01-03	Alabama	2103
...
```

There will be dates that are missing sequence counts or case counts. These should be assumed to be 0.
