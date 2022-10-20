## Pango countries dataset

This dataset only includes Pango lineages with >150 sequences in US dataset. Rarer Pango lineages are collapsed into parental lineages, ie BM.1 was collapsed into BA.2.75.3.

Data preparation followed:

1. Nextstrain-curated metadata TSV of GISAID database was downloaded. Uncompressing and renaming this file resulted in `gisaid_metadata.tsv` via:
```
nextstrain remote download s3://nextstrain-ncov-private/metadata.tsv.gz
gzip -d metadata.tsv.gz -c > gisaid_metadata.tsv
```

2. The metadata file was pruned to only relevant columns via:
```
tsv-select -H -f strain,date,country,division,QC_overall_status,Nextclade_pango gisaid_metadata.tsv > gisaid_metadata_pruned.tsv
```

3. Install `nextclade` CLI following instructions at [docs.nextstrain.org](https://docs.nextstrain.org/projects/nextclade/en/stable/user/nextclade-cli.html)

4. Provision `sars-cov-2-21L` dataset with:
```
nextclade dataset get --name 'sars-cov-2-21L' --output-dir 'sars-cov-2-21L'
```

5. Download canonical Pango sequences provisioned by [@corneliusroemer](https://github.com/corneliusroemer) via:
```
curl -fsSL https://github.com/corneliusroemer/pango-sequences/blob/main/data/pango_consensus_sequences.fasta.zstd?raw=true -o pango_consensus_sequences.fasta.zstd
```
And decompress with:
```
zstdcat pango_consensus_sequences.fasta.zstd > pango_consensus_sequences.fasta
```

6. Run Nextclade on each Pango lineage with:
```
nextclade run \
   --input-dataset sars-cov-2-21L \
   --output-all=output/ \
   pango_consensus_sequences.fasta
```

This will generate the file `output/nextclade.tsv` containing columns `Nextclade_pango` and `partiallyAliased` and rows for each Pango lineage.

7. Prune this file to relevant columns:
```
tsv-select -H -f seqName,clade,Nextclade_pango,partiallyAliased output/nextclade.tsv > output/nextclade_pruned.tsv
```

8. Prune this file to only relevant rows:
```
tsv-filter -H --str-ne clade:outgroup output/nextclade_pruned.tsv > pango_aliasing.tsv
```

9. This `gisaid_metadata_pruned.tsv` is processed in Mathematica by running the notebook `pango-countries_data-prep.nb`. This results in the file `pango_location-variant-sequence-counts.tsv` versioned here. These files represent heavily derived GISAID data and are equivalent to downloadable results from [outbreak.info](https://outbreak.info), [cov-spectrum.org](https://cov-spectrum.org) and [covariants.org](https://covariants.org). This use is allowable under the [GISAID Terms of Use](https://www.gisaid.org/registration/terms-of-use/).

There will be dates that are missing sequence counts. These should be assumed to be 0.
