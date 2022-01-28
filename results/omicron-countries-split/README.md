# Results for Omicron across countries splitting out lineage BA.1 / clade 21K from lineage BA.2 / clade 21L

## Variant frequencies

This shows 7-day smoothed variant frequencies. This includes a logistic growth rate from regression of logit transformed Omicron frequencies.

#### Variant frequencies on natural y axis

![](figures/omicron-countries-split_logistic-growth-natural-axis.png)

#### Variant frequencies on logit y axis

![](figures/omicron-countries-split_logistic-growth-transformed-axis.png)

## Partitioning case counts by variant

This uses 7-day smoothed daily case counts alongside 7-day smoothed variant frequencies to partition into variant-specific case counts.

#### Stacked variant case counts on natural y-axis

![](figures/omicron-countries-split_partitioned-cases.png)

#### Variant case counts on log y-axis

This includes estimate of _r_ from regression of logged Omicron case counts.

![](figures/omicron-countries-split_partitioned-log-cases.png)

## Model outputs

These outputs are using the growth advantage random walk (GARW) model.

#### Variant-specific epidemic growth rate r

![](figures/omicron-countries-split_variant-little-r.png)

#### Variant-specific daily case counts

##### Log y axis

![](figures/omicron-countries-split_variant-estimated-log-cases.png)

##### Natural y axis

![](figures/omicron-countries-split_variant-estimated-cases.png)

#### Variant-specific frequencies

![](figures/omicron-countries-split_variant-estimated-frequency.png)

#### Phase plots of epidemic growth rate vs per-capita incidence

![](figures/omicron-countries-split_variant-cases-vs-rt.png)

## Updating

These results can be updated via:

1. Running the notebook `omicron-countries-split-plotting.nb` that will update figures in `figures/` that are referenced above using data in `../../data/`.
