# Results for Omicron across countries splitting out Omicron sublineages / subclades: BA.1 / 21K, BA.2 / 21L, BA.4 / 22A, BA.5 / 22B, BA.2.12.1 / 22C

## Variant frequencies

This shows 7-day smoothed variant frequencies. This includes a logistic growth rate from regression of logit transformed Omicron frequencies.

#### Variant frequencies on logit y axis

##### Focus on lineage BA.5 / clade 22B

![](figures/omicron-countries-split_logistic-growth-transformed-axis-22B.png)

## Partitioning case counts by variant

This uses 7-day smoothed daily case counts alongside 7-day smoothed variant frequencies to partition into variant-specific case counts.

#### Stacked variant case counts on natural y-axis

![](figures/omicron-countries-split_partitioned-cases.png)

#### Variant case counts on log y-axis

This includes estimate of _r_ from regression of logged Omicron case counts.

![](figures/omicron-countries-split_partitioned-log-cases.png)

##### Focus on lineage BA.5 / clade 22B

![](figures/omicron-countries-split_partitioned-log-cases-22B.png)

## Model outputs

These outputs are using the growth advantage random walk (GARW) model.

#### Variant-specific growth rate

##### Reproductive number _Rt_

![](figures/omicron-countries-split_variant-rt.png)

#### Variant-specific daily case counts

##### Log y axis

![](figures/omicron-countries-split_variant-estimated-log-cases.png)

##### Natural y axis

![](figures/omicron-countries-split_variant-estimated-cases.png)

#### Variant-specific frequencies

![](figures/omicron-countries-split_variant-estimated-frequency.png)

## Updating

These results can be updated via:

1. Running the notebook `omicron-countries-split-plotting.nb` that will update figures in `figures/` that are referenced above using data in `../../data/`.
