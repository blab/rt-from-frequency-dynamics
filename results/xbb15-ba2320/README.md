# Results for clades across countries additionally splitting out lineages XBB.1.5 and BA.2.3.20

## Variant frequencies

This shows 7-day smoothed variant frequencies. This includes a logistic growth rate from regression of logit transformed Omicron frequencies.

#### Variant frequencies on logit y axis

##### Focus on lineage XBB.1.5

![](figures/xbb15-ba2320_logistic-growth-transformed-axis-XBB15.png)

##### Focus on lineage BA.2.3.20

![](figures/xbb15-ba2320_logistic-growth-transformed-axis-BA2320.png)

## Partitioning case counts by variant

This uses 7-day smoothed daily case counts alongside 7-day smoothed variant frequencies to partition into variant-specific case counts.

#### Stacked variant case counts on natural y-axis

![](figures/xbb15-ba2320_partitioned-cases.png)

#### Variant case counts on log y-axis

##### Focus on lineage XBB.1.5

![](figures/xbb15-ba2320_partitioned-log-cases-XBB15.png)

##### Focus on lineage BA.2.3.20

![](figures/xbb15-ba2320_partitioned-log-cases-XBB15.png)

## Model outputs

These outputs are using the growth advantage random walk (GARW) model.

#### Variant-specific growth rate

##### Reproductive number _Rt_

![](figures/xbb15-ba2320_variant-rt.png)

#### Variant-specific daily case counts

##### Log y axis

![](figures/xbb15-ba2320_variant-estimated-log-cases.png)

##### Natural y axis

![](figures/xbb15-ba2320_variant-estimated-cases.png)

#### Variant-specific frequencies

![](figures/xbb15-ba2320_variant-estimated-frequency.png)
