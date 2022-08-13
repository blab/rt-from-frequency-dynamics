# Results for Omicron across countries splitting out Omicron sublineages BA.2.75

## Variant frequencies

This shows 7-day smoothed variant frequencies. This includes a logistic growth rate from regression of logit transformed Omicron frequencies.

#### Variant frequencies on logit y axis

This includes estimate of _r_ from regression of logit Omicron BA.2.75 frequencies.

![](figures/omicron-ba275_logistic-growth-transformed-axis.png)

## Partitioning case counts by variant

This uses 7-day smoothed daily case counts alongside 7-day smoothed variant frequencies to partition into variant-specific case counts.

#### Stacked variant case counts on natural y-axis

![](figures/omicron-ba275_partitioned-cases.png)

#### Variant case counts on log y-axis

This includes estimate of _r_ from regression of logged Omicron BA.2.75 case counts.

![](figures/omicron-ba275_partitioned-log-cases.png)

## Model outputs

These outputs are using the growth advantage random walk (GARW) model.

#### Variant-specific growth rate

##### Reproductive number _Rt_

![](figures/omicron-ba275_variant-rt.png)

#### Variant-specific daily case counts

##### Log y axis

![](figures/omicron-ba275_variant-estimated-log-cases.png)

##### Natural y axis

![](figures/omicron-ba275_variant-estimated-cases.png)

#### Variant-specific frequencies

![](figures/omicron-ba275_variant-estimated-frequency.png)

## Updating

These results can be updated via:

1. Running the notebook `omicron-ba275-plotting.nb` that will update figures in `figures/` that are referenced above using data in `../../data/`.
