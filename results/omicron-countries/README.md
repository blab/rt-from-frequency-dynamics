# Results for Omicron across countries

## Variant frequencies

This shows 7-day smoothed variant frequencies. This includes a logistic growth rate from regression of logit transformed Omicron frequencies.

#### Variant frequencies on natural y axis

![](figures/omicron-countries_logistic-growth-natural-axis.png)

#### Variant frequencies on logit y axis

![](figures/omicron-countries_logistic-growth-transformed-axis.png)

## Partitioning case counts by variant

This uses 7-day smoothed daily case counts alongside 7-day smoothed variant frequencies to partition into variant-specific case counts.

#### Stacked variant case counts on natural y-axis

![](figures/omicron-countries_partitioned-cases.png)

#### Variant case counts on log y-axis

This includes estimate of _r_ from regression of logged Omicron case counts.

![](figures/omicron-countries_partitioned-log-cases.png)

## Model outputs

These outputs are using the growth advantage random walk (GARW) model.

#### Variant-specific growth rate

##### Epidemic growth rate _r_ per day

![](figures/omicron-countries_variant-little-r.png)

##### Reproductive number _Rt_

![](figures/omicron-countries_variant-rt.png)

#### Variant-specific daily case counts

##### Log y axis

![](figures/omicron-countries_variant-estimated-log-cases.png)

##### Natural y axis

![](figures/omicron-countries_variant-estimated-cases.png)

#### Variant-specific frequencies

![](figures/omicron-countries_variant-estimated-frequency.png)

#### Phase plots of epidemic growth rate vs per-capita incidence

![](figures/omicron-countries_variant-cases-vs-rt.png)

## Updating

These results can be updated via:

1. Running the notebook `omicron-countries-plotting.nb` that will update figures in `figures/` that are referenced above using data in `../../data/`.
