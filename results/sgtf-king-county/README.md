# Results for SGTF in King County

This is using SGTF as a proxy for "probable Omicron". SGTF data from [UW Virology](https://github.com/proychou/sgtf) and Seattle Flu Study.

## Variant frequencies

This shows 7-day smoothed variant frequencies. This includes a logistic growth rate from regression of logit transformed Omicron frequencies.

#### Variant frequencies on natural y axis

![](figures/sgtf-king-county_logistic-growth-natural-axis.png)

#### Variant frequencies on logit y axis

![](figures/sgtf-king-county_logistic-growth-transformed-axis.png)

## Partitioning case counts by variant

This uses 7-day smoothed daily case counts alongside 7-day smoothed variant frequencies to partition into variant-specific case counts.

#### Stacked variant case counts on natural y-axis

![](figures/sgtf-king-county_partitioned-cases.png)

#### Variant case counts on log y-axis

This includes estimate of _r_ from regression of logged Omicron case counts.

![](figures/sgtf-king-county_partitioned-log-cases.png)

## Model outputs

These outputs are using the "fixed growth" model.

#### Variant-specific Rt

![](figures/sgtf-king-county_variant-rt.png)

#### Variant-specific epidemic growth rate r

![](figures/sgtf-king-county_variant-little-r.png)

#### Variant-specific daily case counts

#### Log y-axis

![](figures/sgtf-king-county_variant-estimated-log-cases.png)

#### Natural y-axis

![](figures/sgtf-king-county_variant-estimated-cases.png)

## Updating

These results can be updated via:

1. Running the notebook `sgtf-king-county-plotting.nb` that will update figures in `figures/` that are referenced above using data in `../../data/`.
