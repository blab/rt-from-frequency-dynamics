# Scripts

This directory includes various scripts to run models for estimating variant $R_{t}$.


## run_config.py

`run_config.py` uses a configuration file of the form `../estimates/sgtf-king-county/config.yaml` to specify the model type to be used as well as to specify model parameters.

One can run this script using a command like

```
poetry run python3 run_config.py --config <path-to-config>
```


