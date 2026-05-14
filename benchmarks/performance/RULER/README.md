# Running RULER Benchmark

- First, setup the environment by the following command. If there are any missing dependencies, just directly installing them via `pip`.
```
pip install nemo-toolkit[all] # for RULER only
```

- Then, update the model directories in `config_models.sh`

- Run the experiments by 
```
bash run.sh minicpm-4-8B-fullattn-sft synthetic
bash run.sh minicpm-4-8B-infllmv2-sft synthetic
bash run.sh minicpm-4-8B-nsa-sft synthetic
bash run.sh minicpm-4-8B-dashattn-sft synthetic
```

- Note: we provide the data we generated for a better reproducibility. These data are given in the `data4reproduce` path. If you want to use these data, please copy then to the generated folder when running the RULER benchmark.