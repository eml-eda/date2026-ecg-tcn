# DATE 2026 - PLiNIO

## Clone Repo

```shell
git clone git@github.com:eml-eda/date2026-ecg-tcn.git
cd ecg-tcn
```

## Download dataset

**NOTE:** Downloading from the original source is super slow. I found the dataset on [Kaggle](https://www.kaggle.com/datasets/garethwmch/ptb-xl-1-0-3) and got it from there, which is like 100 times faster (literally). Just save it in a `./data` subfolder in your cloned repo

```shell
mkdir data
cd data
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
mv ./physionet.org/files/ptb-xl/1.0.3/ .
rm -r ./physionet.org
```

## Requirements

```shell
pip install numpy pandas matplotlib torch tqdm scikit-learn wfdb
# extras for PLiNIO
pip install tdigest
pip install git+https://github.com/eml-eda/plinio
```

## Run

Run with mostly default options.

```shell
python3 main.py --root data/
```

# Pruning

Run pruning with PIT to obtain a circa 37k parameters model.

```Python
python3 pruning.py --root data --outdir runs_pit --pit-reg-strength 1e-06
```

# QAT

Run QAT at w8a8 precision starting from the pruned model:

```Python
 python3 qat.py --root data --outdir runs_qat --pruned-model ./runs_pit/best_pruned_model.pt
```
