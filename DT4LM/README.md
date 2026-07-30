<h1 align="center">DT4LM</h1>


<p align="center">
<b>
Differential Testing for Reliable Language Model Updates.</b>

## Dependencies
The environment needed to run the experiments are packaged in DT4LM.yaml file. To create the environment, run:
```bash
   $ conda env create -f DT4LM.yaml
```
Then, activate the environment with:
```bash
   $ conda activate DT4LM
```
When the code repository is initialized, or whenever there's a change in the code logic, run:
```bash
   $ pip install .
```
under ~/DT4LM directory to reconstruct the code environment.

## Repo Structure
- `additional_results`: provides supplementary experimental results for RQ1 and RQ4, which were omitted in the paper.
- `datasets`: contains scripts that facilitate data preprocessing and adversarial training data generation. Refer to the "Datasets" section below for details.
- `docs`: contains configurations for code running environment.
- `experiments`: contains scripts for running experiments, for details, please refer to the Experiments section below.
- `statistics`: provides .ipynb files to calculate modification rates, conduct quality assessment (RQ2), and perform statistical testing.
- `textattack`: includes implementations of goal functions and various base recipes. Additionally, different constraints, transformation methods, and search methods are provided, allowing for the creation of new recipes.


## Datasets
The original fine-tuning notebooks are available as a local-output CLI for
SST-2, RTE, MRPC, and MR:

```bash
python datasets/preprocess_dataset.py sst2
python datasets/preprocess_dataset.py rte
```

Outputs default to `outputs/datasets/<dataset>` and can be changed with
`--output-dir`. TextAttack training and attack commands load these local
`save_to_disk` directories directly. Run
`python datasets/preprocess_dataset.py --help` for all source, revision, split,
and optional Hub upload settings.

The adversarial-training notebook is converted to the `sample` and `combine`
subcommands in `datasets/prepare_adversarial_training.py`.

## Experiments
### Preparation (Fine-tuning)
Example files to conduct fine-tuning is provided in `./experiments/finetune`. An example would be:
```bash
   $ bash experiments/finetune/train_albertbasev1_sst2.sh
```
This fine-tunes ALBERT-base-v1 on the default local SST-2 output. A custom
dataset directory and model output directory can be passed as the first and
second arguments. When changing hyperparameters, keep both models in a pair
symmetric to ensure a fair comparison.

For evaluating the model's performance on the test set, run:
```bash
   $ bash evaluate_albertbasev2_sst2.sh
```
The --model argument in the .sh file specifies the path to load the model to be evaluated.

### RQ1: Differential Input Generation (additional results can be found in the `./additional_results/RQ1` folder)
To conduct differential testing with DT4LM, go to the `./experiments/difftest` folder and run the following command (for example):
```bash
   $ bash pair_leap_albertbasev2_sst2.sh
```
This conducts differential testing based on the leap recipe with the SST-2 dataset.

For generating differential inputs with the baseline method, go to the `./experiments/baseline` folder, after running:
```bash
   $ bash leap_debertav3base_sst2.sh
```
Continue to evaluate the old model on the adversarial examples generated for the new model to obtain differential inputs:
```bash
   $ bash debertabase_sst2_leap.sh
```
The differential inputs will be automatically saved for further analysis.

### RQ2: Test Input Quality
We detail the steps to assess test input quality in `./statistics/quality_assessment.ipynb`, the file includes steps for processing the datasets and conduct quality assessment with selected evaluation metrics.

### RQ3: Adversarial Training
There are four steps in adversarial training.
- `Sample Data`: sample the dataset with the above-described .ipynb file.
- `Generate`: generate differential inputs based on different methods, following the same procedures as RQ1 (with dataset changed, e.g., `./experiments/difftest/pair_leap_albertbasev2_sst2_adv.sh`).
- `Mix and Train`: create the adversarial training dataset with the provided .ipynb file and fine-tune the model with the obtained dataset and original hyperparameters (e.g., `./experiments/finetune/advtrain_albertbasev2_sst2_leap.sh`).
- `Evaluate`: evaluate the robustness improvement and the impact on clean accuracy (e.g., `./experiments/baseline/debertabase_sst2_leap.sh`, with model and dataset changed).

### RQ4: Ablation Study (additional results can be found in the `./additional_results/RQ4` folder)
To conduct ablation study, replace the current goal function design with the naive goal function design in `textattack/goal_functions/classification/differential_classification.py`. Subsequently, follow the same instructions in RQ1 to conduct differential input generation and compare the results.

## SemDT and LexiDT Improvements

The first-round workflow uses one shared `pair` recipe with orthogonal objective
and semantic-constraint switches. Dataset hyperparameters and individual
experiment definitions live in separate YAML files:

```text
--differential-objective dynamic|static|lexi
--semantic-constraint original|nli
```

Freeze the config-selected jointly-correct train/test manifests:

```bash
bash experiments/improvements/prepare_manifests.sh \
  experiments/improvements/configs/sst2.yaml
```

Run one selected judge backend at a time:

```bash
bash experiments/improvements/calibrate_semdt.sh \
  experiments/improvements/configs/sst2.yaml \
  configs/openai.secert.yaml
```

Run exactly one experiment at a time:

```bash
bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2.yaml \
  experiments/improvements/configs/experiments/base.yaml
```

The complete Chinese guide, including model training, smoke-test
hyperparameters, independent OpenAI/HF calibration, all experiment commands,
trajectory audits, automatic metrics, and human evaluation, is
[`../docs/DT4LM-改进实验完整指南.md`](../docs/DT4LM-改进实验完整指南.md).

## Acknowledgement
The DT4LM framework is adapted from the [TextAttack](https://github.com/QData/TextAttack) library.
