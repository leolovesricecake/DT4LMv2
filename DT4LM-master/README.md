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
To obtain the fine-tuning datasets with the preprocessing steps mentioned in the paper, execute the .ipynb files in ./datasets folder. Specifically, ./datasets/NL-inference/rte.ipynb processes the [RTE Dataset](https://huggingface.co/datasets/nyu-mll/glue/viewer/rte), ./datasets/semantic-equivalence/mrpc.ipynb processes the [MRPC Dataset](https://huggingface.co/datasets/nyu-mll/glue/viewer/mrpc), ./datasets/sentiment-classification/sst2.ipynb processes the [SST-2 Dataset](https://huggingface.co/datasets/stanfordnlp/sst2), and ./datasets/sentiment-classification/mr.ipynb processes the [MR Dataset](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes).

To obtain the adversarial training data, follow the instruction in `./datasets/adversarial-training/sample.ipynb` file to sample the training instances for differential inputs generation. Subsequently, follow the remaining instruction to create the adversarial training dataset with the generated differential inputs.

## Experiments
### Preparation (Fine-tuning)
Example files to conduct fine-tuning is provided in `./experiments/finetune`. An example would be:
```bash
   $ bash train_albertbasev1_sst2.sh
```
This finetunes the ALBERT-base-v1 model on the SST-2 dataset, where the --dataset argument in the .sh file should be replaced by the processed dataset path. Note that, when changing the hyperparameters, please make sure that the two models within the same model pair still share the same set of hyperparameters, ensuring a fair comparison.

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

## Acknowledgement
The DT4LM framework is adapted from the [TextAttack](https://github.com/QData/TextAttack) library.