## Setup

The code is tested on Python 3.10 and cuda 12.1. Please make sure you have installed cuda >= 11.6 and satisfy the minimal requirement for flash attention.
You can use either
```bash
conda env create -f py310_env.yaml
```
to directly create the conda environment or manually install the required packages by running
```bash
conda create -n rm_dev python=3.10.15
conda activate rm_dev    

pip3 install torch==2.1.2 torchvision torchaudio 
pip3 install numpy==1.26.4
pip3 install flash-attn==2.6.3
pip3 install accelerate==0.33.0 
pip3 install deepspeed==0.12.2
pip3 install transformers==4.43.4
pip3 install wandb peft click datasets sentencepiece bitsandbytes rewardbench loguru
pip3 install "fschat[model_worker,webui]"
pip3 install "huggingface_hub[cli]"
```

Then please login the wandb account by running `wandb login` and huggingface account by running `huggingface-cli login`.

## Usage

We use json configs to manage the experiment settings. You can find all the experiment configs in `paper_experiment_configs/`. To reproduce, first prepare the oracle-annotated dataset by running 
```bash
python prepare_oracle_data.py
```
It would download and annotate the dataset.

To run the experiments, 
```bash
python Section3.py
python Section4.py
```

### Delta sensitivity for Section 4
- Sweep disagreement probabilities (`delta`) across self-consistency and expert monitoring with the same contract grids used in Section 4 and dump results to CSV/plots:  
  ```bash
  python Section4_delta_sensitivity.py
  ```
- Custom deltas and CSV-only (skip EPS plots):  
  ```bash
  python Section4_delta_sensitivity.py --delta 0 0.02 0.05 0.1 0.2 0.3 --skip-plots
  ```
- The script writes plots to `fig_contract/<mode>/...` and a tidy summary table to `fig_contract/delta_sweep_summary.csv` (override with `--summary-csv`). 

### Downstream reward-model experiment (contracts vs monitoring)
- Prepare oracle data once: `python prepare_oracle_data.py`
- Simulate annotated datasets under self-consistency vs expert monitoring and linear/binary contracts (fair monitoring budget):  
  `python downstream_contract_experiment.py --dataset helpsteer --dataset pku --monitor self --monitor expert --contract`
- The script saves simulated datasets to `statdata/simulated/...` with a standard 80/20 train/test split, corrupts only the train split, and keeps the original test split for evaluation. It emits JSON configs in `paper_experiment_configs/` that point `run.py` to train (simulated) and eval (original) paths. Add `--train` to launch reward-model training immediately for each scenario.
- Baselines generated automatically for every dataset/monitor/contract combination:
  - `*-clean`: training split kept uncorrupted.
  - `*-fully_corrupted`: training split corrupted with eta=0 (complete noise).
  - `*-eta{X}`: training split corrupted using the solved eta for that monitor (self/expert).
  Use the same command as above (optionally with `--train`) and all configs/datasets will be written and, if `--train` is set, trained in sequence.
- `run.py` defaults to *not* saving model checkpoints/weights; it still writes `trainer_state.json` (metrics/state). Add `--save-model` when invoking `run.py` if you want checkpoints and weights under `last_ckpt/`.
- **Plotting the downstream comparison:** once reward-model runs finish and you have `bt_models/<Dataset>-<monitor>-<contract>-{eta*/clean/fully_corrupted}/<run>/trainer_state.json`, generate a summary chart (default is oracle CE loss, lower is better) with  
  ```bash
  python3 visualize_downstream_rm.py \
    --results-root bt_models \
    --output fig/downstream_reward_models.png
  ```  

## Acknowledgements
This codebase is built on top of [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/bradley-terry-rm). Special thanks to its creators for their valuable contributions and insights.
