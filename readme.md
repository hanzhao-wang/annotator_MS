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

### Downstream reward-model experiment (contracts vs monitoring)
- Prepare oracle data once: `python prepare_oracle_data.py`
- Simulate annotated datasets under self-consistency vs expert monitoring and linear/binary contracts (fair monitoring budget):  
  `python downstream_contract_experiment.py --dataset helpsteer --dataset pku --monitor self --monitor expert --contract`
- The script saves simulated datasets to `statdata/simulated/...` with a standard 80/20 train/test split, corrupts only the train split, and keeps the original test split for evaluation. It emits JSON configs in `paper_experiment_configs/` that point `run.py` to train (simulated) and eval (original) paths. Add `--train` to launch reward-model training immediately for each scenario.
- **Plotting the downstream comparison:** once reward-model runs finish and you have `bt_models/<Dataset>-<monitor>-<contract>-eta*/<run>/trainer_state.json`, generate a summary chart (default is oracle CE loss, lower is better) with  
  ```bash
  python3 visualize_downstream_rm.py \
    --results-root bt_models \
    --output fig/downstream_reward_models.png
  ```  
  Pass `--metric eval_binary_accuracy --higher-is-better` to switch to accuracy, and `--show` to display the figure interactively. The script writes the plot to `fig/downstream_reward_models.png` and prints per-scenario means/standard deviations so you can compare self vs. expert monitoring performance on the clean eval split.

## Acknowledgements
This codebase is built on top of [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/bradley-terry-rm). Special thanks to its creators for their valuable contributions and insights.
