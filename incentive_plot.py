from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import *
import matplotlib.lines as mlines



from numpy.random import default_rng
import os
from scipy.stats import  binom
import matplotlib.pyplot as plt
import click
import numpy as np
import torch
import wandb
import pandas as pd
from accelerate import Accelerator
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils.argument import ScriptArguments
from utils.data import (
    RewardDataCollatorWithPadding,
    build_dataset,
    post_filter_by_ratio,
    get_data,
)
from utils.trainer import (
    BTTRewardTrainer,
    RewardTrainer,
    RewardTrainerWithOracleCE,
    RewardTrainerWithRingeMargin,
    compute_CE_oracle,
    compute_ML_oracle,
)

from calibration_module.utils import compute_binary_score,compute_calibration_summary
from calibration_module.calibrator import HistogramCalibrator

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set for all GPUs
    set_seed(seed)  # Hugging Face's Trainer consistency

    # Ensure deterministic behavior across multi-GPU environments
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def rej_prob_simulator(
    empirical_data,
    T_func,
    R_func,
    num_experiments,
    num_sampled_data,
    use_log,
    r):

    # Step 1: Generate all random samples at once (vectorized)
    n_data = len(empirical_data)
    rng = np.random.default_rng()
    sample_indices = rng.integers(low=0, high=n_data, size=(num_experiments, num_sampled_data))
        
    # shape = (num_experiments, sample_size)
    D_samples = empirical_data[sample_indices]



    t_values = T_func(D_samples)
    R_values = R_func(D_samples, r)


    # membership is a boolean array of shape (num_experiments,)
    membership = (t_values >= R_values[:, 0]) & (t_values <= R_values[:, 1])

    # Probability estimate is average of membership
    prob_est = membership.mean()
    if not use_log:
        return -prob_est
    else:
        return np.log(1-prob_est)



def rej_prob(mode, original_preference,  num_sampled_data,use_log,r,R_up=None):
    if mode=='self':
        p_mean=(1+r)/2
    else:
        #check if all original_preference>1/2
        if not np.all(original_preference>=1/2):
            raise ValueError("All original preference should be greater than 1/2")
    
        p_mean=np.mean(original_preference)*r+(1-r)/2
    if R_up is None:
       R_up=p_mean
    

    
    prob=binom.cdf(num_sampled_data* R_up, num_sampled_data, p_mean)
    if use_log:
        return np.log(1-prob)
    else:
        return -prob


def reg_prob_derivative(mode, r, original_preference, T_func, R_func, num_experiments=10000, num_sampled_data=1000, use_log=True,h=1e-6):
    """
    Compute the derivative of the rejection probability with respect to r.
    """
    
    empirical_data=agent_data_simulator(original_preference,r, mode)
    v1=rej_prob_simulator(empirical_data, T_func, R_func, num_experiments, num_sampled_data,use_log,r)
    empirical_data=agent_data_simulator(original_preference,r+h, mode)
    v2=rej_prob_simulator(empirical_data, T_func, R_func, num_experiments, num_sampled_data,use_log,r+h)
    dev=(v2-v1)/h


    return dev




def reg_prob_derivative_exact(mode, r, original_preference, T_func, R_func, num_experiments=10000, num_sampled_data=1000, use_log=True,h=5e-7):
    """
    Compute the derivative of the rejection probability with respect to r.
    """
    
   
    v1=rej_prob(mode, original_preference, num_sampled_data,use_log,r)
    v2=rej_prob(mode, original_preference, num_sampled_data,use_log,r+h)
    dev=(v2-v1)/(h)
    return dev
 


def agent_data_simulator(original_preference,r, mode=None):
    length=len(original_preference)
    
    if mode=='self':
        mean_values=(1+r)/2
    else:
        #check if all original_preference>1/2
        if not np.all(original_preference>=1/2):
            raise ValueError("All original preference should be greater than 1/2")
        mean_values=original_preference*r+(1-r)/2

    
    #generate bernoulli samples
    samples=np.random.binomial(1,mean_values,length)
    return samples



def T_create(data_batch):
        # data_batch.mean(axis=1) => shape (num_experiments,)
        return data_batch.mean(axis=1)

def R_create(Ep, mode, data_batch, r):
        # so shape => (num_experiments, 2)
        num_experiments = data_batch.shape[0]
        r=r+1e-4
        if mode =='self':
            high=(1+r)/2
        else:
            high=Ep*r+(1-r)/2
        high=high
        low=0
        # Create a 2D array of repeated intervals
        intervals = np.column_stack((np.full(num_experiments, low),
                                     np.full(num_experiments, high)))
        return intervals    


def get_incentive(
    preference_score,
    r,
    N,
    mode,
    exact=False,
    use_log=False
):
    """
    Compute the incentive for a given preference score vector and r.
    """
    if mode=='self':
        Ep=None
    else:
        Ep=np.mean(preference_score)
    if exact:
        dev=reg_prob_derivative_exact(mode,r, preference_score, T_create, R_create, num_experiments=50000, num_sampled_data=N, use_log=use_log)
    else:
        R_func=partial(R_create, Ep, mode)
        dev=reg_prob_derivative(mode, r, preference_score, T_create, R_func, num_experiments=50000, num_sampled_data=N, use_log=use_log)
    return dev



def calibrate_data(train_dataset,eval_dataset,n_bins=20):
    preference_score = np.array(train_dataset['preference_score'])
    labels=np.array(train_dataset['label'])
    histogram = HistogramCalibrator(n_bins=n_bins)
    histogram.fit(preference_score, labels)
    histogram_probs = histogram.predict(np.array(eval_dataset['preference_score']))
    return histogram_probs


def incentive_plot(
    name #data can be 'sky','PKU','Helpsteer','Ultra'
):
    # change default style figure and font size
    
    n_bins = 30

    # Set the random seed for reproducibility
    seed = 4    
    set_random_seed(seed)


    exact=True

    r_list=[0.9,0.7,0.5,0.3,0.1]
    N_list=np.arange(10,500,20)


    train_dataset,eval_dataset=get_data(script_config_path='/home/zhongzec24/RewardModeling/paper_experiment_configs/llama-'+name+'.json')
 
    
    if name=='sky':    
        histogram_probs=calibrate_data(train_dataset,eval_dataset,n_bins=n_bins)
        #for elements in histogram_probs, if it is less than 0.5, set them to 1-histogram_probs
        
    else:
        histogram_probs=np.array(eval_dataset['preference_score'])
    histogram_probs=np.where(histogram_probs<0.5,1-histogram_probs,histogram_probs)

    incentive_list=[]
    for mode in ['self','expert']:
        for r in r_list:
            for N in N_list:
                incentive=get_incentive(histogram_probs,r,N,mode,exact)
                incentive_list.append(incentive)
    incentive_list=np.array(incentive_list).reshape(2,len(r_list),len(N_list))

    plt.figure(figsize=(10,10))
    mode_line=['-','--']
    #color_list the same length as r_list wtih color bar of red 
    num=len(r_list)
    color_list = plt.cm.Blues(np.linspace(0.5, 0.9, num))
    mode_name=['Self','Expert']
    for i,mode in enumerate(['self','expert']):
        for j,r in enumerate(r_list):
            plt.plot(N_list,incentive_list[i,j],color=color_list[j],linestyle=mode_line[i]
            ,markersize=10,linewidth=5)
    plt.xlabel("$n$")
    plt.ylabel("Incentive")
    #add grid
    plt.grid()

    r_legend_handles = []
    for j, r in enumerate(r_list):
        # "Dummy" line2D just for the legend
        line = mlines.Line2D(
            [], [], 
            color=color_list[j],
            linestyle='-',   # style doesn’t matter here, pick any
            label=f"$\eta =$ {r}",
            markersize=10,
            linewidth=5

        )
        r_legend_handles.append(line)

    legend_r = plt.legend(
        handles=r_legend_handles,
        title='$\eta$',
        loc='upper left'
    )

 
    mode_legend_handles = []
    for i, m_name in enumerate(mode_name):
        # "Dummy" line2D just for the legend
        line = mlines.Line2D(
            [], [],
            color='blue',         # color doesn’t matter, pick a neutral color
            linestyle=mode_line[i],
            label=m_name,
            markersize=10,
            linewidth=5
        )
        mode_legend_handles.append(line)

    legend_mode = plt.legend(
        handles=mode_legend_handles,
        title='Method',
        loc='upper right'
    )

    plt.gca().add_artist(legend_r)
    plt.gca().add_artist(legend_mode)

    os.makedirs("figs", exist_ok=True)
    sv_name=name
    sv_name+='_incentive_n_sample.eps'
    plt.savefig("figs/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure
    '''
    #plot x-axix as r
    plt.figure(figsize=(10,10))
    for i,N in enumerate(N_list):
        plt.plot(r_list,incentive_list[:,i],label=f"N={N}",marker='o')
    plt.xlabel("r")
    plt.ylabel("Incentive")
    plt.legend()
    os.makedirs("fig", exist_ok=True)
    sv_name=name+'_'+mode+'_'
    sv_name+='_incentive_r.eps'
    plt.savefig("fig/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure
    '''


def prob_plot_in_r(
    name #data can be 'sky','PKU','Helpsteer','Ultra'
):
    # change default style figure and font size
    
    n_bins = 30

    # Set the random seed for reproducibility
    seed = 4    
    set_random_seed(seed)

    mode='expert'
    exact=True
    R_up=0.8

    r_list=[0.1,0.3,0.5,0.7,0.9]
    r_list=np.linspace(0.01,0.9,101)

    N_list=[50,100,500,1000]



    train_dataset,eval_dataset=get_data(script_config_path='/home/zhongzec24/RewardModeling/paper_experiment_configs/llama-'+name+'.json')
    
    if name=='sky':    
        histogram_probs=calibrate_data(train_dataset,eval_dataset,n_bins=n_bins)

        #for elements in histogram_probs, if it is less than 0.5, set them to 1-histogram_probs
        
    else:
        histogram_probs=np.array(eval_dataset['preference_score'])
    histogram_probs=np.where(histogram_probs<0.5,1-histogram_probs,histogram_probs)
    incentive_list=[]
    for r in r_list:
        for N in N_list:
            incentive=-rej_prob(mode, histogram_probs, N,False,r,R_up)
            incentive_list.append(incentive)
    incentive_list=np.array(incentive_list).reshape(len(r_list),len(N_list))


    #plot x-axix as r
    plt.figure(figsize=(10,10))
    for i,N in enumerate(N_list):
        plt.plot(r_list,incentive_list[:,i],label=f"N={N}",marker='o')
    plt.xlabel("r")
    plt.ylabel("Rej. Prob")
    plt.legend()
    os.makedirs("fig", exist_ok=True)
    sv_name=name+'_'+mode+'_'
    sv_name+='_prob_r.eps'
    plt.savefig("fig/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure



def main():
    plt.rcParams['font.size'] = 26
    for data_name in ['PKU','sky','Helpsteer','Ultra']:
        incentive_plot(data_name)
        #prob_plot_in_r(data_name)



if __name__ == "__main__":
    main()


