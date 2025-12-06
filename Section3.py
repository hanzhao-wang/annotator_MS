from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import *
import matplotlib.lines as mlines 
import os
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
from scipy.optimize import bisect
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




def histogram_plot(preference_score,name=''):
    ####plot histogram
    
    preference_score=np.array(preference_score)
    preference_score=np.where(preference_score<0.5,1-preference_score,preference_score)

    plt.figure(figsize=(10, 10))  
    #change the frequency to probability
    #color is light blue, edgecolor is black

    plt.hist(preference_score, bins=15, density=True, alpha=0.75, color='lightblue', edgecolor='black')
    plt.xlabel("$\mathbb{P}(y_{chosen} \succ y_{rejected} \mid x)$")
    plt.ylabel("Frequency (%)")
    #add mean value as the legend in the plot
    plt.axvline(np.mean(preference_score), color='b', linestyle='dashed', linewidth=3)
    plt.legend([f'Mean: {np.mean(preference_score):.2f}'])
    os.makedirs("fig", exist_ok=True)
    sv_name=name+'_'
    sv_name+='_histogram.eps'
    plt.savefig("figs/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure

    plt.show()

def calibrate_data(train_dataset,eval_dataset,n_bins=20):
    preference_score = np.array(train_dataset['preference_score'])
    labels=np.array(train_dataset['label'])
    histogram = HistogramCalibrator(n_bins=n_bins)
    histogram.fit(preference_score, labels)
    histogram_probs = histogram.predict(np.array(eval_dataset['preference_score']))
    return histogram_probs
def calibrate_plot(eval_dataset,histogram_probs,n_bins=20,name=''):
    score_col = 'score'
    label_col='label'
    df_pre = pd.DataFrame({
        label_col: eval_dataset['label'],
        score_col: eval_dataset['preference_score']
    })
    df_post = pd.DataFrame({
        label_col: eval_dataset['label'],
        score_col:  histogram_probs
    })

    # key to the dictionary is for giving the result
    # a descriptive name
    eval_dict = {
        'Before Calibrate': df_pre,
        'After Calibrate': df_post
    }


    plt.rcParams['figure.figsize'] = [14, 14]
  
    df_result = compute_calibration_summary(eval_dict, label_col, score_col, n_bins=n_bins,
                                            save_plot_path='figs/'+name+'_calibration.eps')
    #print(df_result)

def KL_bernoulli(expert_prob_vec,anno_prob_vector,flip_prob_1=0,flip_prob_2=0):
    

    
    prob_vec_1=2*(1-flip_prob_1)*(expert_prob_vec-1/2)*(anno_prob_vector-1/2)+1/2
    prob_vec_2=2*(1-flip_prob_2)*(expert_prob_vec-1/2)*(anno_prob_vector-1/2)+1/2
    #truncate prob_vec to 0.0001 and 0.9999 to avoid log(0)
    prob_vec_1=np.clip(prob_vec_1,0.0001,0.9999)
    prob_vec_2=np.clip(prob_vec_2,0.0001,0.9999)
    KL=prob_vec_1*np.log(prob_vec_1/prob_vec_2)+(1-prob_vec_1)*np.log((1-prob_vec_1)/(1-prob_vec_2))
    
    return KL

# Binary entropy in bits.
def binary_entropy(p):
    if p == 0 or p == 1:
        return 0.0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

# Function whose root gives the optimal p^*
def f(p, c_bar):
    # Using natural logarithm here.
    return np.log((0.5 + c_bar*p)/(0.5 - c_bar*p)) - (1 - binary_entropy(0.5 + c_bar))/c_bar

# Mutual information as a function of p.
def mutual_information(p, c_bar):
    return binary_entropy(0.5 + c_bar*p) - ((1-p)*1 + p*binary_entropy(0.5 + c_bar))

def Mul_inf_max(expert_prob_vec,anno_prob_vector):
    p_low=0.0001
    p_high=0.9999
    c_bar=np.mean(np.abs(2* (expert_prob_vec-0.5)*(anno_prob_vector-0.5)))
    p_star = bisect(f, p_low, p_high, args=(c_bar,), xtol=1e-10)
    I_max = mutual_information(p_star, c_bar)
    return I_max

def Mul_inf_compute(expert_prob_vec,anno_prob_vector,eta_prob_vec): 
    N=len(expert_prob_vec)
    K=len(eta_prob_vec)
    print(N,K)
    MI=0
    for i in range(N):
        prob_margin_1=0
        prob_margin_0=0
        for k in range(K):
            prob_1=2*(1-eta_prob_vec[k])*(expert_prob_vec[i]-1/2)*(anno_prob_vector[i]-1/2)+1/2
            prob_0=1-prob_1
            prob_margin_1+=prob_1/K
            prob_margin_0+=prob_0/K
        for k in range(K):
            prob_1=2*(1-eta_prob_vec[k])*(expert_prob_vec[i]-1/2)*(anno_prob_vector[i]-1/2)+1/2
            prob_0=1-prob_1
            MI+=prob_1*np.log(np.clip(prob_1/prob_margin_1,1e-10,1e10))+prob_0*np.log(np.clip(prob_0/prob_margin_0,1e-10,1e10))
    MI=MI/(N*K)
    print(MI)
    return MI

def MI_population(calibrate_probs,expert_mode='random',calulate='max'):
    if expert_mode=='random':
        #expert is random with labeling
        expert_prob_vec=calibrate_probs
        anno_prob_vector=calibrate_probs
    elif expert_mode=='noisy':
        #expert has little bit diffferent preference from the annotator
        expert_prob_vec=calibrate_probs+np.random.normal(0,0.1,len(calibrate_probs))
        #truncate the expert_prob_vec to 0.0001 and 0.9999 to avoid log(0)
        expert_prob_vec=np.clip(expert_prob_vec,0.0001,0.9999)
        anno_prob_vector=calibrate_probs
    else:
        expert_prob_vec=np.ones(len(calibrate_probs))*expert_mode
        anno_prob_vector=calibrate_probs
    if calulate=='max':
        MI=Mul_inf_max(expert_prob_vec,anno_prob_vector)

    else:
        #eta_prob_vec is uniform seperated in [0,1],i.e., [0,0.01,0.02,...,0.99,1] as benchmark
        eta_prob_vec=np.arange(0,1.01,0.01)

        MI=Mul_inf_compute(expert_prob_vec,anno_prob_vector,eta_prob_vec)
    
    return MI
def MI_n_sample_plot(calibrate_probs,epsilons=[0.1],expert_mode='random',num_samples=[10,100,1000,10000],name=''):
    
    plt.figure(figsize=(10,10))
    num=len(epsilons)
    color_list=plt.cm.Blues(np.linspace(0.5, 0.9, num))
    MI=MI_population(calibrate_probs,expert_mode=expert_mode)
    print('name',name)
    print('expert_mode',expert_mode)
    print('MI',MI)
    print('________')
    for i,epsilon in enumerate(epsilons):
        LBs=[]
        
        for n in num_samples:
            LB=1-(n*MI+np.log(2))/(1/np.sqrt(2*epsilon)+1)
            LB=max(0,LB)
            LBs.append(LB)
        #keep 2 decimal places for 1-p

        plt.plot(num_samples,LBs,label=f"$\epsilon$={epsilon:.2f}",markersize=10,linewidth=4,color=color_list[i])
    #log scale of x axis

    plt.xlabel("$n$")
    plt.ylabel("Lower Bound of $\mathbb{P}(\|\hat{\eta}(\mathbf{C})-\eta\|_2\geq \epsilon)$")
    plt.grid()
    if name=='sky':
        plt.legend()
    os.makedirs("figs", exist_ok=True)
    sv_name=name
    sv_name+='mode_'+str(expert_mode)
    sv_name+='_MI.eps'
    
    plt.savefig("figs/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure

def MI_n_sample_plot_set(calibrate_probs,epsilons=[0.1],num_samples=[10,100,1000,10000],name=''):
    plt.figure(figsize=(10,10))
    num=len(epsilons)
    color_list=plt.cm.Blues(np.linspace(0.5, 0.9, num))
    LBs=[]
    for expert_mode in [1,0.8]:
        MI=MI_population(calibrate_probs,expert_mode=expert_mode)
        print('name',name)
        print('expert_mode',expert_mode)
        print('MI',MI)
        for i,epsilon in enumerate(epsilons):
            for n in num_samples:
                LB=1-(n*MI+np.log(2))/(1/np.sqrt(2*epsilon)+1)
                LB=max(0,LB)
                LBs.append(LB)
    LBs=np.array(LBs).reshape(2,len(epsilons),len(num_samples))

    plt.figure(figsize=(10,10))
    mode_line=['-','--']
    #color_list the same length as r_list wtih color bar of red 
    num=len(epsilons)
    color_list = plt.cm.Blues(np.linspace(0.5, 0.9, num))
    mode_name=['$p_e=1$','$p_e=0.8$']

    for i,mode in enumerate([1,0.8]):
        for j,epsilon in enumerate(epsilons):
            plt.plot(num_samples,LBs[i,j,:],markersize=10,linewidth=4,color=color_list[j],linestyle=mode_line[i])

    plt.xlabel("$n$")
    plt.ylabel("Lower Bound of $\mathbb{P}(\|\hat{\eta}(\mathbf{C})-\eta\|_2\geq \epsilon)$")
    plt.grid()
  
    if name=='sky':
        r_legend_handles = []
        for j, r in enumerate(epsilons):
            # "Dummy" line2D just for the legend
            line = mlines.Line2D(
                [], [], 
                color=color_list[j],
                linestyle='-',   # style doesn’t matter here, pick any
                label=f"$\epsilon =$ {r}",
                markersize=10,
                linewidth=5

            )
            r_legend_handles.append(line)

        legend_r = plt.legend(
            handles=r_legend_handles,
            title='$\epsilon$',
            loc='upper right'
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
            title='$p_e$',
            loc='upper left'
        )
        
        plt.gca().add_artist(legend_r)
        plt.gca().add_artist(legend_mode)

    os.makedirs("figs", exist_ok=True)
    sv_name=name
    sv_name+='_MI.eps'
    plt.savefig("figs/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure


def KL_population(calibrate_probs,flip_prob_1=0,flip_prob_2=0,expert_mode='random'):
    if expert_mode=='random':
        #expert is random with labeling
        expert_prob_vec=calibrate_probs
        anno_prob_vector=calibrate_probs
    elif expert_mode=='noisy':
        #expert has little bit diffferent preference from the annotator
        expert_prob_vec=calibrate_probs+np.random.normal(0,0.1,len(calibrate_probs))
        #truncate the expert_prob_vec to 0.0001 and 0.9999 to avoid log(0)
        expert_prob_vec=np.clip(expert_prob_vec,0.0001,0.9999)
        anno_prob_vector=calibrate_probs
    else:
        expert_prob_vec=np.ones(len(calibrate_probs))*expert_mode
        anno_prob_vector=calibrate_probs

    KL=KL_bernoulli(expert_prob_vec,anno_prob_vector,flip_prob_1,flip_prob_2)
    return np.mean(KL)

def LB_n_sample_plot(calibrate_probs,flip_prob=[0],flip_prob_2=0,expert_mode='random',num_samples=[10,100,1000,10000],name=''):
    
    plt.figure(figsize=(10,10))
    num=len(flip_prob)
    color_list=plt.cm.Blues(np.linspace(0.5, 0.9, num))
    for i,p in enumerate(flip_prob):
        LBs=[]
        KL=KL_population(calibrate_probs,flip_prob_1=p,flip_prob_2=flip_prob_2,expert_mode=expert_mode)
        for n in num_samples:
            LB_1=1-np.sqrt(n/2*KL)
            LB_2=1-np.sqrt(1-np.exp(-n*KL))
            LB=max(0,LB_1,LB_2)
            LBs.append(LB)
        #keep 2 decimal places for 1-p

        plt.plot(num_samples,LBs,label=f"$\eta_0$={1-p:.2f}",markersize=10,linewidth=4,color=color_list[i])
    #log scale of x axis

    plt.xlabel("$n$")
    plt.ylabel("Lower Bound of Error Sum")
    plt.grid()
    if name=='sky':
        plt.legend()
    os.makedirs("fig", exist_ok=True)
    sv_name=name
    sv_name+='eta_2_'+str(flip_prob_2)+'_'
    sv_name+='mode_'+str(expert_mode)+'_'
    sv_name+='_LB.eps'
    plt.savefig("figs/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure
    return LBs





def monte_carlo_bernoulli_LRT(p0, p1, N, num_sim=100000, threshold=1.0):
 
    
    data_H0 = np.random.binomial(N, p0, size=num_sim)
    
    ratio_H0 = ((p1**data_H0) * ((1 - p1)**(N - data_H0))) / \
               ((p0**data_H0) * ((1 - p0)**(N - data_H0)))
    

    reject_H0_under_H0 = (ratio_H0 >= threshold)
    
   
    type_I_error = np.mean(reject_H0_under_H0)
    

    data_H1 = np.random.binomial(N, p1, size=num_sim)
    
    ratio_H1 = ((p1**data_H1) * ((1 - p1)**(N - data_H1))) / \
               ((p0**data_H1) * ((1 - p0)**(N - data_H1)))
    
    fail_to_reject_H0_under_H1 = (ratio_H1 < threshold)
    
    # Empirical Type II error (Probability of miss)
    type_II_error = np.mean(fail_to_reject_H0_under_H1)
    
    return type_I_error+type_II_error

def LRT_plot(eta,eta_2, num_samples_list,delta_list,threshold=1.0,benchmark=None,name=''):
    plt.figure(figsize=(10,10))
    color_list=plt.cm.Blues(np.linspace(0.5, 0.9, len(delta_list)))
    for i,delta in enumerate(delta_list):
        power_list=[]
        for n in num_samples_list:
            p_0=0.5*(1-delta)*eta+0.5
            p_1=0.5*(1-delta)*eta_2+0.5
            power=monte_carlo_bernoulli_LRT(p_0,p_1,n,threshold=threshold)
            power_list.append(power)
        if benchmark is not None:
            plt.plot(num_samples_list,power_list,label=f"$Self, \delta$={delta:.2f}",markersize=10,linewidth=4,color=color_list[i])
        else:
            plt.plot(num_samples_list,power_list,label=f"$\delta$={delta:.2f}",markersize=10,linewidth=4,color=color_list[i])
    if benchmark is not None:
        plt.plot(num_samples_list,benchmark,label='Expert Lower Bound',markersize=10,linewidth=4,color='red',linestyle='--')
    plt.xlabel("$n$")
    plt.ylabel("Upper/Lower Bound of Error Sum")
    plt.grid()
    plt.legend()
    os.makedirs("figs", exist_ok=True)
    sv_name='LRT_'
    sv_name+=name
    sv_name+='eta'+str(eta)+'_'
    sv_name+='thr'+str(threshold)+'_'
    sv_name+='eta2'+str(eta_2)+'_'
    sv_name+='_LB.eps'
    plt.savefig("figs/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure






def main():
    # change default style figure and font size
    
    plt.rcParams['font.size'] = 30
    n_bins = 30
    for name in ['PKU','sky','Helpsteer','Ultra']:
 
        train_dataset,eval_dataset=get_data(script_config_path='/home/zhongzec24/RewardModeling/paper_experiment_configs/llama-'+name+'.json')
        
        
        if name in ['PKU','sky']:    
            histogram_probs=calibrate_data(train_dataset,eval_dataset,n_bins=n_bins)
            calibrate_plot(eval_dataset,histogram_probs,n_bins=n_bins,name=name)
    
            histogram_plot(histogram_probs,name=name)
        else:
            histogram_probs=np.array(eval_dataset['preference_score'])
            preference_score = eval_dataset['preference_score']
            histogram_plot(preference_score,name=name)



        num_samples=np.arange(10,500,10)
        flip_probs=[0.05,0.1,0.2,0.4]

        flip_probs_2=[0,0.02,0.04]
        expert_mode=[1]
        for p2 in flip_probs_2:
            for mode in expert_mode:
                    LB=LB_n_sample_plot(histogram_probs,flip_prob=flip_probs,flip_prob_2=p2,expert_mode=mode,num_samples=num_samples,name=name)

        epsilons=[0.05,0.1,0.2]
        expert_mode=['random','noisy']
        for mode in expert_mode:
            
            MI_n_sample_plot(histogram_probs,epsilons,expert_mode=mode,num_samples=num_samples,name=name)

        MI_n_sample_plot_set(histogram_probs,epsilons,num_samples,name=name)

        if name in ['PKU','Helpsteer']:
            eta_list=[0.9,0.8]
            eta_2_list=[1,0.95]
            delta_list=[0.0,0.1,0.2,0.3]
            num_samples_list=np.arange(10,500,10)

            for eta in eta_list:
                for eta_2 in eta_2_list:
                    LB=LB_n_sample_plot(histogram_probs,flip_prob=[1-eta],flip_prob_2=(1-eta_2),expert_mode=[1],num_samples=num_samples_list,name='')
                    LRT_plot(eta,eta_2,num_samples_list,delta_list,threshold=1.0,benchmark=LB,name=name)


if __name__ == "__main__":
    main()

