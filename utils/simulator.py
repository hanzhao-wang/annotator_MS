from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import *

import os
import matplotlib.pyplot as plt
import click
import numpy as np
import torch
import wandb
import pandas as pd
from numpy.random import default_rng
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
)



from calibration_module.utils import compute_binary_score,compute_calibration_summary
from calibration_module.calibrator import HistogramCalibrator


def calibrate_data(train_dataset,eval_dataset,n_bins=20):
    preference_score = np.array(train_dataset['preference_score'])
    labels=np.array(train_dataset['label'])
    histogram = HistogramCalibrator(n_bins=n_bins)
    histogram.fit(preference_score, labels)
    histogram_probs = histogram.predict(np.array(eval_dataset['preference_score']))
    return histogram_probs


class DecisionModel:
    def __init__(self,
                monitor_mode,
                original_preference,
                T_func,
                num_experiments=10000):
        self.mode=monitor_mode
        self.original_preference=original_preference
        self.T_func=T_func
        self.num_experiments=num_experiments # number of experiments to simulate rejection probability
    


    def agent_data_simulator(self,r,error=0):
        length=len(self.original_preference)

        
        if self.mode=='self':
            mean_values=(1+r**2)/2
        else:
            #check if all original_preference>1/2
            if not np.all(self.original_preference>=1/2):
                raise ValueError("All original preference should be greater than 1/2")
            #add error
            original_preference=self.original_preference+error
            #clip above 0.5
            original_preference=np.clip(original_preference,0.5,1)

            
            mean_values=original_preference*r+(1-r)/2

        
        #generate bernoulli samples
        samples=np.random.binomial(1,mean_values,length)
        return samples

    def probability_simulator(
        self, 
        agent_samples, 
        R_interval,
        num_test_samples=100,
    ):
        """
        Estimate P[T_star(D) ∈ R], vectorized over 'num_experiments'.
        """
        D_samples = np.random.choice(agent_samples, size=(self.num_experiments, num_test_samples), replace=True)

        t_values = self.T_func(D_samples)
        R_values = R_interval

   
        # membership is a boolean array of shape (num_experiments,)
        membership = (t_values >= R_values[:, 0]) & (t_values <= R_values[:, 1])

        # Probability estimate is average of membership
        prob_est = membership.mean()
        return 1-prob_est

    def agent_payoff(self,
        S_1,
        function_E,
        r,
        R_interval,
        num_test_samples,
        p_x_error=0):

        agent_data=self.agent_data_simulator(r,error=p_x_error)
        p_est = self.probability_simulator(
                agent_data,
                R_interval,
                num_test_samples)
        # Agent's objective
        obj_val = p_est * S_1 - function_E(r)
        return obj_val

    def principal_payoff(
        self,
        mu,
        function_E,  # the E function that the agent uses
        S_1,
        R_up,
        r_candidates,
        num_test_samples,
        p_x_error=0 # the error of p_x that agent uses
    ):
        # shape of R_intervals is (self.num_experiments,2) with each row being [0,R_up]
        R_intervals = np.zeros((self.num_experiments, 2))
        R_intervals[:, 1] = R_up

        # Agent picks r to maximize p(r)*S_1 - E(r) with the error of p_x
        r_star, agent_pay = self.agent_solve(
            S_1,
            function_E,
            r_candidates,
            p_x_error,
            R_intervals,
            num_test_samples,
        )

        # Principal's payoff: mu(r_star) - p(r_star)*S_1
        agent_data = self.agent_data_simulator(r_star, error=0)  
        p_est = self.probability_simulator(
            agent_data,
            R_intervals,
            num_test_samples,
        )
        principal_payoff = mu(r_star) - p_est * S_1
        return principal_payoff




    # ----------------------------------------------------------------------
    # Agent Solve:
    #    r^*(S_1, function_E) = argmax_r [ P(T(D, r) \notin R) * S_1 - function_E(r) ]
    # ----------------------------------------------------------------------



    def agent_solve(
        self,
        S_1,
        function_E,
        r_candidates,
        p_x_error,
        R_interval,
        num_test_samples,
        
    ):
        """
        Simple line-search over candidate r values, uses vectorized
        probability_simulator for each r.
        """
        best_r = None
        best_obj = -np.inf

        for r in r_candidates:

            obj_val = self.agent_payoff(
                S_1,
                function_E,
                r,
                R_interval,
                num_test_samples,
                p_x_error,
            )
            if obj_val > best_obj:
                best_obj = obj_val
                best_r = r

        return best_r, best_obj

    # ----------------------------------------------------------------------
    # Principal Solve:
    #    max_{S_1,R} [ mu(r^*) - P(T_star(D(r^*)) \notin R) * S_1 ]
    # ----------------------------------------------------------------------
    def principle_solve(
        self,
        mu,
        function_E, #the E function that principal believes the agent uses
        S_1_candidates,
        R_candidates,
        r_candidates,
        num_test_samples,
        p_x_error=0 #the error of p_x that principal uses
    ):

        best_S_1 = None
        best_r_star = None
        best_R=None
        best_obj = -np.inf
        for R_up in R_candidates:
            #shape of R_intervals is (num_experiments,2) with each row being [0,R_up]
            R_intervals=np.zeros((self.num_experiments,2))
            R_intervals[:,1]=R_up
            for S_1 in S_1_candidates:

                # Agent picks r to maximize p(r)*S_1 - E(r)
                r_star, agent_pay = self.agent_solve(
                        S_1,
                        function_E,
                        r_candidates,
                        p_x_error,
                        R_intervals,
                        num_test_samples,
                )

                # Principal's payoff: mu(r_star) - p(r_star)*S_1
                agent_data=self.agent_data_simulator(r_star,error=p_x_error)
                p_est = self.probability_simulator(
                    agent_data,
                    R_intervals,
                    num_test_samples)
                principal_payoff = mu(r_star) - p_est * S_1

                if principal_payoff > best_obj:
                    best_obj = principal_payoff
                    best_S_1 = S_1
                    best_r_star = r_star
                    best_R=R_up
                    agent_payoff = agent_pay

        return best_S_1, best_R, best_obj,best_r_star, agent_payoff





def payment_function(
                data_name,
                monitor_mode,
                mu,
                T_func,
                E_func,
                agent_px_error,
                principal_px_error,
                principal_E_estimate,
                S_1_candidates,
                R_candidates,
                r_candidates,
                num_test_samples,
                num_experiments=10000):


    train_dataset,eval_dataset=get_data(script_config_path='/home/zhongzec24/RewardModeling/paper_experiment_configs/llama-'+data_name+'.json')
    
    if data_name=='sky':    
        histogram_probs=calibrate_data(train_dataset,eval_dataset,n_bins=n_bins)
        #for elements in histogram_probs, if it is less than 0.5, set them to 1-histogram_probs
        
    else:
        histogram_probs=np.array(eval_dataset['preference_score'])
    original_preference=np.where(histogram_probs<0.5,1-histogram_probs,histogram_probs)


    decision_model=DecisionModel(monitor_mode,original_preference,T_func,num_experiments)
    pri_S_1_sol,pri_R_sol,pri_obj_sol,exp_r,_=decision_model.principle_solve(
        mu,
        principal_E_estimate, #the E function that principal believes the agent uses
        S_1_candidates,
        R_candidates,
        r_candidates,
        num_test_samples,
        principal_px_error, #the error of p_x that principal uses
    )

    R_intervals = np.zeros((num_experiments, 2))
    R_intervals[:, 1] = pri_R_sol

    age_r_sol,age_obj_sol=decision_model.agent_solve(
        pri_S_1_sol,
        E_func,
        r_candidates,
        agent_px_error,
        R_intervals,
        num_test_samples,
    )



    #the true payoffs
    agent_payoff=decision_model.agent_payoff(
        pri_S_1_sol,
        E_func,
        age_r_sol,
        R_intervals,
        num_test_samples,
    )
    principal_payoff=decision_model.principal_payoff(
        mu,
        E_func,
        pri_S_1_sol,
        pri_R_sol,
        r_candidates,
        num_test_samples,
        agent_px_error,
    )
    #the different between the principal's r_star and the agent's r_star
    r_diff=exp_r-age_r_sol
    #the different from the true payoffs and the estimated payoffs
    pri_payoff_diff=pri_obj_sol-principal_payoff
    age_payoff_diff=age_obj_sol-agent_payoff
    return pri_S_1_sol,pri_R_sol,exp_r,age_r_sol,agent_payoff,principal_payoff,r_diff,pri_payoff_diff,age_payoff_diff