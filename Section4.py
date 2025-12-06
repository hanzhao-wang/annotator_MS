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
from numpy.random import default_rng
from accelerate import Accelerator
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import binom
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

def calibrate_data(train_dataset,eval_dataset,n_bins=20):
    preference_score = np.array(train_dataset['preference_score'])
    labels=np.array(train_dataset['label'])
    histogram = HistogramCalibrator(n_bins=n_bins)
    histogram.fit(preference_score, labels)
    histogram_probs = histogram.predict(np.array(eval_dataset['preference_score']))
    return histogram_probs


class solver:


    def __init__(self,
            preference_scores,
            effort_space,
            contract_space,
            G_func,
            E_func,
            mu_func,
            n,
            U_0=0,
            delta=0,
            contract_type='linear',
            monitor_type='self',
            simulation_num=5000):
        
        self.original_preference_scores =np.array(preference_scores)
        #check self.original_preference_scores>0.5
        self.preference_scores=self.original_preference_scores
        self.preference_scores[self.original_preference_scores<0.5]=1-self.original_preference_scores[self.original_preference_scores<0.5]
        self.original_preference_scores=self.preference_scores
        #check all >0.5
        if not np.all(self.original_preference_scores>=0.5):
            raise ValueError('preference_scores must be >=0.5')
    

        self.effort_space = effort_space
        self.contract_space = contract_space
        self.G_func = G_func
        self.E_func = E_func
        self.mu_func = mu_func
        self.contract_type = contract_type
        self.monitor_type = monitor_type
        self.simulation_num = simulation_num
        self.delta=delta
        self.n = n
        self.U_0 = U_0

    def Util_compute_exact(self,exp_para):

        """
        Returns a 4D array of shape:
        [len(effort_space), len(c0_values), len(c1_values), len(c2_values)]
        """

        # effort_space is 1D: [effort_1, effort_2, ..., effort_E]
        # contract_space[0] is c0_values (e.g., thresholds)
        # contract_space[1] is c1_values (e.g., variable payment)
        # contract_space[2] is c2_values (e.g., base payment)
        func_G=lambda x:exp_para-exp_para*np.exp(-exp_para*x)

        c0_values, c1_values, c2_values = self.contract_space
        E = len(self.effort_space)
        C0 = len(c0_values)
        C1 = len(c1_values)
        C2 = len(c2_values)

        c0_grid, c1_grid, c2_grid = np.meshgrid(
            c0_values, c1_values, c2_values, indexing="ij"
        )  # shape (C0, C1, C2)

        # Expand these to shape (1, C0, C1, C2) so they broadcast across
        # simulation_num and E.
        c0_grid_4d = c0_grid[np.newaxis, ...]  # shape (1,C0,C1,C2)
        c1_grid_4d = c1_grid[np.newaxis, ...]  # shape (1,C0,C1,C2)
        c2_grid_4d = c2_grid[np.newaxis, ...]  # shape (1,C0,C1,C2)
    
        # 1) Create array of all possible efforts
        efforts = np.array(self.effort_space)
        if self.monitor_type == "self":
            mean_values = (1 + efforts*(1 - self.delta)) / 2  # shape (E,)
        elif self.monitor_type == "expert":
            mean_values =efforts * np.mean(self.original_preference_scores-0.5) + 0.5

        if self.contract_type == "linear":
            mean_values_4d = mean_values[ :, np.newaxis, np.newaxis, np.newaxis]
            payments=mean_values_4d*c1_grid_4d+c2_grid_4d
            G_payments=exp_para-exp_para*np.exp(-exp_para*c2_grid_4d)*(1-mean_values_4d+mean_values_4d*np.exp(-exp_para*c1_grid_4d/self.n))**self.n
        elif self.contract_type == "binary":
            
            survival_2d = binom.sf(np.array(c0_values)[np.newaxis, :]*self.n, self.n , mean_values[:, np.newaxis])
            probs = survival_2d[:, :, np.newaxis, np.newaxis]  # shape (E, C0, 1, 1)
            probs = np.broadcast_to(probs, (E, C0, C1, C2))
            payments=probs*c1_grid_4d+c2_grid_4d
            G_payments = probs *  func_G(c1_grid_4d+ c2_grid_4d)+(1-probs)*func_G(c2_grid_4d)  # shape (E, C0, C1, C2)

        cost_efforts = np.array([self.E_func(e) for e in efforts])
        # Expand to (E,1,1,1) so it matches (E, C0, C1, C2)
        cost_efforts_4d = cost_efforts[:, np.newaxis, np.newaxis, np.newaxis]

        # 8) Final utility => (E, C0, C1, C2)
        agent_utility = G_payments - cost_efforts_4d
          
        mu_efforts=np.array([self.mu_func(e) for e in efforts]) 
        mu_efforts_4d = mu_efforts[:, np.newaxis, np.newaxis, np.newaxis]

        principal_utility = -payments + mu_efforts_4d

        self.agent_utility = agent_utility
        self.principal_utility = principal_utility
        return agent_utility, principal_utility

    

    def Util_compute(self):
        """
        Returns a 4D array of shape:
        [len(effort_space), len(c0_values), len(c1_values), len(c2_values)]
        """

        # effort_space is 1D: [effort_1, effort_2, ..., effort_E]
        # contract_space[0] is c0_values (e.g., thresholds)
        # contract_space[1] is c1_values (e.g., variable payment)
        # contract_space[2] is c2_values (e.g., base payment)

        c0_values, c1_values, c2_values = self.contract_space
        E = len(self.effort_space)
        C0 = len(c0_values)
        C1 = len(c1_values)
        C2 = len(c2_values)

        # 1) Create array of all possible efforts
        efforts = np.array(self.effort_space)  # shape (E,)

        # 2) Compute the mean_value for each effort (shape (E,))
        #    depending on self.monitor_type:
        if self.monitor_type == "self":
            # => mean_value is scalar function of effort
            mean_values = (1 + efforts*(1 - self.delta)) / 2  # shape (E,)
        elif self.monitor_type == "expert":
            # => typically average across self.original_preference_scores
            #    so again we end up with a single mean per effort
            #    (assuming .mean(...) yields scalar for each effort)
            #    shape (E,) after axis=1 if original_preference_scores is 1D
         
            mean_values =efforts * np.mean(self.original_preference_scores-0.5) + 0.5

        # 3) Draw binomial samples for *each* (effort, c0, c1, c2).
        #    That means shape = (simulation_num, E, C0, C1, C2).
        #    We just need to broadcast mean_values into that shape.
        #
        #    np.random.binomial can broadcast `p` if `p.shape` matches the size
        #    argument except for the dimensions that are 1 or identical.
        #    We'll expand mean_values to (E, 1, 1, 1) so it can broadcast
        #    alongside (C0, C1, C2).
        mean_values_5d = mean_values[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        # We want final size = (simulation_num, E, C0, C1, C2).
        samples = np.random.binomial(
            n=self.n,
            p=mean_values_5d,  # shape (1, E, 1, 1, 1) -> broadcast to (simulation_num, E, C0, C1, C2)
            size=(self.simulation_num, E, C0, C1, C2)
        )

        # 4) Create grids for (c0, c1, c2).  We'll eventually want to broadcast them
        #    into shape (simulation_num, E, C0, C1, C2). The typical approach:
        c0_grid, c1_grid, c2_grid = np.meshgrid(
            c0_values, c1_values, c2_values, indexing="ij"
        )  # shape (C0, C1, C2)

        # Expand these to shape (1, 1, C0, C1, C2) so they broadcast across
        # simulation_num and E.
        c0_grid_5d = c0_grid[np.newaxis, np.newaxis, ...]  # shape (1,1,C0,C1,C2)
        c1_grid_5d = c1_grid[np.newaxis, np.newaxis, ...]  # shape (1,1,C0,C1,C2)
        c2_grid_5d = c2_grid[np.newaxis, np.newaxis, ...]  # shape (1,1,C0,C1,C2)

        # 5) Compute the payment for each draw
        #    => shape (simulation_num, E, C0, C1, C2)
        if self.contract_type == "linear":
            # payments = c1_grid * (samples / n) + c2_grid
            payments = (c1_grid_5d * samples / self.n) + c2_grid_5d
        elif self.contract_type == "binary":
            # payments = 1{(samples / n) >= c0_grid} * c1_grid + c2_grid
            payments = ((samples / self.n) >= c0_grid_5d) * c1_grid_5d + c2_grid_5d

        # 6) Apply G_func to payments, then average over simulation_num
        #    => shape after G_func is (simulation_num, E, C0, C1, C2)
        G_of_payments = self.G_func(payments)
        # => shape (E, C0, C1, C2) after averaging out the simulation dimension
        mean_G = np.mean(G_of_payments, axis=0)

        # 7) Subtract cost of effort.  If self.E_func can take a vector of efforts,
        #    you can do: cost_efforts = self.E_func(efforts).  Otherwise do a loop.
        #    shape = (E,).
        cost_efforts = np.array([self.E_func(e) for e in efforts])
        # Expand to (E,1,1,1) so it matches (E, C0, C1, C2)
        cost_efforts_4d = cost_efforts[:, np.newaxis, np.newaxis, np.newaxis]

        # 8) Final utility => (E, C0, C1, C2)
        agent_utility = mean_G - cost_efforts_4d
          
        mu_efforts=np.array([self.mu_func(e) for e in efforts]) 
        mu_efforts_4d = mu_efforts[:, np.newaxis, np.newaxis, np.newaxis]

        principal_utility = -np.mean(payments, axis=0) + mu_efforts_4d

        self.agent_utility = agent_utility
        self.principal_utility = principal_utility
        return agent_utility, principal_utility


    def FB_solve(self):
        

        agent_utility, principal_utility = self.agent_utility,self.principal_utility
        # Unpack sizes
        E = agent_utility.shape[0]
        C0 = agent_utility.shape[1]
        C1 = agent_utility.shape[2]
        C2 = agent_utility.shape[3]

        agent_util_e=  agent_utility  # shape (E, C0, C1, C2)
        # check that agent util >= U_0
        above_reservation = (agent_util_e >=self.U_0)  # shape (E, C0, C1, C2)

        feasible_mask =  above_reservation

        # 3) If there is no feasible combination, return (None, 0, 0, 0).
        if not np.any(feasible_mask):
            return (None, 0, 0, 0), self.mu_func(0),0,self.U_0

        # 4) Among all feasible (e, c0, c1, c2), pick one that maximizes principal_utility[e, c0, c1, c2].
        feasible_principal_util = principal_utility[feasible_mask]  # 1D array of principal utilities
        
        #get all indeces with max utilities from the feasible indices
        best_index_in_feasible = np.argmax(feasible_principal_util)




        # We need the actual (e, c0, c1, c2) indices for that entry.
        feasible_indices = np.where(feasible_mask)  # each is a 1D array of the same length
        best_e_idx = feasible_indices[0][best_index_in_feasible]
        best_c0_idx = feasible_indices[1][best_index_in_feasible]
        best_c1_idx = feasible_indices[2][best_index_in_feasible]
        best_c2_idx = feasible_indices[3][best_index_in_feasible]

        c0_values, c1_values, c2_values = self.contract_space
        best_c0 = c0_values[best_c0_idx]
        best_c1 = c1_values[best_c1_idx]
        best_c2 = c2_values[best_c2_idx]
        return (best_c0, best_c1, best_c2), principal_utility[best_e_idx, best_c0_idx, best_c1_idx, best_c2_idx], self.effort_space[best_e_idx], agent_utility[best_e_idx, best_c0_idx, best_c1_idx, best_c2_idx]

    def SB_solve(self):
        

        agent_utility, principal_utility = self.agent_utility,self.principal_utility
        # Unpack sizes
        E = agent_utility.shape[0]
        C0 = agent_utility.shape[1]
        C1 = agent_utility.shape[2]
        C2 = agent_utility.shape[3]

        # 1) For each (c0, c1, c2), compute the max agent utility across all e
        #    => shape (C0, C1, C2).
        max_util_over_e = agent_utility.max(axis=0)

        # 2) The condition "e is a best response" means
        #    agent_utility[e, c0, c1, c2] == max_util_over_e[c0, c1, c2].
        #    Also we need agent_utility[e, c0, c1, c2] >= U_0.
        #
        #    So define a boolean mask "feasible" of shape (E, C0, C1, C2).
        #    feasible[e, c0, c1, c2] = True if e is among best responses
        #    AND the participation constraint is satisfied.
        #
        #    Then we look for the one among these that yields the maximum principal_utility.
        #
        agent_util_e = agent_utility  # shape (E, C0, C1, C2)

        # We broadcast max_util_over_e to shape (E, C0, C1, C2) by indexing:
        #   max_util_over_e[None, :, :, :]
        # or by comparing each e’s slice to max_util_over_e.
        is_best_response = (agent_util_e+0.01>= max_util_over_e[np.newaxis, ...])  # shape (E, C0, C1, C2), allow a little bit of numerical error

        # Also check that agent util >= U_0
        above_reservation = (agent_util_e >=self.U_0)  # shape (E, C0, C1, C2)

        feasible_mask = is_best_response & above_reservation

        # 3) If there is no feasible combination, return (None, 0, 0, 0).
        if not np.any(feasible_mask):
            return (None, 0, 0, 0), self.mu_func(0),0,self.U_0

        # 4) Among all feasible (e, c0, c1, c2), pick one that maximizes principal_utility[e, c0, c1, c2].
        feasible_principal_util = principal_utility[feasible_mask]  # 1D array of principal utilities

        # Argmax over the feasible set
        best_index_in_feasible = np.argmax(feasible_principal_util)

        # We need the actual (e, c0, c1, c2) indices for that entry.
        feasible_indices = np.where(feasible_mask)  # each is a 1D array of the same length
        best_e_idx = feasible_indices[0][best_index_in_feasible]
        best_c0_idx = feasible_indices[1][best_index_in_feasible]
        best_c1_idx = feasible_indices[2][best_index_in_feasible]
        best_c2_idx = feasible_indices[3][best_index_in_feasible]
        c0_values, c1_values, c2_values = self.contract_space
        best_c0 = c0_values[best_c0_idx]
        best_c1 = c1_values[best_c1_idx]
        best_c2 = c2_values[best_c2_idx]
        return (best_c0, best_c1, best_c2), principal_utility[best_e_idx, best_c0_idx, best_c1_idx, best_c2_idx], self.effort_space[best_e_idx], agent_utility[best_e_idx, best_c0_idx, best_c1_idx, best_c2_idx]
    
    def SB_solve_tilde(self,e_star):
        agent_utility, principal_utility = self.agent_utility,self.principal_utility
        try:
            e_star_index = self.effort_space.index(e_star)  # if effort_space is a list
        except ValueError:
            # e_star not in the effort_space
            return (0, 0, 0),self.mu_func(0),0,self.U_0

 
        e_star_index = int(e_star_index)

        # --------------------------------------------------------------------------
        # 2) The agent's best response for a given (c0, c1, c2) is the
        #    effort that yields the maximum agent_utility among all efforts.
        #
        #    We can find that best response by taking argmax over the
        #    E dimension of agent_utility.
        #
        #    best_effort_map[c0, c1, c2] = e_idx in [0..E-1]
        # --------------------------------------------------------------------------
        best_effort_map = np.argmax(agent_utility, axis=0)  
        # shape: (C0, C1, C2), each entry is an integer e_idx

        # --------------------------------------------------------------------------
        # 3) For the agent to *choose* e_star, we need e_star to be one of 
        #    the maximizers. If we want e_star to be the unique best effort, 
        #    we would require best_effort_map == e_star_index. 
        #
        #    But if you only need e_star to be "among" the best (ties allowed), 
        #    we can do:
        #
        #       agent_utility[e_star_index] == agent_utility.max(axis=0).
        #
        #    We'll show the "among the best" approach:
        # --------------------------------------------------------------------------
        # The maximum utility among all efforts at each contract:
        max_util_over_efforts = agent_utility.max(axis=0)  # shape (C0, C1, C2)

       
        # => e_star, or e_star+1, or e_star-1 is among the best responses
        part_max=np.max(agent_utility[e_star_index-1: max(e_star_index+2,len(agent_utility))],axis=0)
        #part_max=np.max(agent_utility[e_star_index])
        is_e_star_best = ( part_max>= max_util_over_efforts) 



        # --------------------------------------------------------------------------
        # 4) We also need that agent_utility[e_star_index, c0, c1, c2] > U_0
        # --------------------------------------------------------------------------
        above_reservation = (agent_utility[e_star_index] >= self.U_0)

        # --------------------------------------------------------------------------
        # 5) The combined feasibility mask:
        #    (1) e_star is among the best,
        #    (2) utility at e_star > U_0
        # --------------------------------------------------------------------------
        feasible_mask = is_e_star_best & above_reservation
        # shape: (C0, C1, C2). Boolean.

        # If no (c0, c1, c2) is feasible, return (0,0,0)
        if not np.any(feasible_mask):
            print('no feasible contract')
            return (0, 0, 0),self.mu_func(0),0,self.U_0

        # --------------------------------------------------------------------------
        # 6) Among feasible ones, choose the contract that *maximizes* principal utility
        #    at e_star_index.
        # --------------------------------------------------------------------------
        # Slice the principal utility for e_star_index:
        principal_slice = principal_utility[e_star_index]  # shape (C0, C1, C2)

        # We only want to look at entries where feasible_mask==True.
        feasible_princ_util = principal_slice[feasible_mask]

        # Argmax over the feasible set:
        best_idx_in_feasible = np.argmax(feasible_princ_util)

        # Now we need to map that back to the actual (c0, c1, c2) indices:
        feasible_indices = np.where(feasible_mask)  # returns a tuple (array_of_c0, array_of_c1, array_of_c2)
        best_c0_idx = feasible_indices[0][best_idx_in_feasible]
        best_c1_idx = feasible_indices[1][best_idx_in_feasible]
        best_c2_idx = feasible_indices[2][best_idx_in_feasible]

        # --------------------------------------------------------------------------
        # 7) Finally, pick out the actual parameter values from the index:
        # --------------------------------------------------------------------------
        c0_values, c1_values, c2_values = self.contract_space
        best_c0 = c0_values[best_c0_idx]
        best_c1 = c1_values[best_c1_idx]
        best_c2 = c2_values[best_c2_idx]

        return (best_c0, best_c1, best_c2), principal_utility[e_star_index][best_c0_idx, best_c1_idx, best_c2_idx], e_star, agent_utility[e_star_index][best_c0_idx, best_c1_idx, best_c2_idx]
        


def plot_func(monitor_type_list, contract_type_list, n_list, results, name, U_0,delta, plot_mode='gap',save_name=''):
        plt.figure(figsize=(10,10))
        marker_monitor={'self':'o','expert':'s'}
        color_monitor={'self':'blue','expert':'red'}
        line_style={'linear':'-','binary':'--'}

        for monitor_type in monitor_type_list:
            for contract_type in contract_type_list:
                FB_utilities=[]
                SB_utilities=[]
                FT_effort=[]
                SB_effort=[]
                FT_agent_utilities=[]
                SB_agent_utilities=[]
                SB_tilde_utilities=[]
                SB_tilde_agent_utilities=[]
                SB_tilde_effort=[]

                for n in n_list:
                    FB_utilities.append(results[(name,monitor_type,contract_type,n,'FB')]['best_principal_util'])
                    SB_utilities.append(results[(name,monitor_type,contract_type,n,'SB')]['best_principal_util'])
                    FT_effort.append(results[(name,monitor_type,contract_type,n,'FB')]['best_effort'])
                    SB_effort.append(results[(name,monitor_type,contract_type,n,'SB')]['best_effort'])
                    FT_agent_utilities.append(results[(name,monitor_type,contract_type,n,'FB')]['agent_util'])
                    SB_agent_utilities.append(results[(name,monitor_type,contract_type,n,'SB')]['agent_util'])

                    SB_tilde_utilities.append(results[(name,monitor_type,contract_type,n,'SB_tilde')]['best_principal_util'])
                    SB_tilde_agent_utilities.append(results[(name,monitor_type,contract_type,n,'SB_tilde')]['agent_util'])
                    SB_tilde_effort.append(results[(name,monitor_type,contract_type,n,'SB_tilde')]['best_effort'])
                if plot_mode=='gap':
                    y=(np.array(FB_utilities)-np.array(SB_utilities))/np.array(FB_utilities)
                    y=np.clip(y,0,1)
                    plt.plot(n_list,y,marker=marker_monitor[monitor_type],linestyle=line_style[contract_type],markersize=10,linewidth=4,color='blue')
                    y=(np.array(FB_utilities)-np.array(SB_tilde_utilities))/np.array(FB_utilities)
                    y=np.clip(y,0,1)
                    plt.plot(n_list,y,marker=marker_monitor[monitor_type],linestyle=line_style[contract_type],markersize=10,linewidth=4,color='red')
                elif plot_mode=='effort':
                    plt.plot(n_list,np.array(SB_effort),marker=marker_monitor[monitor_type],linestyle=line_style[contract_type],markersize=10,linewidth=4,color='blue')
                    plt.plot(n_list,np.array(SB_tilde_effort),marker=marker_monitor[monitor_type],linestyle=line_style[contract_type],markersize=10,linewidth=4,color='red')
                elif plot_mode=='agent_util':
                    plt.plot(n_list,np.array(SB_agent_utilities),marker=marker_monitor[monitor_type],linestyle=line_style[contract_type],markersize=10,linewidth=4,color='blue')
                    plt.plot(n_list,np.array(SB_tilde_agent_utilities),marker=marker_monitor[monitor_type],linestyle=line_style[contract_type],markersize=10,linewidth=4,color='red')
        plt.xlabel('n')

        if plot_mode=='gap':
           
            plt.ylabel('Utility Gap')
            plt.yscale('symlog', linthresh=0.001)
            plt.ylim(0, 1.1)
            
        elif plot_mode=='effort':
            upb=np.max(FT_effort)
            plt.axhline(y=upb, color='black', linestyle='-.',linewidth=4,label='FB Effort')
            plt.ylabel('Agent Effort')
            plt.ylim(-0.1,1.1)



        elif plot_mode=='agent_util':
            plt.axhline(y=U_0, color='black', linestyle='-.',linewidth=4,label='$U_0$')
            plt.ylabel('Agent Utility')
            plt.ylim(-0.1,0.5)

        monitor_type_handle={'self':'Self','expert':'Expert'}
        monitor_legend_handles = []
        for j, r in enumerate(monitor_type_list):
            # "Dummy" line2D just for the legend
            line = mlines.Line2D(
                [], [], 
                color='blue',         # color doesn’t matter, pick a neutral color
                marker=marker_monitor[r],
                linestyle='-',   # style doesn’t matter here, pick any
                label=monitor_type_handle[r],
                markersize=16,
                linewidth=4
            )
            monitor_legend_handles.append(line)

        legend_monitor = plt.legend(
            handles= monitor_legend_handles,
            title='Monitor',
            loc='upper left'
        )
        contract_type_handle={'linear':'Linear','binary':'Binary'}
        contract_type_legend_handles = []
        for j, r in enumerate(contract_type_list):
            # "Dummy" line2D just for the legend
            line = mlines.Line2D(
                [], [], 
                color='blue',         # color doesn’t matter, pick a neutral color
                linestyle=line_style[r],   # style doesn’t matter here, pick any
                label=contract_type_handle[r],
                markersize=10,
                linewidth=4
            )
            contract_type_legend_handles.append(line)

        legend_contract_type = plt.legend(
            handles= contract_type_legend_handles,
            title='Contract',
            loc='upper right'
        )   
  

        order_type_handle={'SB':'$\mathcal{C}_n$','SB_tilde':r'$\tilde{\mathcal{C}}_n$'}
        colors={'SB':'blue','SB_tilde':'red'}
        order_type_legend_handles = []
        for j, r in enumerate(['SB','SB_tilde']):
            # "Dummy" line2D just for the legend
            line = mlines.Line2D(
                [], [], 
                color=colors[r],         # color doesn’t matter, pick a neutral color
                linestyle='-',   # style doesn’t matter here, pick any
                label=order_type_handle[r],
                markersize=10,
                linewidth=4,
                markerfacecolor='white'
            )
            order_type_legend_handles.append(line)

        legend_order_type = plt.legend(
            handles= order_type_legend_handles,
            loc='center right'
        )
        plt.gca().add_artist(legend_order_type)
        # Add the legends to the plot
     


        plt.gca().add_artist(legend_monitor)
        plt.gca().add_artist(legend_contract_type)
        
        plt.grid()
        
        #check exists
        if not os.path.exists(f'./fig_contract/{plot_mode}'):
            os.makedirs(f'./fig_contract/{plot_mode}')
        if save_name=='':
            plt.savefig(f'./fig_contract/{plot_mode}/{name}'+'delta_'+str(delta)+'U0_'+str(U_0)+'.eps',bbox_inches='tight')
        else :
            plt.savefig(f'./fig_contract/{plot_mode}/{name}'+save_name+'.eps',bbox_inches='tight')
        plt.close()

def plot_main(monitor_type_list, contract_type_list, n_list, U_0,delta, effort_space, G_func, E_func, mu_func, exp_para, save_name=''):
        
    results={}

    for name in ['PKU','sky','Helpsteer','Ultra']: #,
        if name=='PKU':
            n_list=np.arange(1,202,10).tolist()
        
        train_dataset,eval_dataset=get_data(script_config_path='/home/zhongzec24/RewardModeling/paper_experiment_configs/llama-'+name+'.json')
        
        
        if name in ['PKU','sky']:    
            histogram_probs=calibrate_data(train_dataset,eval_dataset,n_bins=30)
        else:
            histogram_probs=np.array(eval_dataset['preference_score'])

        for monitor_type in monitor_type_list:
            for contract_type in contract_type_list:
                if contract_type=='linear':
                        c0=np.arange(0,1,1).tolist()
                        c1=np.arange(0,10,0.05).tolist()
                        c2=np.arange(-10,10,0.05).tolist()
                        contract_space=[c0,c1,c2]
                elif contract_type=='binary':
                        c0=np.arange(0,1.02,0.02).tolist()
                        c1=np.arange(0,10,0.05).tolist()
                        c2=np.arange(-10,10,0.05).tolist()
                        contract_space=[c0,c1,c2]

                for n in n_list:
                    print('===============================================')
                    print(f"Data Name: {name}, Monitor Mode: {monitor_type}, Contract Type: {contract_type}, Num Test Samples: {n}")
                    solver_instance=solver(
                        preference_scores=histogram_probs,
                        effort_space=effort_space,
                        contract_space=contract_space,
                        G_func=G_func,
                        E_func=E_func,
                        mu_func=mu_func,
                        n=n,
                        U_0=U_0,
                        delta=delta,
                        contract_type=contract_type,
                        monitor_type=monitor_type
                    )
                    _,_=solver_instance.Util_compute_exact(exp_para)
                    
                    (best_c0, best_c1, best_c2), best_principal_util, best_effort, agent_util = solver_instance.FB_solve()
                    results[(name,monitor_type,contract_type,n,'FB')]={
                        'best_c0':best_c0,
                        'best_c1':best_c1,
                        'best_c2':best_c2,
                        'best_principal_util':best_principal_util,
                        'best_effort':best_effort,
                        'agent_util':agent_util
                    }
                    print(f"FB: Principal Utility: {best_principal_util}, Best Effort: {best_effort}, Agent Utility: {agent_util}")
                    print(f"FB: contract: {best_c0},{best_c1},{best_c2}")
                    e_star=best_effort
                    (best_c0, best_c1, best_c2), best_principal_util, best_effort, agent_util = solver_instance.SB_solve_tilde(e_star)
                    results[(name,monitor_type,contract_type,n,'SB_tilde')]={
                        'best_c0':best_c0,
                        'best_c1':best_c1,
                        'best_c2':best_c2,
                        'best_principal_util':best_principal_util,
                        'best_effort':best_effort,
                        'agent_util':agent_util
                    }
                    print(f"SB_tilde: Principal Utility: {best_principal_util}, Best Effort: {best_effort}, Agent Utility: {agent_util}")
                    print(f"SB_tilde: contract: {best_c0},{best_c1},{best_c2}")
                    (best_c0, best_c1, best_c2), best_principal_util, best_effort, agent_util = solver_instance.SB_solve()
                    results[(name,monitor_type,contract_type,n,'SB')]={
                        'best_c0':best_c0,
                        'best_c1':best_c1,
                        'best_c2':best_c2,
                        'best_principal_util':best_principal_util,
                        'best_effort':best_effort,
                        'agent_util':agent_util
                    }
                 
                    
                    print(f"SB: Principal Utility: {best_principal_util}, Best Effort: {best_effort}, Agent Utility: {agent_util}")
                    print(f"SB: contract: {best_c0},{best_c1},{best_c2}")
                    

       
                
        plot_func(monitor_type_list, contract_type_list, n_list, results, name, U_0,delta, plot_mode='gap',save_name=save_name)
        plot_func(monitor_type_list, contract_type_list, n_list, results, name, U_0,delta, plot_mode='effort', save_name=save_name)
        plot_func(monitor_type_list, contract_type_list, n_list, results, name, U_0,delta, plot_mode='agent_util', save_name=save_name)



def main():
    #random seed
    set_random_seed(32)
    plt.rcParams['font.size'] = 26
    

    n_list=np.arange(1,102,10).tolist()
    monitor_type_list=['expert','self']
    contract_type_list=['linear','binary']

    effort_space=np.arange(0,1.01,0.01).tolist()

    save_name=''


    G_func=lambda x:2*x**0.5
    

    E_func=lambda e:0.18*e**2
    

    mu_func_high=lambda e:0.5*(e)**0.8
    mu_func_low=lambda e:0.3*(e)**0.8
    mu_func_list={'low':mu_func_low,'high':mu_func_high}


    exp_para_list=[1,0.5]
    U_0_list=[0]

    delta_list=[0,0.02]

    for exp_para in exp_para_list:
        for mu_func_name,mu_func in mu_func_list.items():
            for U_0 in U_0_list:
                for delta in delta_list:
                    save_name=f'exp_para_{exp_para}_mu_{mu_func_name}_U0_{U_0}_delta_{delta}'
                    plot_main(monitor_type_list, contract_type_list, n_list, U_0,delta, effort_space, G_func, E_func, mu_func, exp_para, save_name=save_name)
    
    
if __name__ == "__main__":
    main()