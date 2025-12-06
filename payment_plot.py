from utils.simulator import DecisionModel, payment_function
from functools import partial

def mean_func(data_batch):
        # data_batch.mean(axis=1) => shape (num_experiments,)
        return data_batch.mean(axis=1)

def power_func(power,N,c,r):
    return c*(r*N)**power




def main():
    S_1_candidates=[10,20,30,40,50]
    R_candidates=[0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    r_candidates=[0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    
    N=10000 #sample size
    mu=partial(power_func,1/2,N,1)
    T_func=mean_func
    E_func=partial(power_func,1/3,N,1)
    test_sample_num_list=[100,500,1000]
    agent_px_error=0
    principal_px_error=0
    principal_E_estimate=E_func

    for data_name in ['PKU','sky','Helpsteer','Ultra']:
        for monitor_mode in ['self','expert']:
            for num_test_samples in test_sample_num_list:
                pri_S_1_sol,pri_R_sol,exp_r,age_r_sol,agent_payoff,principal_payoff,r_diff,pri_payoff_diff,age_payoff_diff=payment_function(
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
                num_experiments=10000)
                print(f"Data Name: {data_name}, Monitor Mode: {monitor_mode}, Num Test Samples: {num_test_samples}")
                print(f"Principal S_1: {pri_S_1_sol}, Principal R: {pri_R_sol}, Expert r: {exp_r}, Agent r: {age_r_sol}")
                print(f"Agent Payoff: {agent_payoff}, Principal Payoff: {principal_payoff}, r_diff: {r_diff}")
                print(f"Principal Payoff Diff: {pri_payoff_diff}, Agent Payoff Diff: {age_payoff_diff}")
                print("===============================================")


if __name__ == "__main__":
    main()