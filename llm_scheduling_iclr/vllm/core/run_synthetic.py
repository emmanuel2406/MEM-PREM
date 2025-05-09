from collections import deque
import random

from policy import *
# from vllm.sequence import SequenceGroup
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import simpy
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from dataclasses import dataclass, field
import pandas as pd
from multiprocessing import Pool
import argparse
from simulation import Scheduler, RequestGenerator
from tqdm import tqdm


N_CPUS = 60 # tune this to your machine

def format_name(policy_class, score_params=None) -> str:
    name = policy_class.__name__
    if score_params and "type" in score_params.keys():
        name += f"({score_params['type']})"
    if score_params and "hp" in score_params.keys():
        name += f"_{score_params['hp']}"
    return name

def run_simulation(policy_class, sim_time=1000, arrival_rate=0.5, batch_size=4, job_service_distribution="realistic", length_distribution=None, token_gen_rate=4, score_params=None, seed=None, sigma=0.5):
    """Run a simulation with the given policy and parameters."""
    env = simpy.Environment()
    scheduler = Scheduler(env, policy_class, batch_size, token_gen_rate, sim_time, score_params)
    generator = RequestGenerator(env, scheduler, arrival_rate, job_service_distribution, length_distribution, seed=seed, sigma=sigma, token_gen_rate=token_gen_rate)
    # Run simulation
    env.run(until=sim_time)
    # Drain uncompleted requests
    env.run()

    # Calculate final statistics
    stats = scheduler.stats
    completed = stats["completed_requests"]

    if completed > 0:
        avg_wait = stats["total_wait_time"] / completed
        avg_response = stats["total_response_time"] / completed
        avg_jct_ratio = sum(stats["jct_ratios"]) / completed
    else:
        avg_wait = avg_response = avg_jct_ratio = 0

    # Calculate average utilization
    utilization_samples = stats["utilization_log"]
    avg_utilization = sum(util for _, util in utilization_samples) / len(utilization_samples) if utilization_samples else 0

    policy_name = format_name(policy_class, score_params)
    # if False:
    #     with open(f"llm_scheduling_iclr/vllm/dumps/{policy_name}.txt", "w") as f:
    #         for val in stats["request_lengths"]:
    #             f.write(f"{val}\n")
    return {
        "policy": policy_name,
        "completed_requests": completed,
        "avg_wait_time": avg_wait,
        "mean_response_time": avg_response,
        "avg_jct_ratio": avg_jct_ratio,
        "avg_utilization": avg_utilization,
        "detailed_stats": stats,
        "peak_memory": stats["peak_age"],
        "total_preemptions": stats["total_preemptions"],
    }


def compare_policies(policies, sim_params=None, metrics=None):
    """Compare multiple policies with the same simulation parameters."""
    if sim_params is None:
        sim_params = {
            "sim_time": 5000,
            "arrival_rate": 0.3,
            "batch_size": 4,
            "token_gen_rate": 10
        }
        
    if metrics is None:
        metrics = ["completed_requests", "avg_wait_time", "mean_response_time", "avg_jct_ratio", "avg_utilization"]
        
    results = []
    
    for policy_class in policies:
        print(f"Running simulation with {policy_class.__name__}...")
        result = run_simulation(policy_class, **sim_params)
        results.append(result)
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df


def plot_response_memory(results_df, experiment_id=None, x_label=None, type = None):
    # Set up the figure
    results_df = results_df.sort_values(by="hp")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    response_times = results_df["mean_response_time"].tolist()
    memory = results_df["peak_memory"].tolist()
    hp = results_df["hp"].tolist()

   # Plot mean response time on left y-axis
    ax1.plot(hp, response_times, color='tab:blue', marker='o', label='Mean Response Time', alpha=0.3)
    ax1.set_ylabel('Mean Response Time', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create secondary y-axis for peak memory
    ax2 = ax1.twinx()
    ax2.plot(hp, memory, color='tab:green', marker='s', label='Peak Memory',  alpha=0.3)
    ax2.set_ylabel('Peak Memory', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Shared x-axis label and title
    ax1.set_xlabel(x_label if x_label else "Tuning Parameter")
    ax1.set_title(f"Comparison of Memory Usage and Mean Response Time for DTPRPT with {type} limit function")


    # Save and show
    if experiment_id:
        plt.savefig(f"llm_scheduling_iclr/experiments/response_vs_memory_{type}_{experiment_id}.png")
    plt.tight_layout()
    plt.show()

    return fig


def run_hp_experiment(args):
    policy, sim_params, type, hp, experiment_id = args
    score_params = {"type": type, "hp": hp}
    result = run_simulation(policy, **sim_params, score_params=score_params)
    result["hp"] = hp
    result["type"] = type
    return result


def response_memory_experiment(policy: Policy, sim_params: Dict[str, float], hyperparams: List[float], experiment_id=None):
    DTPRPT_types = ["parabola", "hyperbola", "exponential"]

    if policy == DTPRPT:
        all_jobs = [
            (policy, sim_params, type, hp, experiment_id)
            for type in DTPRPT_types
            for hp in hyperparams
        ]
        with Pool(processes=min(len(all_jobs), N_CPUS)) as pool:
            all_results = []
            for result in tqdm(pool.imap_unordered(run_hp_experiment, all_jobs), total=len(all_jobs)):
                # print(f"✔️ Finished: DTPRPT({result['type']})_{result['hp']}")
                all_results.append(result)

        print(f"\nDetailed Performance Summary: for Experiment ID {experiment_id}")
        print("="*80)
        print()

        # Group by type and plot
        for type in DTPRPT_types:
            type_results = [res for res in all_results if res["type"] == type]
            type_df = pd.DataFrame(type_results)
            
            # Print results for each hyperparameter
            for _, result in type_df.iterrows():
                print(f"DTPRPT({type})_{result['hp']}:")
                print(f"  Completed requests: {int(result['completed_requests'])}")
                print(f"  Average wait time: {result['avg_wait_time']:.3f}")
                print(f"  Mean response time: {result['mean_response_time']:.3f}")
                print(f"  Average JCT ratio: {result['avg_jct_ratio']:.3f}")
                print(f"  Average utilization: {result['avg_utilization']:.2f}")
                print(f"  Peak memory: {int(result['peak_memory'])}")
                print(f"  Total preemptions: {int(result['total_preemptions'])}")
                print()
            
            plot_response_memory(type_df, experiment_id=experiment_id, x_label="h", type=type)
            
        print("="*80)
        print()
    else:
        raise NotImplementedError



def plot_results(results_df, metrics=None, experiment_id=None):
    """Plot comparison of policies based on selected metrics."""
    if metrics is None:
        metrics = ["completed_requests", "avg_wait_time", "mean_response_time", "avg_jct_ratio", "avg_utilization"]

    n_metrics = len(metrics)
    policies = results_df["policy"].tolist()

    # Set up the figure
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 6))
    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = results_df[metric].tolist()

        # Create bar chart
        ax.bar(policies, values)
        ax.set_title(f"{metric}")
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels(policies, rotation=45, ha="right")

        # Add text labels on bars
        for j, v in enumerate(values):
            ax.text(j, v * 1.01, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"llm_scheduling_iclr/experiments/policy_comparison_{experiment_id}.png")
    plt.show()

    return fig

def run_with_args(args):
    policy_class, sim_time, arrival_rate, batch_size, token_gen_rate, job_service_distribution,  length_distribution, seed, sigma, score_params = args
    return run_simulation(
        policy_class=policy_class,
        sim_time=sim_time,
        arrival_rate=arrival_rate,
        batch_size=batch_size,
        token_gen_rate=token_gen_rate,
        job_service_distribution=job_service_distribution,
        length_distribution=length_distribution,
        seed=seed,
        sigma=sigma,
        score_params=score_params,
    )

def raw_experiment(policies: List[Policy], sim_params: Dict[str, float], experiment_id=None, plot=True):
    DTPRPT_types = ["parabola", "hyperbola", "exponential"]
    DTRPRT_hps = [0.6, 0.2, 0.3]
    policy_runs = []

    # Prepare runs for all policies, including DTPRPT variants
    for policy_class in policies:
        if policy_class.__name__ == "DTPRPT":
            for type, hp in zip(DTPRPT_types, DTRPRT_hps):
                score_params = {"type": type, "hp": hp}
                policy_runs.append((policy_class, *sim_params.values(), score_params))
        else:
            policy_runs.append((policy_class, *sim_params.values(), {}))

    # Run all simulations in parallel
    with Pool(processes=min(len(policy_runs), N_CPUS)) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(run_with_args, policy_runs), total=len(policy_runs)):
            # print(f"✔️ Finished: {result['policy']}")
            results.append(result)

    results_df = pd.DataFrame(results)

    if plot:
        # All simulations are now complete; we can plot and analyze.
        metrics = ["mean_response_time", "peak_memory", "avg_wait_time", "avg_jct_ratio"]

        # Plot histogram of response times for each policy
        plt.figure(figsize=(10, 6))
        for result in results:
            response_times = result["detailed_stats"]["response_times"]
            plt.hist(response_times, alpha=0.5, bins=20, label=result["policy"])

        plt.legend()
        plt.title("Response Time Distributions")
        plt.xlabel("Response Time")
        plt.ylabel("Frequency")
        plt.savefig(f"llm_scheduling_iclr/experiments/response_time_distributions_{experiment_id}.png")

        # Plot aggregate metrics
        plot_results(results_df, metrics, experiment_id)

    # Print out a detailed summary report
    print(f"Detailed Performance Summary: for Experiment ID {experiment_id}")
    print("=" * 80)
    for index, row in results_df.iterrows():
        policy = row["policy"]
        print(f"\n{policy}:")
        print(f"  Completed requests: {row['completed_requests']:.0f}")
        print(f"  Average wait time: {row['avg_wait_time']:.3f}")
        print(f"  Mean response time: {row['mean_response_time']:.3f}")
        print(f"  Average JCT ratio: {row['avg_jct_ratio']:.3f}")
        print(f"  Average utilization: {row['avg_utilization']:.2f}")
        print(f"  Peak memory: {row['peak_memory']:.0f}")
        print(f"  Total preemptions: {row['total_preemptions']:.0f}")
    print("=" * 80)



class WorkloadPreset:
    def __init__(self, sim_time, arrival_rate, token_gen_rate, job_service_distribution = None, sigma=None):
        self.sim_time = sim_time
        self.arrival_rate = arrival_rate
        self.token_gen_rate = token_gen_rate
        self.job_service_distribution = job_service_distribution
        self.sigma = sigma

    def get_sim_params(self):
        return { 
            "sim_time": self.sim_time,         # Total simulation time
            "arrival_rate": self.arrival_rate,      # Mean arrivals per time unit
            "batch_size": 1,          # Number of parallel sequences
            "token_gen_rate": self.token_gen_rate,     # Tokens generated per time unit
            "job_service_distribution": self.job_service_distribution or "exponential-exponential",
            "length_distribution": {
                "p1": (0.332, 78, 97),
                "p2": (0.282, 97, 120),
                "p3": (0.245, 120, 148),
                "p4": (0.082, 148, 183),
                "p5": (0.032, 183, 226),
                "p6": (0.013, 226, 279),
                "p7": (0.0005, 279, 344),
                "p8": (0.0005, 344, 425),
                "p9": (0.0003, 425, 525),
                "p10": (0.0001, 525, 648),
                "p11": (0.0001, 648, 800)
            },
            "seed": 42,
            "sigma": self.sigma
        }

burst_workload = WorkloadPreset(
    sim_time=100,
    arrival_rate=100,
    token_gen_rate=256,
    job_service_distribution="exponential-exponential"
)

poisson_workload = WorkloadPreset(
    sim_time=25000,
    arrival_rate=0.6,
    token_gen_rate=256,
    job_service_distribution="exponential-exponential"
) 

natural_workload = WorkloadPreset(
    sim_time=250000,
    arrival_rate=0.6,
    token_gen_rate=256,
    job_service_distribution="realistic-normal",
    sigma=0.4
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=float, default=0.4, help='Sigma parameter for prediction noise')
    parser.add_argument('--token_gen_rate', type=int, default=256, help='Tokens generated per time unit')
    parser.add_argument('--arrival_rate', type=float, default=0.8, help='Arrival rate of poisson process')
    args = parser.parse_args()


    experiment_id = random.randint(100,999)
    print(f"Experiment ID: {experiment_id}")

    # sim_params = WorkloadPreset(
    #     sim_time=100000,
    #     arrival_rate=args.arrival_rate,
    #     token_gen_rate=args.token_gen_rate,
    #     sigma=args.sigma
    # ).get_sim_params()

    workload = poisson_workload
    # workload.sigma = args.sigma
    # workload.token_gen_rate = args.token_gen_rate
    # workload.arrival_rate = args.arrival_rate
    sim_params = workload.get_sim_params()

    # Policies to compare
    # policies = [FCFS, SPRPT, LSPRPT, DTPRPT]  

    # raw_experiment(policies, sim_params, experiment_id, plot=False)

    hyperparams = np.arange(0.05, 1.05, 0.05).round(2).tolist()
    response_memory_experiment(DTPRPT, sim_params=sim_params, experiment_id=experiment_id, hyperparams=hyperparams)
