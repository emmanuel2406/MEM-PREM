from collections import deque
import random

from policy import *
from vllm.sequence import SequenceGroup
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import simpy
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from dataclasses import dataclass, field
import pandas as pd
from multiprocessing import Pool


# Mock classes to simulate vLLM SequenceGroup functionality
@dataclass
class Metrics:
    arrival_time: float = 0.0


@dataclass
class SamplingParams:
    remain_length: List[int] = field(default_factory=list)


@dataclass
class SequenceData:
    computed_tokens: int = 0
    
    def get_num_computed_tokens(self) -> int:
        return self.computed_tokens
    
    def update_computed_tokens(self, tokens: int) -> None:
        self.computed_tokens = tokens


@dataclass
class Sequence:
    data: SequenceData = field(default_factory=SequenceData)


class SequenceGroup:
    def __init__(self, request_id: str, arrival_time: float, predicted_length: int, true_length: int, token_gen_rate: int):
        self.request_id = request_id
        self.metrics = Metrics(arrival_time=arrival_time)
        
        # Create a list of remaining lengths for each position
        # This simulates how predicted remaining tokens decreases as we generate
        remain_length = []
        for i in range(true_length):
            remain_length.append(predicted_length - token_gen_rate * i)

        self.sampling_params = SamplingParams(remain_length=remain_length)
        self.true_length = true_length
        self._seqs = [Sequence()]

    def get_seqs(self) -> List[Sequence]:
        return self._seqs
    
    def update_tokens(self, tokens: int) -> None:
        """Update the number of computed tokens."""
        self._seqs[0].data.update_computed_tokens(tokens)

# Simulation environment
class RequestGenerator:
    def __init__(self, env, scheduler, arrival_rate=1.0, length_distribution=None, seed=None, sigma=10, token_gen_rate=4):
        self.env = env
        self.scheduler = scheduler
        self.arrival_rate = arrival_rate
        self.length_distribution = length_distribution or {
            "short": (0.4, 51, 102),    # 40% chance, 20-50 tokens
            "medium": (0.3, 100, 300), # 30% chance, 100-300 tokens
            "long": (0.3, 500, 1000)   # 30% chance, 500-1000 tokens
        }
        self.random = random.Random(seed)
        self.np_random = np.random.default_rng(seed)
        self.request_count = 0
        self.env.process(self.generate_requests())
        self.sigma = sigma
        self.token_gen_rate = token_gen_rate

    def generate_requests(self):
        while True:
            # Generate next arrival from Poisson process
            interarrival_time = self.random.expovariate(self.arrival_rate)
            yield self.env.timeout(interarrival_time)
            # Generate request length from distribution
            length_type = self.random.choices(
                population=list(self.length_distribution.keys()),
                weights=[dist[0] for dist in self.length_distribution.values()]
            )[0]

            dist = self.length_distribution[length_type]
            true_length = self.random.randint(dist[1], dist[2])
            predicted_length = max(0, int(true_length + self.sigma * self.np_random.standard_normal()))

            # Create sequence group and submit to scheduler
            request_id = f"req_{self.request_count}"
            self.request_count += 1
            seq_group = SequenceGroup(request_id, self.env.now, predicted_length, true_length, self.token_gen_rate)

            self.scheduler.submit_request(seq_group)


class Scheduler:
    def __init__(self, env, policy_class, batch_size=1, token_gen_rate=4,score_params=None):
        self.env = env
        self.policy = policy_class()
        self.batch_size = batch_size  # Number of sequences that can run in parallel
        self.token_gen_rate = token_gen_rate  # Tokens generated per time unit

        self.waiting_queue = deque()
        self.running_sequences = {}  # request_id -> (seq_group, process)
        self.score_params = score_params

        self.stats = {
            "completed_requests": 0,
            "total_wait_time": 0,
            "total_response_time": 0,
            "request_lengths": [],
            "request_completion_times": [],
            "wait_times": [],
            "response_times": [],
            "jct_ratios": [],  # Job Completion Time / Job Size
            "utilization_log": [],
            "peak_age": 0,
            "current_age": 0,
            "total_preemptions": 0,
        }
        
        # Start the scheduler process
        self.env.process(self.run())
        
    def submit_request(self, seq_group):
        self.waiting_queue.append(seq_group)
        
    def run(self):
        while True:
            # Sort waiting queue by priority
            self.waiting_queue = self.policy.sort_by_priority(self.env.now, self.waiting_queue)
            
            # Schedule new sequences if slots are available
            while len(self.running_sequences) < self.batch_size and self.waiting_queue:
                next_seq = self.waiting_queue.popleft()
                self.start_sequence(next_seq)
            
            # Record utilization metrics
            self.stats["utilization_log"].append((self.env.now, len(self.running_sequences) / self.batch_size))
            
            # Wait for the next scheduling opportunity (when a sequence completes)
            if self.running_sequences:
                yield self.env.any_of(list(proc for _, proc in self.running_sequences.values()))
            else:
                # If no running sequences but waiting queue is empty, wait for new arrivals
                if not self.waiting_queue:
                    yield self.env.timeout(1.0)  # Arbitrary wait time
    
    def start_sequence(self, seq_group):
        # Calculate how long the request has been waiting
        wait_time = self.env.now - seq_group.metrics.arrival_time
        self.stats["total_wait_time"] += wait_time
        self.stats["wait_times"].append(wait_time)
        self.stats["request_lengths"].append(seq_group.true_length)

        # Add to running sequences and track with policy
        process = self.env.process(self.process_sequence(seq_group))
        self.running_sequences[seq_group.request_id] = (seq_group, process)
        self.policy.add_running_sequence(seq_group)
    
    def process_sequence(self, seq_group):
        request_id = seq_group.request_id
        true_remain_length = seq_group.true_length - seq_group.get_seqs()[0].data.get_num_computed_tokens()
        current_tokens = 0
        
        start_time = self.env.now
        
        while current_tokens < true_remain_length:
            # Update token count
            tokens_to_generate = min(self.token_gen_rate, true_remain_length - current_tokens)
            current_tokens += tokens_to_generate
            seq_group.update_tokens(current_tokens)

            # Simulate token generation time
            yield self.env.timeout(1.0)  # Each step takes 1 time unit

            # Update age
            self.stats["current_age"] += tokens_to_generate

            # Check if we should be preempted
            if self.should_preempt(seq_group):
                # Add back to waiting queue
                self.waiting_queue.append(seq_group)
                self.policy.preempt_sequence(seq_group)
                del self.running_sequences[request_id]
                return  # Exit the process
        
        # Sequence is complete
        end_time = self.env.now
        response_time = end_time - seq_group.metrics.arrival_time

        # Update stats
        self.stats["completed_requests"] += 1
        self.stats["total_response_time"] += response_time
        self.stats["request_completion_times"].append(end_time)
        self.stats["response_times"].append(response_time)
        self.stats["jct_ratios"].append(response_time / seq_group.true_length)
        self.stats["peak_age"] = max(self.stats["peak_age"], self.stats["current_age"])
        self.stats["current_age"] -= seq_group.true_length

        # Remove from running sequences
        self.policy.complete_sequence(seq_group)
        del self.running_sequences[request_id]

    def should_preempt(self, running_seq):
        """Check if any waiting sequence should preempt this running sequence."""
        if not self.waiting_queue:
            return False

        # For policies with a compare method
        if hasattr(self.policy, "compare"):
            waiting_seq = self.waiting_queue[0]  # The highest priority waiting sequence
            return self.policy.compare(waiting_seq, running_seq, self.env.now) > 0

        # For policies without a compare method, use priority
        running_priority = self.policy.get_priority(self.env.now, running_seq, **self.score_params)
        for waiting_seq in self.waiting_queue:
            waiting_priority = self.policy.get_priority(self.env.now, waiting_seq, **self.score_params)
            if waiting_priority > running_priority:
                self.stats["total_preemptions"] += 1
                return True

        return False


def format_name(policy_class, score_params=None) -> str:
    name = policy_class.__name__
    if score_params and "type" in score_params.keys():
        name += f"({score_params['type']})"
    if score_params and "hp" in score_params.keys():
        name += f"_{score_params['hp']}"
    return name

def run_simulation(policy_class, sim_time=1000, arrival_rate=0.5, batch_size=4, length_distribution=None, token_gen_rate=4, score_params=None, seed=None, sigma=10):
    """Run a simulation with the given policy and parameters."""
    env = simpy.Environment()
    scheduler = Scheduler(env, policy_class, batch_size, token_gen_rate, score_params)
    generator = RequestGenerator(env, scheduler, arrival_rate, length_distribution=length_distribution, seed=seed, sigma=sigma, token_gen_rate=4)
    # Run simulation
    env.run(until=sim_time)

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
    if False:
        with open(f"llm_scheduling_iclr/vllm/dumps/{policy_name}.txt", "w") as f:
            for val in stats["request_lengths"]:
                f.write(f"{val}\n")
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
    fig, ax1 = plt.subplots(figsize=(10, 6))
    response_times = results_df["mean_response_time"].tolist()
    memory = results_df["peak_memory"].tolist()
    hp = results_df["hp"].tolist()

   # Plot mean response time on left y-axis
    ax1.plot(hp, response_times, color='tab:blue', marker='o', label='Mean Response Time')
    ax1.set_ylabel('Mean Response Time', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create secondary y-axis for peak memory
    ax2 = ax1.twinx()
    ax2.plot(hp, memory, color='tab:green', marker='s', label='Peak Memory')
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

        N_CPUS = 32 # tune this to your machine
        with Pool(processes=min(len(all_jobs), N_CPUS)) as pool:
            all_results = pool.map(run_hp_experiment, all_jobs)

        # Group by type and plot
        for type in DTPRPT_types:
            type_results = [res for res in all_results if res["type"] == type]
            type_df = pd.DataFrame(type_results)
            plot_response_memory(type_df, experiment_id=experiment_id, x_label="h", type=type)
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
    policy_class, sim_time, arrival_rate, batch_size, token_gen_rate, length_distribution, seed, sigma, score_params = args
    return run_simulation(
        policy_class=policy_class,
        sim_time=sim_time,
        arrival_rate=arrival_rate,
        batch_size=batch_size,
        token_gen_rate=token_gen_rate,
        length_distribution=length_distribution,
        seed=seed,
        sigma=sigma,
        score_params=score_params,
    )

def raw_experiment(policies: List[Policy], sim_params: Dict[str, float], experiment_id=None):
    DTPRPT_types = ["parabola", "hyperbola", "exponential"]
    DTRPRT_hps = [0.7, 0.7, 0.6]
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
    with Pool(processes=len(policy_runs)) as pool:
        results = pool.map(run_with_args, policy_runs)

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

    results_df = pd.DataFrame(results)
    # Plot aggregate metrics
    plot_results(results_df, metrics, experiment_id)

    # Print out a detailed summary report
    print(f"Detailed Performance Summary: for Experiment ID {experiment_id}")
    print("=" * 80)
    for index, row in results_df.iterrows():
        policy = row["policy"]
        print(f"\n{policy}:")
        print(f"  Completed requests: {row['completed_requests']:.0f}")
        print(f"  Average wait time: {row['avg_wait_time']:.2f}")
        print(f"  Mean response time: {row['mean_response_time']:.2f}")
        print(f"  Average JCT ratio: {row['avg_jct_ratio']:.2f}")
        print(f"  Average utilization: {row['avg_utilization']:.2f}")
        print(f"  Peak memory: {row['peak_memory']:.0f}")
        print(f"  Total preemptions: {row['total_preemptions']:.0f}")
    print("=" * 80)




if __name__ == "__main__":

    experiment_id = random.randint(100,999)
    # Define simulation parameters
    sim_params = {
        "sim_time": 50000,         # Total simulation time
        "arrival_rate": 0.05,      # Mean arrivals per time unit
        "batch_size": 1,          # Number of parallel sequences
        "token_gen_rate": 5,     # Tokens generated per time unit
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
        "sigma": 50
    }
    # Policies to compare
    policies = [FCFS, SPRPT, LRPSPRPT, DTPRPT] 

    raw_experiment(policies, sim_params, experiment_id)

    # hyperparams = np.arange(0.0, 1.05, 0.05).round(2).tolist()
    # response_memory_experiment(DTPRPT, sim_params=sim_params, experiment_id=experiment_id, hyperparams=hyperparams)
