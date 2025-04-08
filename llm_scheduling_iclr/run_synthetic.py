from collections import deque
import random
from vllm.core.policy import *
from vllm.sequence import SequenceGroup
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import simpy
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from dataclasses import dataclass, field
import pandas as pd


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
    def __init__(self, request_id: str, arrival_time: float, predicted_length: int):
        self.request_id = request_id
        self.metrics = Metrics(arrival_time=arrival_time)
        
        # Create a list of remaining lengths for each position
        # This simulates how predicted remaining tokens decreases as we generate
        remain_length = []
        for i in range(predicted_length):
            remain_length.append(predicted_length - i)
        
        self.sampling_params = SamplingParams(remain_length=remain_length)
        self._seqs = [Sequence()]
        
    def get_seqs(self) -> List[Sequence]:
        return self._seqs
    
    def update_tokens(self, tokens: int) -> None:
        """Update the number of computed tokens."""
        self._seqs[0].data.update_computed_tokens(tokens)

# Simulation environment
class RequestGenerator:
    def __init__(self, env, scheduler, arrival_rate=1.0, length_distribution=None):
        self.env = env
        self.scheduler = scheduler
        self.arrival_rate = arrival_rate
        self.length_distribution = length_distribution or {
            "short": (0.4, 20, 50),    # 40% chance, 20-50 tokens
            "medium": (0.3, 100, 300), # 30% chance, 100-300 tokens
            "long": (0.3, 500, 1000)   # 30% chance, 500-1000 tokens
        }
        self.request_count = 0
        self.env.process(self.generate_requests())
        
    def generate_requests(self):
        while True:
            # Generate next arrival from Poisson process
            interarrival_time = random.expovariate(self.arrival_rate)
            yield self.env.timeout(interarrival_time)
            
            # Generate request length from distribution
            length_type = random.choices(
                population=list(self.length_distribution.keys()),
                weights=[dist[0] for dist in self.length_distribution.values()]
            )[0]
            
            dist = self.length_distribution[length_type]
            length = random.randint(dist[1], dist[2])
            
            # Create sequence group and submit to scheduler
            request_id = f"req_{self.request_count}"
            self.request_count += 1
            seq_group = SequenceGroup(request_id, self.env.now, length)
            
            self.scheduler.submit_request(seq_group)


class Scheduler:
    def __init__(self, env, policy_class, batch_size=4, token_gen_rate=4):
        self.env = env
        self.policy = policy_class()
        self.batch_size = batch_size  # Number of sequences that can run in parallel
        self.token_gen_rate = token_gen_rate  # Tokens generated per time unit
        
        self.waiting_queue = deque()
        self.running_sequences = {}  # request_id -> (seq_group, process)
        
        self.stats = {
            "completed_requests": 0,
            "total_wait_time": 0,
            "total_turnaround_time": 0,
            "request_lengths": [],
            "request_completion_times": [],
            "wait_times": [],
            "turnaround_times": [],
            "jct_ratios": [],  # Job Completion Time / Job Size
            "utilization_log": []
        }
        
        # Start the scheduler process
        self.env.process(self.run())
        
    def submit_request(self, seq_group):
        self.waiting_queue.append(seq_group)
        # Log request size
        self.stats["request_lengths"].append(seq_group.sampling_params.remain_length[0])
        
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
        
        # Add to running sequences and track with policy
        process = self.env.process(self.process_sequence(seq_group))
        self.running_sequences[seq_group.request_id] = (seq_group, process)
        self.policy.add_running_sequence(seq_group)
    
    def process_sequence(self, seq_group):
        request_id = seq_group.request_id
        total_length = seq_group.sampling_params.remain_length[0]
        current_tokens = 0
        
        start_time = self.env.now
        
        while current_tokens < total_length:
            # Update token count
            tokens_to_generate = min(self.token_gen_rate, total_length - current_tokens)
            current_tokens += tokens_to_generate
            seq_group.update_tokens(current_tokens)
            
            # Simulate token generation time
            yield self.env.timeout(1.0)  # Each step takes 1 time unit
            
            # Check if we should be preempted
            if self.should_preempt(seq_group):
                # Add back to waiting queue
                self.waiting_queue.append(seq_group)
                self.policy.preempt_sequence(seq_group)
                del self.running_sequences[request_id]
                return  # Exit the process
        
        # Sequence is complete
        end_time = self.env.now
        turnaround_time = end_time - seq_group.metrics.arrival_time
        
        # Update stats
        self.stats["completed_requests"] += 1
        self.stats["total_turnaround_time"] += turnaround_time
        self.stats["request_completion_times"].append(end_time)
        self.stats["turnaround_times"].append(turnaround_time)
        self.stats["jct_ratios"].append(turnaround_time / total_length)
        
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
        running_priority = self.policy.get_priority(self.env.now, running_seq)
        for waiting_seq in self.waiting_queue:
            waiting_priority = self.policy.get_priority(self.env.now, waiting_seq)
            if waiting_priority > running_priority:
                return True
                
        return False


def run_simulation(policy_class, sim_time=1000, arrival_rate=0.5, batch_size=4, token_gen_rate=4):
    """Run a simulation with the given policy and parameters."""
    env = simpy.Environment()
    scheduler = Scheduler(env, policy_class, batch_size, token_gen_rate)
    generator = RequestGenerator(env, scheduler, arrival_rate)
    
    # Run simulation
    env.run(until=sim_time)
    
    # Calculate final statistics
    stats = scheduler.stats
    completed = stats["completed_requests"]
    
    if completed > 0:
        avg_wait = stats["total_wait_time"] / completed
        avg_turnaround = stats["total_turnaround_time"] / completed
        avg_jct_ratio = sum(stats["jct_ratios"]) / completed
    else:
        avg_wait = avg_turnaround = avg_jct_ratio = 0
        
    # Calculate average utilization
    utilization_samples = stats["utilization_log"]
    avg_utilization = sum(util for _, util in utilization_samples) / len(utilization_samples) if utilization_samples else 0
        
    return {
        "policy": policy_class.__name__,
        "completed_requests": completed,
        "avg_wait_time": avg_wait,
        "avg_turnaround_time": avg_turnaround,
        "avg_jct_ratio": avg_jct_ratio,
        "avg_utilization": avg_utilization,
        "detailed_stats": stats
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
        metrics = ["completed_requests", "avg_wait_time", "avg_turnaround_time", "avg_jct_ratio", "avg_utilization"]
        
    results = []
    
    for policy_class in policies:
        print(f"Running simulation with {policy_class.__name__}...")
        result = run_simulation(policy_class, **sim_params)
        results.append(result)
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df


def plot_results(results_df, metrics=None):
    """Plot comparison of policies based on selected metrics."""
    if metrics is None:
        metrics = ["completed_requests", "avg_wait_time", "avg_turnaround_time", "avg_jct_ratio", "avg_utilization"]
        
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
        ax.set_xticklabels(policies, rotation=45, ha="right")
        
        # Add text labels on bars
        for j, v in enumerate(values):
            ax.text(j, v * 1.01, f"{v:.2f}", ha="center")
            
    plt.tight_layout()
    plt.savefig("policy_comparison.png")
    plt.show()
    
    return fig


# Main experiment
if __name__ == "__main__":
    # Policies to compare
    policies = [FCFS, SJF, SPRPT, LSPRPT, RPSPRPT, LRPSPRPT, DTPRPT]
    
    # Define simulation parameters
    sim_params = {
        "sim_time": 5000,         # Total simulation time
        "arrival_rate": 0.3,      # Mean arrivals per time unit
        "batch_size": 4,          # Number of parallel sequences
        "token_gen_rate": 10      # Tokens generated per time unit
    }
    
    # Metrics to compare
    metrics = ["completed_requests", "avg_wait_time", "avg_turnaround_time", "avg_jct_ratio", "avg_utilization"]
    
    # Run comparison
    results = compare_policies(policies, sim_params, metrics)
    print(results)
    
    # Plot results
    plot_results(results, metrics)
    
    # Additional analysis: plot detailed wait time distributions
    plt.figure(figsize=(10, 6))
    for policy_class in policies:
        policy_name = policy_class.__name__
        print(f"Running detailed analysis for {policy_name}...")
        
        result = run_simulation(policy_class, **sim_params)
        wait_times = result["detailed_stats"]["wait_times"]
        
        plt.hist(wait_times, alpha=0.5, bins=20, label=policy_name)
    
    plt.legend()
    plt.title("Wait Time Distributions")
    plt.xlabel("Wait Time")
    plt.ylabel("Frequency")
    plt.savefig("wait_time_distributions.png")
    plt.show()
    
    # Print out a detailed summary report
    print("\nDetailed Performance Summary:")
    print("="*80)
    for index, row in results.iterrows():
        policy = row["policy"]
        print(f"\n{policy}:")
        print(f"  Completed requests: {row['completed_requests']:.0f}")
        print(f"  Average wait time: {row['avg_wait_time']:.2f}")
        print(f"  Average turnaround time: {row['avg_turnaround_time']:.2f}")
        print(f"  Average JCT ratio: {row['avg_jct_ratio']:.2f}")
        print(f"  Average utilization: {row['avg_utilization']:.2f}")
    print("="*80)