from collections import deque
import random

from policy import *
# from vllm.sequence import SequenceGroup
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
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
        self.computed_tokens += tokens


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
    def __init__(self, env, scheduler, arrival_rate=1.0, job_service_distribution="realistic-normal", length_distribution=None, seed=None, sigma=0.5, token_gen_rate=4):
        self.env = env
        self.scheduler = scheduler
        self.arrival_rate = arrival_rate
        self.job_service_distribution = job_service_distribution
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
            if self.env.now + interarrival_time > self.scheduler.sim_time:
                return
            yield self.env.timeout(interarrival_time)

            if self.job_service_distribution == "realistic-normal":
                # Generate request length from distribution
                length_type = self.random.choices(
                    population=list(self.length_distribution.keys()),
                    weights=[dist[0] for dist in self.length_distribution.values()]
                )[0]    

                dist = self.length_distribution[length_type]
                true_length = self.random.randint(dist[1], dist[2])
                predicted_length = max(0, int(true_length + true_length * self.sigma * self.np_random.standard_normal()))
            elif self.job_service_distribution == "exponential-exponential":
                # job service time of mean 1
                true_length = max(1, int(self.random.expovariate(1/ self.token_gen_rate)))
                predicted_length = int(self.random.expovariate(1 / true_length))
            elif self.job_service_distribution == "exponential-perfect":
                true_length = max(1, int(self.random.expovariate(1/ self.token_gen_rate)))
                predicted_length = true_length
            else:
                raise ValueError(f"Invalid job service distribution: {self.job_service_distribution}")


            # Create sequence group and submit to scheduler
            request_id = f"req_{self.request_count}"
            self.request_count += 1
            seq_group = SequenceGroup(request_id, self.env.now, predicted_length, true_length, self.token_gen_rate)

            self.scheduler.submit_request(seq_group)


class Scheduler:
    def __init__(self, env, policy_class, batch_size=1, token_gen_rate=4, sim_time=1000, score_params=None):
        self.env = env
        self.policy = policy_class()
        self.batch_size = batch_size  # Number of sequences that can run in parallel
        self.token_gen_rate = token_gen_rate  # Tokens generated per time unit
        self.sim_time = sim_time
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
        while self.env.now < self.sim_time or self.waiting_queue or self.running_sequences:
            # Sort waiting queue by priority
            self.waiting_queue = self.policy.sort_by_priority(self.env.now, self.waiting_queue)
            
            # Schedule new sequences if slots are available
            while len(self.running_sequences) < self.batch_size and self.waiting_queue:
                next_seq = self.waiting_queue.popleft()
                self.start_sequence(next_seq)
            
            # Making sure the peak age is updated correctly
            total = sum(
                seq.get_seqs()[0].data.get_num_computed_tokens()
                for seq, _ in self.running_sequences.values()
            ) + sum(
                seq.get_seqs()[0].data.get_num_computed_tokens()
                for seq in self.waiting_queue
            )
            self.stats["peak_age"] = max(self.stats["peak_age"], total)

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
        if seq_group.get_seqs()[0].data.get_num_computed_tokens() == 0:
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
            seq_group.update_tokens(tokens_to_generate)

            # Simulate token generation time
            yield self.env.timeout(1.0)  # Each step takes 1 time unit

            # Update age
            self.stats["current_age"] += tokens_to_generate

            # Check if we should be preempted
            if self.should_preempt(seq_group):
                # Add back to waiting queue
                self.waiting_queue.append(seq_group)
                self.policy.preempt_sequence(seq_group)
                self.stats["peak_age"] = max(self.stats["peak_age"], self.stats["current_age"])
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