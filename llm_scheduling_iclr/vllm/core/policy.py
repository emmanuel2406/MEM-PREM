from collections import deque
from typing import Deque, Dict

# from vllm.sequence import SequenceGroup
from simulation import SequenceGroup
import numpy as np
import sys
import math

AGE_CAPACITY = 512

class Policy:
    def __init__(self):
        # Initialize tracking variables
        self.active_sequences: Dict[str, SequenceGroup] = {}  # Currently running sequences
        self.preempted_sequences: Dict[str, SequenceGroup] = {}  # Preempted but not completed sequences
        self.age_capacity = AGE_CAPACITY  # Maximum sum of token ages threshold (configurable)

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))
    def add_running_sequence(self, seq_group: SequenceGroup) -> None:
        """Track a new running sequence."""
        seq_id = seq_group.request_id
        self.active_sequences[seq_id] = seq_group

    def preempt_sequence(self, seq_group: SequenceGroup) -> None:
        """Move a sequence from running to preempted state."""
        seq_id = seq_group.request_id
        if seq_id in self.active_sequences:
            self.preempted_sequences[seq_id] = self.active_sequences[seq_id]
            del self.active_sequences[seq_id]

    def complete_sequence(self, seq_group: SequenceGroup) -> None:
        """Remove a sequence that has completed."""
        seq_id = seq_group.request_id
        if seq_id in self.active_sequences:
            del self.active_sequences[seq_id]
        if seq_id in self.preempted_sequences:
            del self.preempted_sequences[seq_id]

    def get_age_ratio(self, cur_group: SequenceGroup, soap_compliant: bool) -> float:
        """
        Calculate ratio of total sequence ages to age capacity.
        This represents how much memory resource is currently used by
        all active and preempted sequences.
        """
        if soap_compliant:
            return cur_group.get_seqs()[0].data.get_num_computed_tokens() / self.age_capacity
    
        # Calculate sum of ages for all active and preempted sequences
        total_age = 0.0
        # Add ages of active sequences
        for seq_id, seq_group in self.active_sequences.items():
            total_age += seq_group.get_seqs()[0].data.get_num_computed_tokens()
        # Add ages of preempted sequences
        for seq_id, seq_group in self.preempted_sequences.items():
            total_age += seq_group.get_seqs()[0].data.get_num_computed_tokens()
        # Calculate ratio
        return total_age / self.age_capacity


class FCFS(Policy):
    """
    First-come, first-served policy
    """

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.metrics.arrival_time

class SPRPT(Policy):
    """
    Shortest Preemption Remaining Processing Time
    """
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        #return seq_group.get_seqs()[0].expected_out_len
        predicted_len = seq_group.sampling_params.remain_length[0]
        generated_len = seq_group.get_seqs()[0].data.get_num_computed_tokens()
        score = predicted_len - generated_len

        return -score

class SJF(Policy):
    """
    Shortest Job First
    """
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return -seq_group.sampling_params.remain_length[0]

class LSPRPT(Policy):
    """
    Limited Shorest Preemption Remaining Processing Time
    """
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        t_limit: float = 0.8,
    ) -> float:
        #return seq_group.get_seqs()[0].expected_out_len
        #higher score means higher priority
        predicted_len = seq_group.sampling_params.remain_length[0]
        generated_len = seq_group.get_seqs()[0].data.get_num_computed_tokens()
        if generated_len > t_limit*predicted_len:
            score = sys.maxsize
        else:
            score = - predicted_len + generated_len
        return score


class RPSPRPT(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        t_limit: float = 0.5,
    ) -> float:
        generated_len = seq_group.get_seqs()[0].data.get_num_computed_tokens()
        if generated_len < len(seq_group.sampling_params.remain_length):
            predicted_remaining_len = seq_group.sampling_params.remain_length[generated_len]
        else:
            predicted_remaining_len = seq_group.sampling_params.remain_length[-1]

        return -predicted_remaining_len

    def compare(self, waiting_seq: SequenceGroup, running_seq: SequenceGroup, now: float) -> int:
        """
        Compare two sequences based on their priority. Returns:
        > 0 if waiting_seq has higher priority than running_seq,
        < 0 if running_seq has higher priority than waiting_seq,
        0 if both have the same priority.
        """
        waiting_priority = self.get_priority(now, waiting_seq)
        running_priority = self.get_priority(now, running_seq)

        return waiting_priority - running_priority

class LRPSPRPT(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        #t_limit: float = 0.5,
        t_limit: float = 0.8,
    ) -> float:
        predicted_initial_len = seq_group.sampling_params.remain_length[0]
        generated_len = seq_group.get_seqs()[0].data.get_num_computed_tokens()
        if generated_len > t_limit*predicted_initial_len:
            score = sys.maxsize
        else:
            if generated_len < len(seq_group.sampling_params.remain_length):
                predicted_remaining_len = seq_group.sampling_params.remain_length[generated_len]
            else:
                predicted_remaining_len = seq_group.sampling_params.remain_length[-1]
            score = -predicted_remaining_len
        return score

    def compare(self, waiting_seq: SequenceGroup, running_seq: SequenceGroup, now: float) -> int:
        """
        Compare two sequences based on their priority. Returns:
        > 0 if waiting_seq has higher priority than running_seq,
        < 0 if running_seq has higher priority than waiting_seq,
        0 if both have the same priority.
        """
        waiting_priority = self.get_priority(now, waiting_seq)
        running_priority = self.get_priority(now, running_seq)

        return waiting_priority - running_priority

class DTPRPT(Policy):
    """
    Dynamic Threshold Preemption Remaining Processing Time
    """
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        type: str = 'parabola',
        hp: float = 0.8, #hyperparameter in type
        soap_compliant: bool = False,
        **kwargs
    ) -> float:
        m0 =  self.get_age_ratio(cur_group=seq_group, soap_compliant=soap_compliant)
        if type == 'parabola':
            """
                hp represents the fixed point of (hp, 0.8) on the parabola to tune the shape of the curve
            """
            c1 = (hp - 0.2) / (hp ** 2 - hp + 1e-8)
            c2 = -1 - c1
            t_limit = c1 * m0 ** 2 + c2 * m0 + 1
        elif type == 'hyperbola':
            """
                hp represents the fixed point of (hp, 0.8) on the hyperbola to tune the shape of the curve
            """
            c = 0.2 - 0.2 * hp
            t_limit = c / (m0 - 1 + 1e-8) + 1
        elif type == 'exponential':
            """
                hp represents the scaling factor in the exponential
            """
            t_limit = (1 - math.exp(-hp + hp * m0)) / (1 - math.exp(-hp) + 1e-8)

        predicted_len = seq_group.sampling_params.remain_length[0]
        generated_len = seq_group.get_seqs()[0].data.get_num_computed_tokens()
        if generated_len > t_limit*predicted_len:
            score = sys.maxsize
        else:
            score = - predicted_len + generated_len
        return score
    

class PasSPoRT(Policy):
    """
    Shortest Preemption Remaining Processing Time
    """
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        m0 =  self.get_age_ratio(cur_group=seq_group, soap_compliant=True)
        predicted_len = seq_group.sampling_params.remain_length[0]
        generated_len = seq_group.get_seqs()[0].data.get_num_computed_tokens()
        # score = predicted_len - generated_len **(1 + m0)
        score = predicted_len - generated_len *  np.exp(m0 * generated_len) /(1 + np.exp(m0 * generated_len))
        return -score

class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS,
                        'SPRPT': SPRPT,
                        'LSPRPT': LSPRPT,
                        'RPSPRPT': RPSPRPT,
                        'LRPSPRPT': LRPSPRPT,
                        'SJF': SJF,
                        'DTPRPT': DTPRPT}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
