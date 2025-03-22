from collections import deque
from typing import Deque

from vllm.sequence import SequenceGroup
import sys


class Policy:

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


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.metrics.arrival_time

class SPRPT(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        #return seq_group.get_seqs()[0].expected_out_len
        predicted_len = seq_group.sampling_params.remain_length
        generated_len = seq_group.get_seqs()[0].data.get_num_computed_tokens()
        score = predicted_len - generated_len

        return -score

class SJF(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return -seq_group.sampling_params.remain_length[0]

class LSPRPT(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        t_limit: float = 0.5,
    ) -> float:
        #return seq_group.get_seqs()[0].expected_out_len
        #higher score means higher priority
        predicted_len = seq_group.sampling_params.remain_length
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

class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS,
                        'SPRPT': SPRPT,
                        'LSPRPT': LSPRPT,
                        'RPSPRPT': RPSPRPT,
                        'LRPSPRPT': LRPSPRPT,
                        'SJF': SJF,}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
