from typing import List
import numpy as np
from pprint import pprint
import copy

import sgs_algorithm
from sgs_algorithm import sgs_algorithm

import mitigation_tools
from mitigation_tools import MitigationTools


class InvLMSGS(MitigationTools):

    # OK
    def __init__(self,
                 num_clbits: int,
                 cal_matrices: List[np.array],
                 mit_pattern: List[List[int]] = None,
                 meas_layout: List[int] = None) -> None:
        """
        Initialize the TensoredMitigation class
        
        Arguments
            num_clbits: number of measured qubits (int)
            cal_matrices: calibration matrices (list of 2 * 2 numpy array)
            meas_layout: the mapping from classical registers to qubits
        """
        super().__init__(num_clbits=num_clbits,
                         cal_matrices=cal_matrices,
                         mit_pattern=mit_pattern,
                         meas_layout=meas_layout)

    # OK
    def apply(self,
              counts: dict,
              shots: int = None,
              sgs: bool = True,
              rescale: bool = True) -> dict:
        """
        O(4^n) time and O(2^n) space

        Arguments
            counts: raw counts (dict of str to int)
            shots: total number of shot (int)
        Returns
            mitigated_counts: mitigated counts (dict of str to float)
        """

        if shots is None:
            shots = sum(counts.values())

        # make probability vector (dict), O(s) time
        y = {int(state, 2): counts[state] / shots for state in counts}

        print("Lagrange Multiplier + SGS algorithm")

        # preprocess 1: compute sum of x, O(s * n * 2^n) time in total
        sum_of_x = 0
        x = {state_idx: 0 for state_idx in range(2 ** self.num_clbits)}  # O(s) space # e basis
        for state_idx in range(2 ** self.num_clbits):  # O(2^n) time
            sum_of_col = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinv_matrices))  # O(n) time
            sum_of_x += sum_of_col * y[state_idx]
            x[state_idx] = self.mitigate_one_state(state_idx, y)  # O(n * s) time
        print("sum of mitigated probability vector x:", sum(x.values()))
        print("sum_of_x:", sum_of_x)

        # preprocess 2: compute the denominator of delta naively, O(n * 2^n) time in total
        delta_denom = 0
        for state_idx in range(2 ** self.num_clbits):  # O(2^n) time
            sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvVs))  # O(n) time
            lambda_i = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvSigmas))  # O(n) time
            delta_denom += (sum_of_vi ** 2) / (lambda_i ** 2)
        delta_coeff = (1 - sum_of_x) / delta_denom  # O(1) time

        # prepare x_hat, O(4^n) time in total
        x_hat = copy.deepcopy(x)
        for col_idx in range(2 ** self.num_clbits):  # O(2^n) time
            sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(col_idx, self.pinvVs))  # O(n) time
            lambda_i = self.sum_of_tensored_vector(self.choose_vecs(col_idx, self.pinvSigmas))  # O(n) time
            delta_col = sum_of_vi / (lambda_i ** 2)
            v_col = self.v_basis(0, x_hat.keys())
            for state_idx in x_hat: # O(2^n) time
                x_hat[state_idx] += delta_coeff * delta_col * v_col[state_idx]

        # algorithm by Smolin et al. # O(n * 2^n) time
        x_tilde = sgs_algorithm(x_hat) if sgs else x_hat

        print("main process: Done!")
        mitigated_counts = {format(state, "0"+str(self.num_clbits)+"b"): x_tilde[state] * shots for state in x_tilde} if rescale else x_tilde # rescale to counts
        return mitigated_counts