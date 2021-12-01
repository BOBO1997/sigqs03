from typing import List
import numpy as np
from pprint import pprint
import time

import sgs_algorithm
from sgs_algorithm import sgs_algorithm

from mitigation_tools import MitigationTools

class InvSGS(MitigationTools):

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
              rescale: bool = True,
              silent: bool = False) -> dict:
        """
        O(s * n * 2^n) time and O(2^n) space

        Arguments
            counts: raw counts (dict of str to int)
            shots: total number of shot (int)

        Returns
            mitigated_counts: mitigated counts (dict of str to float)
        """

        if shots is None:
            shots = sum(counts.values())

        # make probability vector (dict)
        y = {int(state, 2): counts[state] / shots for state in counts}

        if not silent:
            print("strict inverse + SGS algorithm")

        t1 = time.time()
        # mitigate raw counts y using tensored mitigation # total O(s * n * 2^n)
        x = {state_idx: 0 for state_idx in range(2 ** self.num_clbits)}  # O(s) space # e basis
        for state_idx in range(2 ** self.num_clbits):  # O(2^n) time
            x[state_idx] = self.mitigate_one_state(state_idx, y)  # O(s * n)
        if not silent:
            print("sum of mitigated probability vector x:", sum(x.values()))

        # algorithm by Smolin et al. # O(n * 2^n) time
        x_tilde = sgs_algorithm(x, silent=silent) if sgs else x
        
        t2 = time.time()
        self.time = t2 - t1

        if not silent:
            print(t2 - t1, "s")

        if not silent:
            print("main process: Done!")
        mitigated_counts = {format(state, "0"+str(self.num_clbits)+"b"): x_tilde[state] * shots for state in x_tilde} if rescale else x_tilde # rescale to counts
        return mitigated_counts
