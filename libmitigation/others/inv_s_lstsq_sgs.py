from typing import List
import numpy as np
from pprint import pprint

import sgs_algorithm
from sgs_algorithm import sgs_algorithm

from scipy.optimize import minimize

from mitigation_tools import MitigationTools


class InvSLstsqSGS(MitigationTools):

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
              silent: bool = True) -> dict:
        """
        O(s * s * n) time and O(s) space

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
            print("Restriction to labels of y + SciPy optimization + SGS algorithm")
        # O(s * s * n) time
        x_s = {state_idx: 0 for state_idx in y}  # O(s) space # e basis
        for state_idx in y:  # O(s) time
            x_s[state_idx] = self.mitigate_one_state(
                state_idx, y)  # O(n * s) time
        if not silent:
            print("sum of mitigated probability vector x_s:", sum(x_s.values()))

        # algorithm by Smolin et al. # O(s * log(s)) time
        # x_tilde = sgs_algorithm(x_s) if sgs else x_s

        keys = sorted(x_s.keys())
        target_ndarr = np.zeros(len(list(keys)))
        for i, k in enumerate(keys):
            target_ndarr[i] = x_s[k]

        def fun(x):
            return sum((x - target_ndarr) ** 2)

        x0 = np.random.rand(len(list(x_s.keys())))
        x0 = x0 / sum(x0)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        res = minimize(fun, x0, method='SLSQP', constraints=cons, tol=1e-6)

        x_tilde = {}
        for i, k in enumerate(keys):
            x_tilde[k] = res.x[i]
        if not silent:
            print("sum of mitigated probability vector x_tilde:", sum(x_tilde.values()))
        
        x_tilde = sgs_algorithm(x_tilde, silent=silent)

        if not silent:
            print("main process: Done!")
        mitigated_counts = {format(state, "0"+str(self.num_clbits)+"b")
                                   : x_tilde[state] * shots for state in x_tilde} if rescale else x_tilde  # rescale to counts
        return mitigated_counts
