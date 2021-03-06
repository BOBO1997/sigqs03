from typing import List
import numpy as np
from pprint import pprint
import time
from libmitigation.nation_etal import NationEtal

import mitigation_tools
from mitigation_tools import MitigationTools

from itertools import combinations
import copy


class NewLM(NationEtal):

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

        # self.index_to_keys = None
        self.A_tilde = None

    # OK
    def apply(self,
              counts: dict,
              shots: int = None,
              d: bool = True,
              rescale: bool = True,
              silent: bool = False) -> dict:
        """
        Arguments
            counts: raw counts (dict of str to int)
            shots: total number of shot (int)

        Returns
            mitigated_counts: mitigated counts (dict of str to float)
        """

        if shots is None:
            shots = sum(counts.values())

        # make probability vector (dict)
        # y = {int(state, 2): counts[state] / shots for state in counts}
        y = {state: counts[state] / shots for state in counts}
        print(y)

        if not silent:
            print("Method by Nation, Kang, Sundaresan, and Gambatta")

        # Prepare small calibration matrix A tilde
        keys = self.extend_keys(list(y.keys()), d)
        self.prepare_A_tilde(sorted(keys))

        extended_y = self.extend_vectors(y, keys)
        x = self.apply_inverse_A(extended_y)
        if not silent:
            print("sum of mitigated probability vector x:", sum(x.values()))

        if not silent:
            print("main process: Done!")

        mitigated_counts = {format(state, "0"+str(self.num_clbits)+"b"): x[state] * shots for state in x} if rescale else x  # rescale to counts
        return mitigated_counts
