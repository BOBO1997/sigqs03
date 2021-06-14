# Library of Proposed Methods

## Overview

The whole process of algorithms contains two parts.
1. Satisfying the sum-to-one condition
2. Negative cancelling using [Smolin, Gambetta, Smith (SGS) algorithm](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.070502)

## inv_sgs.py (class `InvSGS`)

Total: $O(sn2^n)$ time, $O(2^n)$ space

1. Apply inverse of the calibration matrix to get whole $2^n$ sized vector.
2. Apply SGS algorithm.

## inv_s_sgs.py (class `InvSSGS`)

Total: $O(s^2n)$ time, $O(s)$ space

1. Apply inverse of the calibration matrix to get the vector of the restricted elements with size $s$.
2. Apply SGS algorithm.

## inv_lm_sgs.py (class `InvLMSGS`)

Total: $O(4^n)$ time, $O(2^n)$ space

1. Apply inverse of the calibration matrix to get whole $2^n$ sized vector.
2. Meet the sum-to-one condition using Lagrange multiplier.
3. Apply SGS algorithm.

## inv_s_lm_sgs.py (class `InvSLMSGS`)

Total: $O(sn2^n)$ time, $O(s)$ space

1. Apply inverse of the calibration matrix to get the vector of the restricted elements with size $s$.
2. Meet the sum-to-one condition using Lagrange multiplier.
3. Apply SGS algorithm.

## inv_s_lm0_sgs.py (class `InvSLM0SGS`)

Total: $O(sn)$ time, $O(s)$ space

1. Apply inverse of the calibration matrix to get the vector of the restricted elements with size $s$.
2. Meet the sum-to-one condition using Lagrange multiplier.
3. SGS algorithm