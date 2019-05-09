#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plugin for accelerated gradient

Created on Mon Aug  6 16:43:14 2018

@author: sarkissi
"""


import astra
import numpy as np
import six




class AcceleratedGradientPlugin(astra.plugin.base):
    """Accelerated Gradient Descend a la Nesterov.
    Options:
    'MinConstraint': constrain values to at least this (optional)
    'MaxConstraint': constrain values to at most this (optional)
    """

    astra_name = "AGD-PLUGIN"


    def power_iteration(self, A, num_simulations):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = np.random.rand(A.shape[1])
        b_k1_norm = 1

        print('running power iteration to determine step size', flush=True)
        for i in range(num_simulations):

            # calculate the matrix-by-vector product Ab
            b_k1 = A.T*A*b_k

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm
        return b_k1_norm

    def initialize(self, cfg, liptschitz=1, MinConstraint = None, MaxConstraint = None):
        self.W = astra.OpTomo(cfg['ProjectorId'])
        self.vid = cfg['ReconstructionDataId']
        self.sid = cfg['ProjectionDataId']
        self.min_constraint = MinConstraint
        self.max_constraint = MaxConstraint

        try:
            v = astra.data2d.get_shared(self.vid)
            s = astra.data2d.get_shared(self.sid)
            self.data_mod = astra.data2d
        except Exception:
            v = astra.data3d.get_shared(self.vid)
            s = astra.data3d.get_shared(self.sid)
            self.data_mod = astra.data3d


        self.liptschitz = self.power_iteration(self.W, 10)
        self.nu = 1/self.liptschitz

        self.ATy = self.W.BP(s)
        self.obj_func = None
        print('plugin initialized.', flush=True)

    def run(self, its):
        v = self.data_mod.get_shared(self.vid)
        s = self.data_mod.get_shared(self.sid)
        W = self.W
        ATy = self.ATy
        x_apgd = v
        nu = self.nu

        # New variables
        t_acc         = 1
        x_old         = x_apgd.copy();
        NRMx          = np.zeros_like(v)  # normal operator A'*A
        NRMx_old      = NRMx.copy();
        gradient      = (NRMx - ATy);

        self.obj_func = np.zeros(its)

        print('running', str(its), 'iterations of Accelerated Gradient plugin.', flush=True)
        for i in range(its):
            if i%10 == 0:
                print('iteration', str(i), '/', str(its), flush=True)

            tau = (t_acc-1)/(t_acc+2)
            t_acc = t_acc + 1

            # Compute descent direction
            descent_direction = gradient - tau/nu * (x_apgd - x_old) + tau * (NRMx - NRMx_old);

            # update x
            x_old[:]  =  x_apgd[:]
            x_apgd -= nu * descent_direction
            if self.min_constraint is not None or self.max_constraint is not None:
                x_apgd.clip(min=self.min_constraint, max=self.max_constraint, out=v)

            # Compute all other updates
            Wx            = W.FP(x_apgd)
            NRMx_old[:]   = NRMx[:]
            NRMx          = W.BP(Wx)
            gradient      = NRMx - ATy
            self.obj_func[i] = 0.5*np.linalg.norm(Wx - s)**2

            if(self.obj_func[i] > np.min(self.obj_func[0:i+1])):
                t_acc = 1; # restart acceleration
                print('acceleration restarted!', flush=True)
