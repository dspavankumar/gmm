#!/usr/bin/python

## ----------------------------------------------------------------------------
##    GMM Training using Multiprocessing on Large Datasets
##    Copyright (C) 2014,  D S Pavan Kumar (Email: dspavankumar[at]gmail.com)
##
##    This program is free software; you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation; either version 2 of the License, or
##    (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License along
##    with this program; if not, write to the Free Software Foundation, Inc.,
##    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
## ----------------------------------------------------------------------------

import numpy as np
from multiprocessing import Process, Queue, cpu_count
from iofile import *
import time

## Class definition for diagonal GMM
class GMM:
    ## Initialisation
    def __init__ (self, dim, mix=1):
        self.weights    = (1.0/mix) * np.ones(mix)
        self.means      = np.random.randn(mix, dim)
        self.vars       = np.ones((mix, dim))
        self.mix        = mix
        self.dim        = dim
        self.varfloor   = 0.01

    ## Printing means, variances and weights
    def __str__ (self):
        return "Means:\n%s\nVariances:\n%s\n Weights:\n%s\n" % (self.means, self.vars, self.weights)

    ## Likelihood of a vector
    def pdf(self, vec):
        vecr    = np.tile (vec, self.mix).reshape(self.mix, self.dim)
        return (self.weights * ((2*np.pi)**(-0.5*self.dim)) * np.prod(self.vars,1)**(-0.5) ) * \
                ( np.exp(-0.5 * np.sum( (vecr-self.means)**2 /self.vars, 1) ) )

    ## Posterior of a vector (normalised likelihood vector, sums to unity)
    def posterior(self, vec):
        if self.mix == 1:
            return np.array([1])
        
        post            = self.pdf (vec)
        postsum         = np.sum (post)
        return post/postsum #if (postsum > 1e-12) else np.zeros((self.mix))

    ## Double the number of mixtures
    def doublemixtures (self):
        bias            = np.zeros ((self.mix, self.dim))
        for i in range(self.mix):
            argmaxv             = np.argmax (self.vars[i])
            bias[i][argmaxv]    = self.vars[i][argmaxv]

        self.means      = np.vstack ((self.means + 0.2*bias, self.means - 0.2*bias))
        self.weights    = np.tile (self.weights/2, 2)
        self.vars       = np.vstack ((self.vars, self.vars))
        self.mix        = 2*self.mix

    ## Training step 1 of 3: Initialise statistics accumulation
    def initstats (self):
        self.sgam       = np.zeros(self.mix)
        self.sgamx      = np.zeros((self.mix, self.dim))
        self.sgamxx     = np.zeros((self.mix, self.dim))

    ## Training step 3 of 3: Recompute GMM parameters
    def finishstats (self):
        self.weights    = self.sgam / np.sum(self.sgam)
        denom           = self.sgam.repeat(self.dim).reshape((self.mix,self.dim))
        self.means      = self.sgamx / denom
        self.vars       = self.sgamxx / denom - (self.means**2)
        self.vars[self.vars < self.varfloor] = self.varfloor

    ## Training step 2 of 3: Update the statistics from a set of features
    def update_worker (self, mfclist, Q):
        sgam        = np.zeros(self.mix)
        sgamx       = np.zeros((self.mix, self.dim))
        sgamxx      = np.zeros((self.mix, self.dim))

        for mfcfile in mfclist:
            feats   = readfile (mfcfile)
            for feat in feats:
                gam         = self.posterior(feat)
                sgam        += gam
                sgamx       += np.outer(gam, feat)
                sgamxx      += np.outer(gam, feat**2)
        
        Q.put([sgam, sgamx, sgamxx])

    ## GMM update routine - master
    def updatestats (self, mfclist, threads=cpu_count()):
        with open(mfclist, 'r') as f:
            mfcfiles = f.read().splitlines()

        Q = Queue()
        processes = []
        for thr in xrange(threads):
            p = Process (target=self.update_worker, args=(mfcfiles[thr::threads], Q))
            p.start()
            processes.append(p)

        while Q.qsize < threads:
            time.sleep(0.01)
        
        for thr in xrange(threads):
            sgam, sgamx, sgamxx = Q.get()
            self.sgam       += sgam
            self.sgamx      += sgamx
            self.sgamxx     += sgamxx

    ## Expectation-Maximisation (EM)
    def em (self, mfclist, threads=cpu_count()):
        print "Running EM on", str(self.mix), "mixtures"
        self.initstats()
        self.updatestats(mfclist, threads)
        self.finishstats()

    ## Train GMM (wrapper)
    def train(self, mfclist, mix, threads=cpu_count()):
        print "CPU threads being used:", str(threads)
        if not (np.log(mix)/np.log(2)).is_integer():
            print "Current version supports mixtures only in powers of 2. Training more mixtures."

        m = self.mix
        if m >= mix:
            self.__init__(self.dim)
            m = 1

        self.em (mfclist)
        if mix == 1:
            return

        while m < mix:
            self.doublemixtures()
            for i in range(3):
                self.em (mfclist)
            m *= 2
        for i in range(3):
            self.em (mfclist)

    ## Save the GMM
    def saveme (self, filename):
        import pickle
        with open (filename, 'w') as f:
            pickle.dump (self, f)
