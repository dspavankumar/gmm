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
from iofile import *
from gmm import GMM
from multiprocessing import Process, Queue, cpu_count
import pickle
import time

class SPLICE ():
    ## Initialisation through GMM object
    def __init__ (self, gmmfile):
        """Initialisation through GMM object"""
        with open(gmmfile, 'r') as f:
            self.gmm = pickle.load (f)

        if not isinstance(self.gmm, GMM):
            raise ValueError ('Initialise with a GMM object file')

    ## Training step 1 of 3: Initialise statistics to zeros
    def __init_stats__ (self):
        """Initialises statistics computation"""
        self.__numer__      = np.zeros((self.gmm.mix, self.gmm.dim, self.gmm.dim + 1))
        self.__denom__      = np.zeros((self.gmm.mix, self.gmm.dim + 1, self.gmm.dim + 1))
        self.W          = np.zeros((self.gmm.mix, self.gmm.dim, self.gmm.dim + 1))

    ## Training step 2 of 3: Accumulate statistics
    def __update_worker__ (self, spairs, Q):
        """Updates statistics - worker routine"""
        numer      = np.zeros((self.gmm.mix, self.gmm.dim, self.gmm.dim + 1))
        denom      = np.zeros((self.gmm.mix, self.gmm.dim + 1, self.gmm.dim + 1))
        
        for spair in spairs:
            nfile, cfile = spair.split()
            nfeats       = readfile(nfile)
            cfeats       = readfile(cfile)
        
            if nfeats.shape != cfeats.shape:
                raise ValueError('Sizes of clean and noisy features mismatch')
            nfeats = np.column_stack ((np.ones((nfeats.shape[0],1)), nfeats))

            for nf,cf in zip(nfeats, cfeats):
                gam         = self.gmm.posterior(nf[1:])
                numer       += np.outer(gam, np.outer(cf,nf)).reshape((self.gmm.mix, self.gmm.dim, self.gmm.dim + 1))
                denom       += np.outer(gam, np.outer(nf,nf)).reshape((self.gmm.mix, self.gmm.dim + 1, self.gmm.dim + 1))

        Q.put([numer, denom])

    ## Training step 3 of 3: Compute parameters
    def __finish_stats__ (self):
        """Estimate the SPLICE matrices from the accumulated statistics"""
        for mix in range(self.gmm.mix):
            self.W[mix] = np.dot (self.__numer__[mix], np.linalg.inv(self.__denom__[mix]))

    ## Train SPLICE matrices
    def train (self, stereolist, threads=cpu_count()):
        """Trains SPLICE matrices - wrapper"""
        self.__init_stats__()
        
        with open(stereolist, 'r') as f:
            spairs = f.read().splitlines()

        Q = Queue()
        processes = []

        for thr in xrange(threads):
            p = Process (target=self.__update_worker__, args=(spairs[thr::threads], Q))
            p.start()
            processes.append(p)

        while Q.qsize < threads:
            time.sleep(0.01)
    
        for thr in xrange(threads):
            numer, denom = Q.get()
            self.__numer__  += numer
            self.__denom__  += denom
        
        self.__finish_stats__()

    ## SPLICE compensation on test features
    def __scompensate_worker__ (self, ncpairs, Q):
        """Comensates test features using SPLICE - worker routine"""
        for pair in ncpairs:
            nfile, cfile = pair.split()
            nfeats = readfile(nfile)
            cfeats = np.zeros(nfeats.shape)
            nfeats = np.column_stack ((np.ones((nfeats.shape[0],1)), nfeats))

            for i in range(nfeats.shape[0]):
                gam = self.gmm.posterior(nfeats[i][1:])
                for mix in range(self.gmm.mix):
                    cfeats[i] += gam[mix] * np.dot(self.W[mix], nfeats[i])
            writefile (cfile, cfeats)
        Q.put(0)

    ## Apply SPLICE compensation on test features
    def apply (self, nclist, threads=cpu_count()):
        """Applies SPLICE compensation on test features"""
        with open(nclist, 'r') as f:
            ncpairs = f.read().splitlines()

        Q = Queue()
        processes = []
        for thr in xrange(threads):
            p = Process (target=self.__scompensate_worker__, args=(ncpairs[thr::threads], Q))
            p.start()
            processes.append(p)

        while Q.qsize < threads:
            time.sleep(0.01)

    ## Save the SPLICE object
    def saveas (self, filename):
        """Saves the SPLICE object"""
        import pickle
        with open (filename, 'w') as f:
            pickle.dump (self, f)

