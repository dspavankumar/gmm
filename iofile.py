#!/usr/bin/python

## ----------------------------------------------------------------------------
##    Functions for reading and writing text, HTK and Sphinx format files
##    Copyright (C) 2014,  D S Pavan Kumar
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

## IMPORTANT: Set I/O format below. Options: txt htk sphinx
ioformat = 'htk'

## ----------------------------------- Reading and writing files --------------------------------------
def readfile (mfcfile):
    if ioformat == 'txt':
        return np.loadtxt(mfcfile)
    elif ioformat == 'htk':
        return readhtk(mfcfile)
    elif ioformat == 'sphinx':
        return readsph(mfcfile)
    else:
        raise TypeError('Unknown format. Please use one of: txt htk sphinx')

def writefile (mfcfile, data):
    if ioformat == 'txt':
        np.savetxt(mfcfile, data)
    elif ioformat == 'htk':
        writehtk(mfcfile, data, 13, 100000, 8198)
    elif ioformat == 'sphinx':
        writesph(mfcfile, data)
    else:
        raise TypeError('Unknown format. Please use one of: txt htk sphinx')


## -------------------------------------------- HTK formats -------------------------------------------
def readhtk (mfcfile):
    with open (mfcfile, 'r') as f:
        num_frames = np.fromfile (f, dtype=np.int32, count=1)
        frame_rate = np.fromfile (f, dtype=np.int32, count=1)
        byte_count = np.fromfile (f, dtype=np.int16, count=1)
        htype_code = np.fromfile (f, dtype=np.int16, count=1)
        data = np.fromfile(f, dtype=np.float32)
    return np.reshape(data, (num_frames, byte_count/4))

def writehtk (mfcfile, data, cep_count=13, frame_rate=100000, htype_code=8198):
    num_samples = len(data)
    num_frames = num_samples / cep_count
    byte_count = 4 * cep_count
    with open (mfcfile, 'w') as f:
        np.array(num_frames).astype(np.int32).tofile(f)
        np.array(frame_rate).astype(np.int32).tofile(f)
        np.array(byte_count).astype(np.int16).tofile(f)
        np.array(htype_code).astype(np.int16).tofile(f)
        data.astype(np.float32).tofile(f)

## ------------------------------------------ Sphinx formats ------------------------------------------
def readsph (mfcfile, cep_count=13):
    with open (mfcfile, 'r') as f:
        num_samples = np.fromfile (f, dtype=np.int32, count=1)
        data = np.fromfile(f, dtype=np.float32)
    return np.reshape(data, (len(data)/cep_count, cep_count))

def writesph (mfcfile, data):
    num_samples = len(data)
    with open (mfcfile, 'w') as f:
        np.array(num_samples).astype(np.int32).tofile(f)
        data.astype(np.float32).tofile(f)
