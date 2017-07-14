#!/usr/bin/env python

from __future__ import print_function

import sys

import numpy as np

#---------------------------------------------------------
# Simulation parameters
#---------------------------------------------------------
mass = 2000.0
nsteps = 10000
nskip = 5

#---------------------------------------------------------
# Avoided crossing parameters
#---------------------------------------------------------
def model(xx):
    out = np.zeros([len(xx),2,2])

    a, b, c, d = 0.01, 1.6, 0.005, 1.0
    v0 = np.copysign(a, xx) * (1.0 - np.exp(-b * np.abs(xx)))
    v01 = c * np.exp(-d * xx**2)
    out[:,0,0] = v0
    out[:,1,1] = -v0
    out[:,0,1] = v01
    out[:,1,0] = v01

    return out

#---------------------------------------------------------
# Absorbing boundary conditions
#---------------------------------------------------------
def absorber(xx, lbound, rbound, width):
    absorb = np.zeros(xx.shape)

    for i, x in enumerate(xx):
        if x > lbound and x < rbound:
            absorb[i] = 0.0
        else:
            c = 2.62
            if x > rbound:
                x -= rbound
            if x < lbound:
                x = -(x - lbound)
            x = (c + 1e-3) * x / width
            y = 4/(c-x)**2 + 4/(c+x)**2 - 8/c**2
            absorb[i] = y * 2.0 * np.pi**2 / width**2

    return absorb

#---------------------------------------------------------
# Initial Conditions
#---------------------------------------------------------
x0 = -10
p0 = 11.0
sg = 20/p0
def psi0(x): # gaussian centered at x0 with momentum k0
    out = np.zeros([len(x),2], dtype=np.complex128)
    out[:,0] = np.exp(-(x-x0)**2/sg + 1j * p0 * x)
    out /= np.linalg.norm(out)
    return out

xr = (-50.0, 50.0)
ngrid = 1 << 11
absorbingsteps = 1 << 7
dt = 5.0e-0
v = model # function for potential

#---------------------------------------------------------
# Internal parameters
#---------------------------------------------------------

def exponentiate(alpha, vherm, vabsorb):
    eigs, vecs = np.linalg.eigh(vherm)
    # add vabsorb to eigs
    facs = np.zeros(eigs.shape, dtype=np.complex128)
    for i in range(len(vabsorb)):
        facs[i,:] = eigs[i,:] - 1j * vabsorb[i]

    facs = np.exp(alpha * facs)
    out = np.zeros(vherm.shape, dtype=np.complex128)
    for i in range(vherm.shape[0]):
        out[i,:,:] = np.linalg.multi_dot( [vecs[i,:,:],
            np.diag(facs[i,:]),
            vecs[i,:,:].T.conj() ])
    return out

xx = np.linspace(xr[0], xr[1], ngrid)
dx = xx[1] - xx[0]

absorberwidth = (absorbingsteps - 1) * dx
absorberleft = xr[0] + absorberwidth
absorberright = xr[1] - absorberwidth

p = np.fft.fftfreq(ngrid, dx) * 2.0 * np.pi

potential = v(xx)
cpotential = absorber(xx, absorberleft, absorberright, absorberwidth)
kinetic = 0.5*p*p/mass

# 2 x 2 x ngrid
prop_v = exponentiate(-1j * dt * 0.5, potential, cpotential)

# 1 x ngrid
prop_t = np.exp(-1j * kinetic * dt)


def propagate(psi):
    '''Steps psi forward in place by one timestep using prop_v and prop_t'''
    for i in range(psi.shape[0]):
        psi[i,:] = np.dot(prop_v[i,:,:], psi[i,:])

    # transform to momentum space
    psi = np.fft.fft(psi, axis=0)

    for i in range(psi.shape[1]):
        psi[:,i] *= prop_t

    # transform to position space
    psi = np.fft.ifft(psi, axis=0)

    for i in range(psi.shape[0]):
        psi[i,:] = np.dot(prop_v[i,:,:], psi[i,:])

    return psi

psi = psi0(xx)

fmom = open("momentum.snaps", "w")

def print_snapshot(phi):
    print_density(phi, sys.stdout)
    print_momentum(phi, fmom)

def print_density(phi, fil):
    for ix, x in enumerate(xx):
        to_print = [ x, np.abs(phi[ix,0])**2, np.abs(phi[ix,1])**2, potential[ix,0,0], potential[ix,1,1] ]
        print(" ".join(["%12.8f" % j for j in to_print ]), file=fil)
    print(file=fil)
    print(file=fil)

def print_momentum(phi, fil):
    phik = np.fft.fft(phi, axis=0)
    for i in range(phik.shape[0]):
        print("%12.6f %12.6f" % (p[i], np.dot(phik[i,:].conj(), phik[i,:])), file=fil)
    print(file=fil)
    print(file=fil)

print_snapshot(psi)

for i in range(nsteps):
    psi  = propagate(psi)

    if i%nskip == 0:
        print_snapshot(psi)
