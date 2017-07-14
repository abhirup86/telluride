#!/usr/bin/env python

from __future__ import print_function

import numpy as np

#---------------------------------------------------------
# Simulation parameters
#---------------------------------------------------------
mass = 1.0
nsteps = 100000
nskip = 100

#---------------------------------------------------------
# Harmonic Oscillator parameters
#---------------------------------------------------------
k0 = 0.5
omega = np.sqrt(2.0 * k0 / mass)
sg = 2.0 / (mass * omega)

def harmonic(x):
    return k0 * x * x

#---------------------------------------------------------
# Square well parameters
#---------------------------------------------------------
v0 = 13.0
squarewidth = 1.0
def square(x):
    out = np.zeros(xx.shape)
    for i, x in enumerate(xx):
        if abs(x) < squarewidth * 0.5:
            out[i] = v0
        else:
            out[i] = 0.0

    return out

#---------------------------------------------------------
# Absorbing boundary conditions
#---------------------------------------------------------
def absorber(xx, lbound, rbound, width):
    out = np.zeros(xx.shape)

    for i, x in enumerate(xx):
        if x > lbound and x < rbound:
            out[i] = 0.0
        else:
            c = 2.62
            if x > rbound:
                x -= rbound
            if x < lbound:
                x = -(x - lbound)
            x = (c + 1e-3) * x / width
            y = 4/(c-x)**2 + 4/(c+x)**2 - 8/c**2
            out[i] = y * 2.0 * np.pi**2 / width**2

    return out

#---------------------------------------------------------
# Initial Conditions
#---------------------------------------------------------
x0 = -10
p0 = 5.0
sg = 20/p0
def psi0(x): # gaussian centered at x0 with momentum k0
    out = np.zeros(x.shape, dtype=np.complex128)
    out[:] = np.exp(-(x-x0)**2/sg + 1j * p0 * x)
    out /= np.linalg.norm(out)
    return out

xr = (-25.0, 25.0)
ngrid = 1 << 10
absorbingsteps = 1 << 6
dt = 5.0e-4
v = square # function for potential

#---------------------------------------------------------
# Internal parameters
#---------------------------------------------------------

xx = np.linspace(xr[0], xr[1], ngrid)
dx = xx[1] - xx[0]

absorberwidth = (absorbingsteps - 1) * dx
absorberleft = xr[0] + absorberwidth
absorberright = xr[1] - absorberwidth

p = np.fft.fftfreq(ngrid, dx) * 2.0 * np.pi

potential = v(xx)
cpotential = -1j * absorber(xx, absorberleft, absorberright, absorberwidth)
kinetic = 0.5*p*p/mass

prop_v = np.exp(-1j * (potential + cpotential) * dt * 0.5)
prop_t = np.exp(-1j * kinetic * dt)

psi = psi0(xx)

def print_snapshot(psi):
    for x, rho, v in zip(xx, np.abs(psi)**2, potential):
        print("%12.8f %12.8f %12.8f" % (x, rho, v))
    print()
    print()


print_snapshot(psi)

for i in range(nsteps):
    psi *= prop_v
    psi = np.fft.fft(psi)
    psi *= prop_t
    psi = np.fft.ifft(psi)
    psi *= prop_v

    if (dt.imag != 0.0):
        psi /= np.linalg.norm(psi)

    if i%nskip == 0:
        print_snapshot(psi)
