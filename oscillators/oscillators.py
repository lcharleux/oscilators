# OSCILLATORS
# ludovic.charleux@univ-savoie.fr 03/2015


import numpy as np
import matplotlib.pyplot as plt  
from scipy import integrate, misc, fftpack

    

class Oscillator(object):
  """
  Solves: d2x/dt2 + a * dx/dt + dep/dx = d2x0/dt
  """
  def __init__(self, mass, dampening, energy, drive, time, x_start = np.array([0., 0.]), dx = 1.e-6):
    self.mass = mass
    self.dampening = dampening
    self.energy = energy
    self.drive = drive
    self.time = time
    self.x_start = x_start 
    self.dx = dx
  
  def derivative(self, x, t):
    #F = x[0]
    F = misc.derivative(self.energy, x[0], dx = self.dx) 
    M = self.mass
    Mu = self.dampening
    D = self.drive(t)
    return np.array([
      x[1], 
      - Mu/M * x[1] - F/M + D ])
  
  def solve(self):
    X = integrate.odeint(
      self.derivative,
      self.x_start,
      self.time)     
    self.position = X[:,0]
    self.speed = X[:,1]
    n = len(self.time)
    deriv_X = np.zeros(n)
    for i in xrange(n):
      deriv_X[i] = self.derivative(X[i], self.time[i])[1]
    self.acceleration = deriv_X  



