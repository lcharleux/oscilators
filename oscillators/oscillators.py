# OSCILLATORS
# ludovic.charleux@univ-savoie.fr 03/2015


import numpy as np
import matplotlib.pyplot as plt  
from scipy import integrate, misc, fftpack


class SSLO(object):
  """
  Steady State Linear Oscillator
  """
  
  def __init__(self, omega0 = 1., a = 1.,):
    self.omega0 = omega0
    self.a = a
  
  def A(self, Ad = 1., omegad = 1.):
    """
    Returns the amplitude of the oscillation in a steady state regime.
    """
    return Ad * omegad**2 / ( self.a**2 * omegad*2 + (self.omega0**2 - omegad**2 )**2 )**.5
    
  def pa(self, Ad = 1., omegad = 1.):
    """
    Returns the average power extracted by the damping from the oscillator
    """
    return self.a * omegad**2 * self.A(Ad, omegad)**2 / 2.
       
      
          

class Oscillator(object):
  """
  Solves: d2x/dt2 + a * dx/dt + dep/dx = d2xd/dt
  """
  def __init__(self, a, ep, ddotxd, time, x_start = np.array([0., 0.]), dx = 1.e-6):
    self.a = a # damping / mass
    self.ep = ep # potential energy / mass
    self.ddotxd = ddotxd
    self.time = time
    self.x_start = x_start 
    self.dx = dx
  
  def derivative(self, x, t):
    f = misc.derivative(self.ep, x[0], dx = self.dx) 
    a = self.a
    ddotxd = self.ddotxd(t)
    return np.array([
      x[1], 
      - a * x[1] - f - ddotxd ])
  
  def solve(self):
    X = integrate.odeint(
      self.derivative,
      self.x_start,
      self.time)     
    self.x = X[:,0]
    self.dotx = X[:,1]
    n = len(self.time)
    deriv_X = np.zeros(n)
    for i in xrange(n):
      deriv_X[i] = self.derivative(X[i], self.time[i])[1]
    self.ddotx = deriv_X  
  
  def average_power(self, start_time  = 0.):
    t = self.time
    loc = np.where(t>=start_time)
    t = t[loc]
    s = self.dotx[loc]
    p = self.a * integrate.simps(s**2, t) / (t[-1]-t[0])
    return p
     
  

