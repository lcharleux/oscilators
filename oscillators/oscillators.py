# OSCILLATORS
# ludovic.charleux@univ-savoie.fr 03/2015

import time
import numpy as np
import matplotlib.pyplot as plt  
from scipy import integrate, misc, fftpack

"""
def rolling_average(y, dx):
  out
"""

def integrate_func(func, x):
  """
  Integrates a function on each value of x 
  """
  out = np.zeros_like(x)
  for i in xrange(1, len(x)):
    out[i] = out[i-1] + integrate.quad(func, x[i-1], x[i])[0]
  return out
  
  
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
    return Ad * omegad**2 / ( self.a**2 * omegad**2 + (self.omega0**2 - omegad**2 )**2 )**.5
    
  def pa(self, Ad = 1., omegad = 1.):
    """
    Returns the average power extracted by the damping from the oscillator
    """
    return self.a * omegad**2 * self.A(Ad, omegad)**2 / 2.
       
      
          

class Oscillator(object):
  """
  Solves: d2x/dt2 + a * dx/dt + dep/dx = d2xd/dt
  """
  def __init__(self, a, ep_func, ddotxd_func, dt = .01, nt = 100 , X0 = np.array([0., 0.]), dx = 1.e-6, tstart = 0., steady_state_criterion = 0.001):
    self.a = a # damping / mass
    self.ep_func = ep_func # potential energy / mass
    self.ddotxd_func = ddotxd_func
    self.dt = dt
    self.nt = nt
    self.dx = dx
    self.time = np.array([tstart])
    self.X = np.array([X0]) 
    self.steady_state_criterion = steady_state_criterion
    
  
  def derivative(self, x, t):
    f = misc.derivative(self.ep_func, x[0], dx = self.dx) 
    a = self.a
    ddotxd = self.ddotxd_func(t)
    return np.array([x[1], -a * x[1] -f -ddotxd ])
  
  def solve(self, append = True):
    if append == False: self.reset()
    tstart = self.time.max()
    nt = self.nt  
    dt = self.dt
    duration = dt * nt    
    new_time = np.linspace(0., duration, nt + 1) + tstart
    X = integrate.odeint( self.derivative, self.X[-1], new_time)
    self.time = np.concatenate((self.time, new_time[1:]))
    self.X    = np.concatenate((self.X, X[1:]))
             
  
  def __repr__(self):
    pattern = "<Oscillator: {0} iters. computed>"
    return pattern.format(self.niter())
  
  def __str__(self):
    pattern = "Oscillator:\n* Iterations computed: {0}\n* Steady state reached: {1} (criterion = {2})"
    return pattern.format(self.niter(),  self.steady_state(), self.steady_state_criterion)
  
  def reset(self):
    self.time = np.array([])
    del self.X  
  
  def get_x(self): return self.X[:,0]
  x = property(get_x)
  
  def get_dotx(self): return self.X[:,1]
  dotx = property(get_dotx)
  
  def get_ddotx(self): 
    n = len(self.time)
    deriv_X = np.zeros(n)
    for i in xrange(n):
      deriv_X[i] = self.derivative(X[i], self.time[i])[1]
    return deriv_X
  ddotx = property(get_ddotx)
    
  def get_ec(self): return .5 * self.dotx**2 
  ec = property(get_ec)
  
  def get_ep(self): return self.ep_func(self.x)
  ep = property(get_ep)
  
  def get_e(self):  return self.ec + self.ep
  e = property(get_e)
  
  def get_pa(self): return self.dotx**2 * self.a
  pa = property(get_pa)
  
  def get_pa_av(self):
    pa = self.pa
    return pa[-self.nt:].mean()
  pa_av = property(get_pa_av)
    
  
  def get_ph(self): return -self.ddotxd * self.dotx  
  ph = property(get_ph)  
 
  def get_ddotxd(self): return self.ddotxd_func(self.time)
  ddotxd = property(get_ddotxd)  
 
  def get_dotxd(self): return integrate_func(self.ddotxd_func, self.time)
  dotxd = property(get_dotxd)  
    
  def niter(self):
    return len(self.time) -1
   
  def steady_state_error(self, criterion = None):
    if criterion != None: self.steady_state_criterion = criterion
    nt = self.nt
    dt = self.dt
    time = self.time[-nt:]
    time = time - time.min()
    pa = self.pa[-nt:]
    duration = time.max()
    loc0 = np.where(time <= duration /2.)
    loc1 = np.where(time >= duration /2.)
    pa0 = pa[loc0].mean()
    pa1 = pa[loc1].mean()
    error = (abs((pa1 - pa0) / pa0)) 
    ta = self.time.max() - duration / 2
    niter = len(self.time - 1)
    return (abs((pa1 - pa0) / pa0), ta, niter)
  
    
  def steady_state(self, criterion = None):
    if criterion != None: self.steady_state_criterion = criterion  
    return self.steady_state_error(criterion = criterion)[0] <= self.steady_state_criterion
    
  def find_steady_state(self, criterion = None, maxiter = 100):
    if criterion != None: self.steady_state_criterion = criterion
    count = 0
    t0 = time.time()
    while True:
      self.solve()
      if self.steady_state(): break
      count += 1
      if count >= maxiter: 
        print "Maxiter exceeded"
        break
    t1 = time.time()
    print "<Oscillator: converged in {0:.2e}>".format(t1-t0)  
        

     
 
  
  
       
  

