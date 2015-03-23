import numpy as np
import matplotlib.pyplot as plt  
from scipy import integrate, misc, fftpack
from oscillators.oscillators import Oscillator

 
time    = np.linspace(0.0,10.,1024)
a = 1.
def ep(x): 
  y0 = .1
  x0 = .1
  k = 100.
  l0 = np.sqrt(x0**2 + y0**2)
  l  = np.sqrt(x**2 + y0**2)
  return .5 * k* (l -l0)**2 

def ddotxd(t):  return 2. * np.sin(2. * np.pi * 1. * t)
#def ddotxd(t):  return 0.*t

o = Oscillator(
  a = a,
  ep = ep,
  ddotxd = ddotxd,
  time = time,
  x_start = np.array([.1, 0.]))
o.solve()



fig = plt.figure(0)
plt.clf()
fig.add_subplot(3,1,1)
plt.plot(time, o.x, "b-", linewidth = 1.5)
#plt.xlabel("Time, $t$")
plt.ylabel("Position, $x$")
plt.grid()
fig.add_subplot(3,1,2)
plt.plot(time, o.dotx, "b-", linewidth = 1.5)
plt.xlabel("Time, $t$")
plt.ylabel("Speed, $\dot x$")
plt.grid()
fig.add_subplot(3,1,3)
plt.plot(time, o.ddotx, "b-", linewidth = 1.5, label = "$\ddot x$")
plt.plot(time, o.ddotxd(time), "r-", linewidth = 1.5, label = "\ddot x_0")
plt.xlabel("Time, $t$")
plt.ylabel("Acceleration")
plt.grid()
plt.legend()
#plt.show()
plt.savefig("oscillator_signal.pdf")

# Energy
pos = o.x
xE = np.linspace(pos.min(), pos.max(), 100)
E = np.array([ep(xx) for xx in xE])


fig = plt.figure(1)
plt.clf()
fig.add_subplot(2,1,1)
plt.plot(xE, E, "b-", linewidth = 1.5)
#plt.xlabel("Time, $t$")
plt.ylabel("Potential Energy, $E_p(x)$")
plt.grid()
fig.add_subplot(2,1,2)
plt.plot(pos, time, "b-", linewidth = 1.5)
plt.ylabel("Time, $t$")
plt.xlabel("Position, $x$")
plt.grid()
plt.savefig("oscillator_energy.pdf")

