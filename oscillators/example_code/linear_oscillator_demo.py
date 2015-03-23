import numpy as np
import matplotlib.pyplot as plt  
from scipy import integrate, misc, fftpack
from oscillators.oscillators import Oscillator

# Inputs 

a = .01 # damping / mass
omega0 =  1. # resonance pulsation
omegad = 1. # drive pulsation
def ep(x): return .5 * omega0**2 * x**2
def ddotxd(t):  return np.sin(omegad * t)
#def ddotxd(t):  return 0.*t
Td = 2. * np.pi / omegad
time    = np.linspace(0. ,200. * Td, 4096)

o = Oscillator(
  a = a,
  ep = ep,
  ddotxd = ddotxd,
  time = time,
  x_start = np.array([.0, 0.]))
o.solve()



fig = plt.figure(0)
plt.clf()

fig.add_subplot(3,1,1)
plt.title("Linear oscillator: $={0}$")
plt.plot(time, o.x, "b-", linewidth = 1.5)
#plt.xlabel("Time, $t$")
plt.ylabel("Position, $x$")
plt.grid()
fig.add_subplot(3,1,2)
plt.plot(time, o.dotx, "b-", linewidth = 1.5)
#plt.xlabel("Time, $t$")
plt.ylabel("Speed, $\dot x$")
plt.grid()
fig.add_subplot(3,1,3)
plt.plot(time, o.ddotx, "b-", linewidth = 1.5, label = "$\ddot x$")
#plt.plot(time, o.ddotxd(time), "r-", linewidth = 1.5, label = "$\ddot x_0$")
plt.xlabel("Time, $t$")
plt.ylabel("Acceleration")
plt.grid()
plt.legend()
#plt.show()
plt.savefig("linear_oscillator_signal.pdf")



