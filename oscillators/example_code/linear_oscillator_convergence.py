import numpy as np
import matplotlib.pyplot as plt  
from scipy import integrate, misc, fftpack, ndimage
from oscillators.oscillators import Oscillator
# Inputs 

a = .01 # damping / mass
f0 = 1.
fd = 10.
omega0 = 2. * np.pi * f0 # resonance pulsation
omegad = 2. * np.pi * fd # drive pulsation
Td = 1. / fd
def ep_func(x): return .5 * omega0**2 * x**2
def ddotxd_func(t):  return np.cos(omegad * t)
#def ddotxd_func(t):  return 0.*t
Td = 2. * np.pi / omegad
dt = Td / 16.
nt = 1024


o = Oscillator(
  a = a,
  ep_func = ep_func,
  ddotxd_func = ddotxd_func,
  dt = dt,
  nt = nt,
  X0 = np.array([0., 0.]))

o.find_steady_state(criterion = 1.e-4, maxiter = 50)

filter_core = Td * 20.

fig = plt.figure(0)
plt.clf()

fig.add_subplot(3,1,1)
plt.title("Linear oscillator: $\omega_d/\omega_0={0}$, $a = {1}$".format(omegad / omega0, a))
plt.plot(o.time, o.x, "b-", linewidth = 1.5)
#plt.xlabel("Time, $t$")
plt.ylabel("Position, $x$")
plt.grid()
fig.add_subplot(3,1,2)
#plt.plot(o.time, o.pa, "b-", linewidth = .5, label = "$p_a$")
#plt.plot(o.time, o.ph, "r-", linewidth = .5, label = "$p_h$")
plt.plot(o.time, ndimage.gaussian_filter(o.pa, filter_core), "b-", linewidth = 2., label = r"$\bar p_a$")
plt.plot(o.time, ndimage.gaussian_filter(o.ph, filter_core), "r-", linewidth = 2., label = r"$\bar p_h$")
#plt.xlabel("Time, $t$")
#plt.xlabel("Time, $t$")
plt.ylabel("Power, $p$")
plt.legend(loc = "upper left", ncol = 2)
plt.grid()
fig.add_subplot(3,1,3)
plt.plot(o.time, o.ep, "r-", linewidth = 1., label = "$e_p$")
plt.plot(o.time, o.ec, "g-", linewidth = 1., label = "$e_c$" )
plt.plot(o.time, o.e, "b-", linewidth = 1., label = "$e$")
plt.plot(o.time, ndimage.gaussian_filter(o.e, filter_core), "b--", linewidth = 2., label = r"$\bar e$")
plt.xlabel("Time, $t$")
plt.ylabel("Energy, $e$")
plt.grid()
plt.legend(loc = "upper left", ncol = 2)
plt.show()
#plt.savefig("linear_oscillator_energy.pdf")



