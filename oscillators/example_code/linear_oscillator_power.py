import numpy as np
import matplotlib.pyplot as plt  
from scipy import integrate, misc, fftpack
from oscillators.oscillators import Oscillator, SSLO

# Inputs 

omega0 = 1.
a = np.array([0.01, 0.02, 0.1, 1., 5.]) # damping / mass
#a = [1.]
omegad_sim = np.array([.1, .5, 0.75, 1., 1.25, 1.5, 2., 5., 10.]) # drive pulsation
omegad_th = np.concatenate(( np.linspace(0.1, 1., 1000) , np.linspace(1., 10., 1000) ))
ep_func = lambda x: .5 * omega0**2 * x**2

colors = ["r", "g", "b", "c", "m", "y", "k"]
markers = ["o", "s", "v", "p", "*", "^", "<"]

fig = plt.figure(0)
plt.clf()

for i in xrange(len(a)):
  sslo = SSLO(a = a[i], omega0 = omega0)
  p_th = sslo.pa(Ad = omegad_th**-2, omegad = omegad_th )
  plt.plot(omegad_th, p_th, "-" + colors[i], label = "$a = {0}$ (th.)".format(a[i]))
  p_sim = np.zeros_like(omegad_sim)
  
  for j in xrange(len(p_sim)):
    def ddotxd_func(t):  return np.sin(omegad_sim[j] * t) 
    o = Oscillator(
      a = a[i],
      ep_func = ep_func,
      ddotxd_func = ddotxd_func,
      dt = 2. * np.pi / omegad_sim[i] / 16.,
      nt = 512,
      X0 = np.array([0., 0.])
      )
    o.find_steady_state(criterion = 1.e-3, maxiter = 50)
    
    p_sim[j] = o.pa_av
  plt.plot(omegad_sim, p_sim, markers[i]+ colors[i], label = "$a = {0}$ (simu.)".format(a[i]))
  
plt.xlabel("Relative pulsation, $\omega_d / \omega_0$")
plt.yscale("log")
#plt.xscale("log")
plt.ylabel(r"Power vs. mass vs. acceleration, $\bar p_a / C_d^2$ [Ws$^4$m$^{2-}$kg$^{-1}$]")
plt.legend(ncol =2)
plt.grid()
#plt.show()
plt.savefig("linear_oscillator_power.pdf")


