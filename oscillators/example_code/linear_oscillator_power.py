import numpy as np
import matplotlib.pyplot as plt  
from scipy import integrate, misc, fftpack
from oscillators.oscillators import Oscillator, SSLO

# Inputs 

omega0 = 1.
a = np.array([0.01, 0.02, 0.05, 0.1, 0.2]) # damping / mass
#a = [1.]
omegad_sim = np.array([.1, .5, 1., 2., 5., 10.]) # drive pulsation
omegad_th = np.linspace(0.1, 10., 1000)
ep = lambda x: .5 * omega0**2 * x**2

colors = ["r", "g", "b", "c", "m", "k"]
markers = ["o", "s", "v", "p", "*", "^"]

fig = plt.figure(0)
plt.clf()

for i in xrange(len(a)):
  sslo = SSLO(a = a[i], omega0 = omega0)
  #p_th = .5 * a[i]  * omegad_th**2 / ( a[i]**2 * omegad_th*2 + (omega0**2 - omegad_th**2 )**2 )
  p_th = sslo.pa(Ad = omegad_th**-2, omegad = omegad_th )
  plt.plot(omegad_th, p_th, "-" + colors[i], label = "$a = {0}$ (th.)".format(a[i]))
  p_sim = np.zeros_like(omegad_sim)
  
  for j in xrange(len(p_sim)):
    Td = 2. * np.pi / omegad_sim[j]
    time = np.linspace(0. ,200. *Td ,4096)
    def ddotxd(t):  return np.sin(omegad_sim[j] * t) 
    o = Oscillator(
      a = a[i],
      ep = ep,
      ddotxd = ddotxd,
      time = time,
      x_start = np.array([0., 0.]))
    o.solve()
    
    p_sim[j] = o.average_power(start_time = .75 * time.max())
  plt.plot(omegad_sim, p_sim, markers[i]+ colors[i], label = "$a = {0}$ (simu.)".format(a[i]))
  
plt.xlabel("Relative pulsation, $\omega_d / \omega_0$")
plt.yscale("log")
#plt.xscale("log")
plt.ylabel(r"Power vs. mass vs. acceleration, $\bar p_a / C_d^2$ [Ws$^2$/m]")
plt.legend(ncol =2)
plt.grid()
#plt.show()
plt.savefig("linear_oscillator_power.pdf")


