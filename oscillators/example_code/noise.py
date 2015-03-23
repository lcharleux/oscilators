import numpy as np
import matplotlib.pyplot as plt  
from scipy import integrate, misc, fftpack

class Noise(object):
  def __init__(self, amplitude = 1., band = [10., 500.]):
    self.amplitude = amplitude
    self.band = band
  
  def spectrum(self, frequency):
    N2 = len(frequency)
    modulus = self.amplitude * (frequency > self.band[0]) * (frequency <= self.band[1])
    #angle = 2 * np.pi * np.random.rand(N2) 
    angle = np.zeros(N2)
    Apos = modulus * (np.cos(angle) + 1j * np.sin(angle))
    Aneg = np.conjugate(Apos[::-1])
    A = np.concatenate([Apos, Aneg])
    return A
    
  def make(self, time):
    N = len(time)
    fe = (N+1) / (time[-1] - time[0])
    frequency = np.linspace(0., fe/2., N/2)
    A = self.spectrum(frequency)
    a = (fftpack.ifft(A)).real
    return a
    

time = np.linspace(0., 10., 2**16)
N = len(time)
fe = (N+1) / (time[-1] - time[0])
frequency = np.linspace(0., fe/2., N/2)
noise = Noise()
drive_spectrum = noise.spectrum(frequency)
drive_amp = noise.make(time)

fig = plt.figure(0)
plt.clf()
fig.add_subplot(1,1,1)
plt.plot(time, abs(drive_amp), "b-", linewidth = 1.5)
plt.xlabel("Time, $t$")
plt.ylabel("Amplitude, $a$")
plt.grid()
plt.savefig("noise_amp.pdf")



fig = plt.figure(0)
plt.clf()
fig.add_subplot(2,1,1)
plt.plot(frequency, abs(drive_spectrum)[0:1024], "b-", linewidth = 1.5)
plt.xlabel("Time, $t$")
plt.ylabel("Amplitude, $a$")
plt.grid()
fig.add_subplot(2,1,2)
plt.plot(frequency, np.angle(drive)[0:1024], "b-", linewidth = 1.5)
plt.xlabel("Time, $t$")
plt.ylabel("Amplitude, $a$")
plt.grid()
plt.savefig("noise_spectrum.pdf")
    
