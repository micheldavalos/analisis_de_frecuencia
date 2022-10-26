import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reference: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html

data = pd.read_csv('diametro.csv')
N = len(data['T'])

t = np.linspace(0, 10, N)
X = np.fft.fft(data['D'])
X_mag = np.abs(X)

fig, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
fig.suptitle('Closed-Loop (Diámetro)')
ax1.plot(data['T'],  data['D'], '.-')
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('Diámetro')
fig, ax2 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
fig.suptitle('FFT')
ax2.stem(t[1:], X_mag[1:], 'b',
         markerfmt=" ", basefmt="-b")
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Amplitud')
fig, ax3 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
fig.suptitle('Inversa FFT')
ax3.plot(data['T'], np.fft.ifft(X), '.-')
ax3.set_xlabel('Frecuencia (Hz)')
ax3.set_ylabel('Amplitud')

plt.show()
