import numpy as np
import matplotlib.pyplot as plt
import json

def main():
    noise = np.fromfile('pulsar.dat')
    f = np.fft.fft(noise)
    for i in range (f.size):
        if abs(f[i].real) < 0.5e24:
            f[i] = 0
    #убрали слабый шум
    f[:5000] = 0
    f[-5000:] = 0
    #убрали особо малые частоты
    signal = np.fft.ifft(f)
    plt.plot(signal[:100].real)
    plt.savefig('pulsar.png')
    #картинка есть, ищем период
    peak = 0
    up = True
    got_l = False
    got_r = False
    for i in range(50):
        if signal[i] > 1e19 and up:
            peak +=1
            up = False
        if signal[i] < 0:
            up = True
        if peak == 1 and not got_l:
            left = i
            got_l = True
        if peak == 3 and not got_r:
            right = i
            got_r = True
            break
    T = right - left
    d = {"period": T}
    with open('pulsar.json', 'w') as f:
        json.dump(d, f)
    
if __name__ == '__main__':
    main()