import numpy as np
import json

def main():
    with open('wifi/Слинько.dat') as wifi:
        lines = wifi.readlines()
        signal = [((float(x)/2 + 1)) for x in lines ]
    code = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1], dtype = np.int8)
    code = np.repeat(code, 5)
    conv = np.convolve(signal, code[::-1], mode = 'same')
    low = -20
    high = 10
    #Значения получены эмпирически, с графика
    bit = []
    for i in range(1,conv.size-1):
            if conv[i] < low and conv[i-1] > conv[i] and conv[i+1] > conv[i]:
                bit.append(0)
            if conv[i] > high  and conv[i-1] < conv[i] and conv[i+1] < conv[i]:
                bit.append(1)
    bits = np.packbits(np.array(bit, dtype=int))
    _bytes = bits.tobytes()
    _ascii = _bytes.decode('ascii')
    d = {"message": _ascii}
    with open('wifi.json', 'w') as f:
        json.dump(d, f)

if __name__ == '__main__':
    main()