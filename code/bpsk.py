import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
current_path = os.path.dirname(os.path.abspath(__file__))

"""
BER for BPSK in Rayleigh channel
y = hx + n
y: recived symbol
h: Rayleigh multipath channel, var=1
x: BPSK symbol(+1, -1)
n: Gaussian Noise
"""

# number of BPSK symbol
n = 1_000

# SNR values
SNR = np.linspace(-3,35,100)

# generating BPSK symbol
def generate_BPSK(len_symbol):
    return np.random.randint(2, size=len_symbol) * 2 - 1

# Rayleigh Channel
def empiricall_rayleigh_channel_BER(SNR, N):
    """
    SNR: Eb/No
    N: number of symbol
    return: n_error(number of errors)
    """
    x = generate_BPSK(N)
    n_error = []

    for i in tqdm(range(1,len(SNR))):
        n = (1/np.sqrt(2)) * (np.random.randn(1,N) + 1j * np.random.randn(1,N)) # white gaussian noise, 0dB variance
        h = (1/np.sqrt(2)) * (np.random.randn(1,N) + 1j * np.random.randn(1,N)) # Rayleigh channel

        # Rx
        y = (h*x) + (10**(SNR[i]/20))*n

        # equalization
        y_hat = y / h

        # recevier - hard decoding
        y_ = np.real(y_hat) > 0
        print(y_)
        decoding = lambda x_: 1 if x_ == True else -1
        decode_y = np.array([decoding(tmp_y) for tmp_y in y_])
        print(f'decode: {decode_y}')

        # counting the errors
        n_error.append(list(decode_y != x).count(True))
    
    return n_error

print(empiricall_rayleigh_channel_BER(SNR, n))