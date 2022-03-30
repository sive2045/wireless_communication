import numpy as np
import multiprocessing
from multiprocessing.dummy import freeze_support
import parmap
import matplotlib.pyplot as plt
from scipy import special

num_cores = multiprocessing.cpu_count() 

"""
BER for BPSK in Rayleigh channel
y = hx + n
y: recived symbol
h: Rayleigh multipath channel, var=1
x: BPSK symbol(+1, -1)
n: Gaussian Noise

REF: http://www.dsplog.com/2008/08/10/ber-bpsk-rayleigh-channel/
"""

# number of BPSK symbol
n = 1_000_000

# SNR[dB] values
SNR = np.linspace(-3,35,100)

# generating BPSK symbol
def generate_BPSK(len_symbol):
    return np.random.randint(2, size=len_symbol) * 2 - 1

# Rayleigh Channel
def empirical_rayleigh_channel_BER(SNR, N):
    """
    SNR: Eb/No
    N: number of symbol
    return: n_error(number of errors)
    """
    x = generate_BPSK(N)
    n_error = []

    for i in range(0,len(SNR)):
        n = (1/np.sqrt(2)) * (np.random.randn(1,N) + 1j * np.random.randn(1,N)) # white gaussian noise, 0dB variance
        h = (1/np.sqrt(2)) * (np.random.randn(1,N) + 1j * np.random.randn(1,N)) # Rayleigh channel

        # Rx
        y = (h*x) + (10**(-SNR[i]/20))*n

        # equalization
        y_hat = y / h

        # recevier - hard Decision
        y_ = (np.real(y_hat) > 0).reshape((N,))
        decoding = lambda x_: 1 if x_ else -1
        decode_y = np.array([decoding(tmp_y) for tmp_y in y_])
        #print(f'decode: {decode_y}')

        # counting the errors
        n_error.append(list(decode_y != x).count(True))
    
    return n_error

def plot_figure(SNR, N, num_cores):
    """
    SNR: Eb/No
    N: number of symbol
    """
    #############################################################################################################
    # Multiprocessing for empirical rayleigh channel

    splited_data = np.array_split(SNR, num_cores)
    n_error = parmap.map(empirical_rayleigh_channel_BER, splited_data, N, pm_pbar=True, pm_processes=num_cores)
    n_error = np.array([item for l in n_error for item in l])
    
    #############################################################################################################

    empirical_BER = n_error / N
    theory_AWGN_BER = 0.5 * special.erfc(np.sqrt(10**(SNR/10)))
    SNR_ = 10**(SNR/10)
    thoery_rayleigh_BER = 0.5 * ( 1 - np.sqrt(SNR_/(SNR_ + 1)) )

    # plot
    plt.plot(SNR,theory_AWGN_BER, '-.g', label='AWGN-Theory')
    plt.plot(SNR, empirical_BER, '-XC4', label='Rayleigh-Empirical')
    plt.plot(SNR, thoery_rayleigh_BER, '-1C5', label='Ralyleigh-Thoery')
    plt.semilogy(); plt.xlim(-3,35); plt.ylim(1e-5, 0.5); plt.grid(); plt.legend()
    plt.ylabel('Bit Error Rate'); plt.xlabel('SNR[dB]')
    plt.title("BER for BPSK in Rayleigh channel")
    plt.show()
    

if __name__ == '__main__':
    freeze_support()
    plot_figure(SNR, n, num_cores)