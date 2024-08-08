import numpy as np
import matplotlib.pyplot as plt

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L

n_epoch = 1000
beta_np_cyc = frange_cycle_linear(0.0, 1.0, 500, 10, 1)
plt.figure(figsize=(10, 6))
plt.plot(beta_np_cyc, label='Beta Value')
plt.title('Cycle Linear Schedule')
plt.xlabel('Epoch')
plt.ylabel('Beta')
plt.legend()
plt.grid(True)
plt.show()
