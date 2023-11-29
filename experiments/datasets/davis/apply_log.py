import numpy as np

y = np.loadtxt("affinity.txt")
log_y = - np.log10(y)
np.savetxt("log_affinity.txt", log_y)

print("Saved to log_affinity.txt")
