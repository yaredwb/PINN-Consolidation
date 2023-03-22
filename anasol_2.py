import numpy as np 
import matplotlib.pyplot as plt

cv = 0.1
Nz = 100
Nt = 101
H = 1
tmax = 1

u = np.zeros((Nz, Nt))
z = np.linspace(0, H, Nz)
t = np.linspace(0, 1, Nt)

# Initial condition (t = 0)
u[:,0] = 1

# Boundary conditions (t > 0)
u[0,1:] = u[-1,1:] = 0

for i in range(1, len(t)):
    sum = 0
    for n in range(1, 1000):
        term1 = 2 * (1 - (-1)**n) / (n * np.pi)
        term2 = np.sin(n * np.pi * z[1:-1] / H)
        term3 = np.exp(-cv * (n * np.pi / H)**2 * t[i])
        sum += (term1 * term2 * term3)
    u[1:-1,i] = sum

# Solution for drained top and undrained bottom
u2 = np.zeros((Nz, Nt))
z2 = np.linspace(0, H, Nz)
t2 = np.linspace(0, 1.0, Nt)

# Initial condition (t = 0)
u2[:,0] = 1

# Boundary conditions (t > 0)
u2[-1,1:] = 0

for i in range(1, len(t)):
    sum2 = 0
    for k in range(1, 1000):
        term4 = (-1)**(k-1) / (2 * k - 1)
        term5 = np.cos((2 * k - 1) * (np.pi * z[0:-1]) / (2 * H))
        term6 = np.exp(-(2 * k - 1)**2 * (np.pi**2 * cv * t2[i]) / (4 * H**2))
        sum2 += (term4 * term5 * term6)
    u2[0:-1,i] = (4 / np.pi) * sum2

np.savez_compressed('drained_top.npz', z=z2, t=t2[1:], u=u2[:,1:])

# Solution for drained top and bottom boundaries
u3 = np.zeros((Nz, Nt))
z3 = np.linspace(-H / 2.0, H / 2.0, Nz)
t3 = np.linspace(0, 1.0, Nt)

# Initial condition (t = 0)
u3[:,0] = 1

# Boundary conditions (t > 0)
u3[0,1:] = u3[-1,1:] = 0

for i in range(1, len(t3)):
    sum3 = 0
    for k in range(1, 1000):
        term7 = (-1)**(k-1) / (2 * k - 1)
        term8 = np.cos((2 * k - 1) * (np.pi * z3[1:-1]) / (1 * H))
        term9 = np.exp(-(2 * k - 1)**2 * (np.pi**2 * cv * t3[i]) / (1 * H**2))
        sum3 += (term7 * term8 * term9)
    u3[1:-1,i] = (4 / np.pi) * sum3

for i in range(len(t3)):
    plt.plot(u3[:,i], z3)

np.savez_compressed('drained_top_and_bottom.npz', z=z3, t=t3[1:], u=u3[:,1:])