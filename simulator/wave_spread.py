import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1) Define a minimal JONSWAP spectral function
# ---------------------------------------------------
def jonswap_spectrum(omega, Hs, Tp, gamma=3.3, g=9.81):
    """
    Returns S_eta(omega): wave elevation spectrum [m^2*s] (simplified JONSWAP).
    
    Parameters:
    -----------
    omega : float or np.array
        Angular frequency [rad/s].
    Hs : float
        Significant wave height [m].
    Tp : float
        Peak period [s].
    gamma : float
        Peak enhancement factor (~3.3 typically).
    g : float
        Gravity [m/s^2].
    """
    # Peak freq
    wp = 2.0 * np.pi / Tp
    
    # alpha (Phillips constant) approximation
    alpha = 0.076 * (Hs**2 * wp**4 / g**2)**(-0.22)
    
    # JONSWAP sigma (width) depends on freq:
    sigma = np.where(omega <= wp, 0.07, 0.09)
    
    # exponential factor
    r = np.exp(-((omega - wp)**2) / (2.0 * sigma**2 * wp**2))
    
    # JONSWAP expression
    Sj = alpha * (g**2 / omega**5) * np.exp(-1.25*(wp/omega)**4) * gamma**r
    Sj = np.maximum(Sj, 0.0)  # ensure no negative
    return Sj

# ---------------------------------------------------
# 2) Directional spreading function (cos^{2s})
# ---------------------------------------------------
def directional_spreading(psi, psi0, s):
    """
    Returns D(psi - psi0), normalized so that 
    integral from -pi to +pi is ~1. 
    Faltinsen's cos^{2s} method:
      D(x) = (2^{2s-1} * s!) / [ pi * (2s-1)! ] * cos^{2s}(x), for |x| < pi/2
    """
    x = psi - psi0
    
    # Restrict to -pi/2 < x < pi/2
    # Outside that, D=0
    half_pi = np.pi/2
    if np.abs(x) > half_pi:
        return 0.0
    
    from math import factorial, cos
    # prefactor
    numerator   = 2.0**(2*s - 1) * factorial(s)
    denominator = np.pi * factorial(2*s - 1)
    C = numerator / denominator
    
    return C * (cos(x))**(2*s)

# ---------------------------------------------------
# 3) Parameters for the wave
# ---------------------------------------------------
g       = 9.81
Hs      = 2.5    # significant wave height [m]
Tp      = 8.0    # peak period [s]
gammaJS = 3.3    # JONSWAP peak enhancement
s       = 2      # spreading exponent (ITTC might use s=1 or 2)
psi0    = 0.0    # mean wave direction [radians], e.g. 0=waves along +x

# Frequency range
omega_min = 0.2
omega_max = 2.5
N_omega   = 40

omega_vec = np.linspace(omega_min, omega_max, N_omega)
domega    = omega_vec[1] - omega_vec[0]

# Direction range
psi_min   = -np.pi
psi_max   =  np.pi
N_psi     = 36

psi_vec   = np.linspace(psi_min, psi_max, N_psi, endpoint=False)
dpsi      = psi_vec[1] - psi_vec[0]

# ---------------------------------------------------
# 4) Build the 1D wave spectrum and then 2D S(omega, psi)
# ---------------------------------------------------
S_1D = jonswap_spectrum(omega_vec, Hs, Tp, gammaJS, g=g)

# Create S_2D by multiplying by D(psi - psi0)
S_2D = np.zeros((N_omega, N_psi))
for i in range(N_omega):
    for j in range(N_psi):
        D_val = directional_spreading(psi_vec[j], psi0, s)
        S_2D[i,j] = S_1D[i] * D_val

# ---------------------------------------------------
# 5) Convert S_2D into amplitude for each (omega_i, psi_j)
#    We'll build a random wave by superposition
# ---------------------------------------------------
A_ij = np.sqrt(2.0 * S_2D * domega * dpsi)  # amplitude
phase_ij = 2.0 * np.pi * np.random.rand(N_omega, N_psi)  # random phases

# ---------------------------------------------------
# 6) Time-domain synthesis of wave elevation at (x=0, y=0)
# ---------------------------------------------------
T_sim = 200.0
dt    = 0.2
time  = np.arange(0, T_sim, dt)
eta   = np.zeros_like(time)

for i in range(N_omega):
    omega_i = omega_vec[i]
    k_i     = omega_i**2 / g  # deep water approx
    for j in range(N_psi):
        psi_ij = psi_vec[j]
        amp_ij = A_ij[i,j]
        ph_ij  = phase_ij[i,j]
        
        # Summation in time
        # wave component traveling in direction psi_ij
        # at (x=0, y=0), the phase is just (-omega_i t + ph_ij) ignoring x,y
        for it, t in enumerate(time):
            eta[it] += amp_ij * np.cos(-omega_i*t + ph_ij)

# ---------------------------------------------------
# 7) Plot the resulting wave elevation vs time
# ---------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(time, eta, label='Directional Random Wave')
plt.xlabel('Time [s]')
plt.ylabel(r'$\eta(t)$ [m]')
plt.title('Directional Wave Elevation at (x=0, y=0)')
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------------------------------
# 8) Check total variance
# ---------------------------------------------------
# The variance of eta(t) should roughly match the total integral of S_2D.
var_time_domain = np.var(eta)
var_freq_domain = np.sum(S_2D)*domega*dpsi  # approximate integral
print(f"Variance (time-domain) ~ {var_time_domain:.3f}")
print(f"Variance (freq x dir)  ~ {var_freq_domain:.3f}")
