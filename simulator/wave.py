import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1) Define a minimal JONSWAP spectral function
#    (for demonstration, ignoring gamma & sigma specifics)
# ---------------------------------------------------
def jonswap_spectrum(omega, Hs, Tp, gamma=3.3, g=9.81):
    """
    Returns S_eta(omega): wave elevation spectrum [m^2 s],
    using a simplified JONSWAP form.
    
    Parameters:
    -----------
    omega : float or np.array
        Angular frequency [rad/s].
    Hs : float
        Significant wave height [m].
    Tp : float
        Peak period [s].
    gamma : float
        Peak enhancement factor (default ~3.3 for JONSWAP).
    g : float
        Gravity [m/s^2].
    """
    # Peak frequency
    wp = 2.0 * np.pi / Tp
    
    # alpha (Phillips constant) can be related to Hs,
    # but let's do a simplified approach:
    # a typical Pierson-Moskowitz or JONSWAP formula might be:
    alpha = 0.076 * (Hs**2 * wp**4 / g**2)**(-0.22)
    
    sigma = np.where(omega <= wp, 0.07, 0.09)  # JONSWAP piecewise sigma
    
    r = np.exp(- (omega - wp)**2 / (2 * sigma**2 * wp**2))
    # JONSWAP core
    Sj = alpha * g**2 / omega**5 * np.exp(-1.25*(wp/omega)**4) * gamma**r
    
    # Ensure non-negative
    return np.maximum(Sj, 0.0)


# ---------------------------------------------------
# 2) Set up wave / simulation parameters
# ---------------------------------------------------
g = 9.81               # gravity
Hs = 2.5               # significant wave height [m]
Tp = 8.0               # peak period [s]
gamma_js = 3.3         # JONSWAP peak enhancement
w_min, w_max = 0.4, 2.5  # range of angular frequencies [rad/s]
N_omega = 50           # number of frequency bins
rho = 1025.0           # sea water density [kg/m^3]

# Conceptual ship / lumped-volume parameters:
disp_vol = 500.0       # displaced volume [m^3] (example)
x_offset =  10.0       # offset from CG in x for moment calc [m]
y_offset = -2.0        # offset from CG in y for moment calc [m]

# Time marching setup
Tsim = 200.0           # total simulation time [s]
dt   = 0.2             # time step [s]
time = np.arange(0, Tsim, dt)

# Single wave heading, for demonstration (0 = +x direction)
wave_heading_deg = 30.0
theta_wave = np.radians(wave_heading_deg)


# ---------------------------------------------------
# 3) Discretize the wave spectrum
# ---------------------------------------------------
omega_vec = np.linspace(w_min, w_max, N_omega)
domega = omega_vec[1] - omega_vec[0]

# Evaluate JONSWAP spectrum
S_omega = jonswap_spectrum(omega_vec, Hs, Tp, gamma_js, g=g)

# For each freq bin, amplitude ~ sqrt(2 * S(omega) * delta_omega)
# (in a random approach, you'd also assign random phases)
A = np.sqrt(2.0 * S_omega * domega)  # wave amplitudes for each bin

# Random phases (for a random sea). For a single realization:
phases = 2.0 * np.pi * np.random.rand(N_omega)

# Wave numbers in deep water: k = omega^2 / g
k_vec = omega_vec**2 / g


# ---------------------------------------------------
# 4) Compute wave-induced velocity & acceleration at reference point (0,0,z=0)
#    ignoring vertical decay (z=0), just focusing on horizontal plane
#    If you want below surface effect, you can do e^(k z) factor
# ---------------------------------------------------
# We will do a time-marching sum of wave components.

u_x_total = np.zeros_like(time)
u_y_total = np.zeros_like(time)

# Also store accelerations if we want them directly
a_x_total = np.zeros_like(time)
a_y_total = np.zeros_like(time)

# Loop over each freq bin, add velocity contributions
for i in range(N_omega):
    omega_i = omega_vec[i]
    k_i     = k_vec[i]
    amp_i   = A[i]
    phi_i   = phases[i]
    
    # In deep water, the incident wave potential amplitude for this component:
    # phi_ampl = g * amp_i / omega_i  (if ignoring e^(kz) factor at z=0)
    # Then velocity components:
    #   u_x = d(phi)/dx,  u_y = d(phi)/dy
    # But simpler to recall known linear wave formulas for horizontal velocity
    #
    #   u_x(t) =  omega_i * amp_i * cos(k_i*x - omega_i*t + phi_i) * cos(theta_wave)
    # or a variant depending on exact sign conventions.
    #
    # We'll define a wave traveling at angle theta_wave, so wave vector:
    #   kx = k_i cos(theta_wave), ky = k_i sin(theta_wave)
    
    # Let's do a direct approach for velocity at (x=0,y=0):
    # Horizontal velocity amplitude at surface ~ omega_i * amp_i
    # direction cosines ~ (cos(theta_wave), sin(theta_wave))
    
    # "HORIZONTAL VELOCITY" (in linear theory, at z=0):
    #   u_comp = omega_i * amp_i * cos(k_i*(x cosT + y sinT) - omega_i t + phi_i)
    # We'll set x=0,y=0, so the phase is just (-omega_i t + phi_i).
    
    # Then project that onto x,y directions:
    
    for it, t in enumerate(time):
        # instantaneous phase
        phase_t = -omega_i * t + phi_i
        # wave velocity magnitude in the direction of wave propagation:
        u_mag = omega_i * amp_i * np.cos(phase_t)
        
        # x,y components
        u_x = u_mag * np.cos(theta_wave)
        u_y = u_mag * np.sin(theta_wave)
        
        # Store the sum
        u_x_total[it] += u_x
        u_y_total[it] += u_y

# Now, accelerations are just d(u)/dt; we can either do a numerical derivative
# or use the linear wave formula for acceleration directly:
# a_x = partial derivative wrt time => -omega_i^2 * amp_i * sin(phase_t)*cos(theta_wave)
# But let's do a second loop to be consistent:

# Reset totals
a_x_total[:] = 0.0
a_y_total[:] = 0.0

for i in range(N_omega):
    omega_i = omega_vec[i]
    amp_i   = A[i]
    phi_i   = phases[i]
    
    for it, t in enumerate(time):
        phase_t = -omega_i * t + phi_i
        a_mag = - (omega_i**2) * amp_i * np.sin(phase_t)
        
        a_x_total[it] += a_mag * np.cos(theta_wave)
        a_y_total[it] += a_mag * np.sin(theta_wave)

# ---------------------------------------------------
# 5) Compute "lumped-volume" wave load in 3 DOFs (surge, sway, yaw)
# ---------------------------------------------------
# We approximate F_x = rho * V * a_x, etc. (Froude-Krylov ignoring diffraction)
# Then yaw moment Mz = x_offset * F_y - y_offset * F_x

Fx = rho * disp_vol * a_x_total
Fy = rho * disp_vol * a_y_total
Mz = x_offset * Fy - y_offset * Fx

# ---------------------------------------------------
# 6) Quick plotting of time series
# ---------------------------------------------------
plt.figure()
plt.plot(time, Fx, label='Surge Force Fx [N]')
plt.plot(time, Fy, label='Sway Force Fy [N]')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.title('Wave-Induced Forces (Froude-Krylov Only)')
plt.grid(True)

plt.figure()
plt.plot(time, Mz, label='Yaw Moment Mz [Nm]')
plt.xlabel('Time [s]')
plt.ylabel('Moment [Nm]')
plt.title('Wave-Induced Yaw Moment (Froude-Krylov Only)')
plt.legend()
plt.grid(True)

plt.show()
