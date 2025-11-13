#Name-Eklavya Chauhan
#Date-2024-06-12
#Project-P346 DIY Project: Simulating the Orbit of the Moon around the Earth usiging RK4 Method
#Roll no- 2311067

import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -----------------------
# Constants and Initial Conditions
# -----------------------
G = 6.67430e-11
M_earth = 5.972e24
m_moon = 7.348e22

r0_km = 384400
r0 = r0_km * 1000
x0, y0 = r0, 0
vx0, vy0 = 0, 1045   # adjust for eccentricity

dt = 30
t_max_days = 30
t_max = t_max_days * 24 * 3600

# -----------------------
# Functions
# -----------------------
def acceleration(x, y):
    r2 = x*x + y*y
    r = math.sqrt(r2)
    factor = -G * M_earth / (r2 * r)
    return factor * x, factor * y

def rk4_step(state, dt):
    x, y, vx, vy = state

    def deriv(s):
        sx, sy, svx, svy = s
        ax, ay = acceleration(sx, sy)
        return svx, svy, ax, ay

    k1 = deriv((x, y, vx, vy))
    k2 = deriv((x + 0.5*dt*k1[0], y + 0.5*dt*k1[1],
                vx + 0.5*dt*k1[2], vy + 0.5*dt*k1[3]))
    k3 = deriv((x + 0.5*dt*k2[0], y + 0.5*dt*k2[1],
                vx + 0.5*dt*k2[2], vy + 0.5*dt*k2[3]))
    k4 = deriv((x + dt*k3[0], y + dt*k3[1],
                vx + dt*k3[2], vy + dt*k3[3]))

    x_new = x + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    y_new = y + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    vx_new = vx + (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    vy_new = vy + (dt/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    return x_new, y_new, vx_new, vy_new

def kinetic_energy(vx, vy):
    return 0.5 * m_moon * (vx*vx + vy*vy)

def potential_energy(x, y):
    r = math.sqrt(x*x + y*y)
    return -G * M_earth * m_moon / r

# -----------------------
# Simulation
# -----------------------
def simulate():
    state = (x0, y0, vx0, vy0)
    t = 0.0

    times, KE, PE, TE = [], [], [], []
    r_list, xs, ys = [], [], []
    L_list = []

    while t <= t_max:
        x, y, vx, vy = state
        r = math.sqrt(x*x + y*y)

        ke = kinetic_energy(vx, vy)
        pe = potential_energy(x, y)
        te = ke + pe
        L = m_moon * (x * vy - y * vx)

        times.append(t)
        KE.append(ke)
        PE.append(pe)
        TE.append(te)
        L_list.append(L)
        r_list.append(r)
        xs.append(x)
        ys.append(y)

        state = rk4_step(state, dt)
        t += dt

    return times, KE, PE, TE, r_list, xs, ys, L_list

# -----------------------
# Analysis
# -----------------------
def analyze(times, KE, PE, TE, r_list, L_list):
    r_min = min(r_list)
    r_max = max(r_list)
    r_avg = sum(r_list)/len(r_list)

    v_avg = math.sqrt(2 * sum(KE) / (m_moon * len(KE)))
    E_avg = sum(TE)/len(TE)
    L_avg = sum(L_list)/len(L_list)

    eps = E_avg / m_moon
    mu = G * M_earth

    if eps < 0:
        a = -mu/(2*eps)
        T_sec = 2*math.pi * math.sqrt(a**3 / mu)
        T_days = T_sec/(24*3600)
    else:
        T_days = None

    e = (r_max - r_min)/(r_max + r_min)

    return {
        "period": T_days,
        "r_avg": r_avg/1000,
        "r_min": r_min/1000,
        "r_max": r_max/1000,
        "e": e,
        "v_avg": v_avg,
        "E_avg": E_avg,
        "L_avg": L_avg
    }

# -----------------------
# Plot: Energy
# -----------------------
def plot_energy(times, KE, PE, TE, r_list):
    t_days = [t/(24*3600) for t in times]
    plt.figure(figsize=(8,6))
    plt.plot(t_days, KE, label="KE (J)")
    plt.plot(t_days, PE, label="PE (J)")
    plt.plot(t_days, TE, label="TE (J)")

    i_rmin = r_list.index(min(r_list))
    i_rmax = r_list.index(max(r_list))

    for arr, name in zip([KE,PE,TE],["KE","PE","TE"]):
        plt.scatter(t_days[i_rmin], arr[i_rmin], color="red", edgecolor="black",
                    label=f"Perihelion ({name})")
        plt.scatter(t_days[i_rmax], arr[i_rmax], color="lime", edgecolor="black",
                    label=f"Aphelion ({name})")

    plt.xlabel("Time (days)")
    plt.ylabel("Energy (J)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("energy.png",dpi=200)
    print("Saved energy.png")

# -----------------------
# Plot: Orbit
# -----------------------
def plot_orbit(xs, ys, r_list):
    xs_km = [x/1000 for x in xs]
    ys_km = [y/1000 for y in ys]
    i_rmin = r_list.index(min(r_list))
    i_rmax = r_list.index(max(r_list))

    plt.figure(figsize=(6.5,6))
    plt.plot(xs_km, ys_km, 'k-', linewidth=0.9)
    plt.scatter(0,0,color="blue",s=80,label="Earth")
    plt.scatter(xs_km[i_rmin],ys_km[i_rmin],color="red",s=55,edgecolor="black",label="Perihelion")
    plt.scatter(xs_km[i_rmax],ys_km[i_rmax],color="lime",s=55,edgecolor="black",label="Aphelion")

    plt.text(xs_km[i_rmin]*1.02, ys_km[i_rmin]*1.02, "Perihelion", color="red")
    plt.text(xs_km[i_rmax]*1.02, ys_km[i_rmax]*1.02, "Aphelion", color="green")

    ax = plt.gca()
    ax.set_aspect("equal","box")

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci',axis='both',scilimits=(0,0))

    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("orbit.png",dpi=200)
    print("Saved orbit.png")

# -----------------------
# Plot: Angular Momentum
# -----------------------
def plot_angular_momentum(times, L_list):
    t_days = [t/(24*3600) for t in times]
    plt.figure(figsize=(8,5))
    plt.plot(t_days, L_list, color="purple")
    plt.xlabel("Time (days)")
    plt.ylabel("L (kg m²/s)")
    plt.title("Angular Momentum vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("angular_momentum.png", dpi=200)
    print("Saved angular_momentum.png")

# -----------------------
# Plot: Effective Potential
# -----------------------
def plot_effective_potential(r_list, L_list):
    L = sum(L_list)/len(L_list)
    mu = G*M_earth

    r_vals = sorted(list(set(r_list)))
    r_vals = r_vals[::int(len(r_vals)/500)+1]

    V_eff = []
    for r in r_vals:
        U_grav = -mu*m_moon/r
        U_cent = (L**2)/(2*m_moon*r**2)
        V_eff.append(U_grav + U_cent)

    r_km = [r/1000 for r in r_vals]

    plt.figure(figsize=(8,5))
    plt.plot(r_km, V_eff, color="brown")
    plt.xlabel("r (km)")
    plt.ylabel("Effective Potential V_eff (J)")
    plt.title("Effective Potential of Moon-Earth System")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("effective_potential.png", dpi=200)
    print("Saved effective_potential.png")

# -----------------------
# Save Parameters
# -----------------------
def save_parameters(params):
    with open("parameters.txt","w") as f:
        for k,v in params.items():
            f.write(f"{k}: {v}\n")
    print("Saved parameters.txt")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    print("Simulating Moon orbit (Earth fixed)...")

    times, KE, PE, TE, r_list, xs, ys, L_list = simulate()
    params = analyze(times, KE, PE, TE, r_list, L_list)

    plot_energy(times, KE, PE, TE, r_list)
    plot_orbit(xs, ys, r_list)
    plot_angular_momentum(times, L_list)
    plot_effective_potential(r_list, L_list)

    save_parameters(params)

    print("\nSummary of results:")
    for k, v in params.items():
        print(f"{k:15s}: {v}")

    print("\nDone. All plots and parameters saved.")
