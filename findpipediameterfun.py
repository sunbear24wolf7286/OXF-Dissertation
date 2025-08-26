import math
import matplotlib.pyplot as plt

def find_pipe_diameter(
    friction_rate,  # target friction per length [MPa/m]
    rho_avg,        # density [kg/m³]
    mass_rate,      # mass flow rate [kg/s]
    mu_avg,         # dynamic viscosity [Pa·s]
    D_guess,        # initial diameter guess [m]
    roughness,      # absolute roughness [m]
    D_step,         # increment for diameter [m]
    max_iter = 1000      # max loops
):

    # bind locals for speed
    pi = math.pi
    log = math.log
    sqrt = math.sqrt
    
    Ds = []
    frictions = []
    D = D_guess
    
    for _ in range(max_iter):
        A   = 0.25 * pi * D * D
        vel = mass_rate / (rho_avg * A)
        Re  = rho_avg * vel * D / mu_avg
        
        if Re < 2300.0:
            f = 24.0 / Re
        else:
            K_rel = roughness / D
            a = K_rel / 3.7
            b = 2.51 / Re
            c = -2.0 / log(10)
            X0 = b * abs(-1.8 * math.log10(a**1.11 + 6.9 / Re))
            c1 = b * c / (3 * (a + X0)**3)
            c2 = -b * c / (2 * (a + X0)**2) - 3 * c1 * X0
            c3 = 3 * c1 * X0**2 + (b * c / (a + X0)**2) * X0 + b * c / (a + X0) - 1
            c4 = (b * c * log(a + X0)
                  - (b * c / (a + X0)) * X0
                  - (b * c / (2 * (a + X0)**2)) * X0**2
                  - c1 * X0**3)
            sigma = c3 / (3 * c1) - c2*c2 / (9 * c1*c1)
            phi   = c4 / (2 * c1) + (c2/(3*c1))**3 - c2*c3/(6*c1*c1)
            term  = (sqrt(phi*phi + sigma**3) - phi)**(1/3) \
                  - (sqrt(phi*phi + sigma**3) + phi)**(1/3) \
                  + (a + 3*X0)/2
            f = (b*b) * term**(-2)

        actual_fric = (f / D) * 0.5 * rho_avg * vel*vel / 1e6
        
        Ds.append(D)
        frictions.append(actual_fric)
        
        if actual_fric <= friction_rate:
            break
        
        D += D_step
    else:
        raise RuntimeError("Failed to meet target friction_rate within max_iter")

    plt.figure(figsize=(3,3))
    plt.plot(Ds, frictions, 'o-')
    plt.axhline(friction_rate, linestyle='--', label=f"target {friction_rate}")
    plt.xlabel("D (m)")
    plt.ylabel("Friction (MPa/m)")
    plt.title("Diameter vs Friction")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"Final diameter: {D:.6f} m")
    print(f"Final friction: {actual_fric:.6f} MPa/m")
    return D
