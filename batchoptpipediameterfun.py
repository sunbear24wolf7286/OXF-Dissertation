import pandas as pd
import math
import numpy as np

def batch_opt_pipe_diameter(
    friction_rate,  # MPa/m
    rho_avg,        # kg/m³
    mu_avg,         # Pa·s
    D_start,        # m, initial guess for the very first mass rate
    roughness,      # m
    D_step,         # m
    max_iter = 1000000
):

    def ff(K, Re):
        a = K * (1/3.7)
        b = 2.51 / Re
        c = -2.0 / math.log(10)
        X0 = b * abs(-1.8 * math.log10(a**1.11 + 6.9 / Re))
        c1 = b * c / (3 * (a + X0)**3)
        c2 = -b * c / (2 * (a + X0)**2) - 3 * c1 * X0
        c3 = 3 * c1 * X0**2 + (b * c / (a + X0)**2) * X0 + b * c / (a + X0) - 1
        c4 = (b * c * math.log(a + X0)
              - (b * c / (a + X0)) * X0
              - (b * c / (2 * (a + X0)**2)) * X0**2
              - c1 * X0**3)
        σ = c3/(3*c1) - c2*c2/(9*c1*c1)
        φ = c4/(2*c1) + (c2/(3*c1))**3 - c2*c3/(6*c1*c1)
        term = ((φ*φ + σ*σ*σ)**0.5 - φ)**(1/3) \
               - ((φ*φ + σ*σ*σ)**0.5 + φ)**(1/3) \
               + (a + 3*X0)/2
        return (b*b) * term**(-2)

    results = []
    D_prev = D_start

    # sweep mass rates from 1 to 1000 kg/hr (converted to kg/s)
    for m in np.arange(1, 1001) / 3600.0:
        D = D_prev
        for _ in range(max_iter):
            A   = 0.25 * math.pi * D * D
            vel = m / (rho_avg * A)
            Re  = rho_avg * vel * D / mu_avg

            if Re < 2300:
                f = 24.0 / Re
            else:
                f = ff(roughness / D, Re)

            # MPa/m:
            actual_fr = (f / D) * 0.5 * rho_avg * vel**2 / 1e6
            if actual_fr <= friction_rate:
                break

            D += D_step
        else:
            raise RuntimeError(f"m={m}: didn't converge")

        results.append((m, D))
        D_prev = D 

    return pd.DataFrame(results, columns=['mdot','Dsol'])
