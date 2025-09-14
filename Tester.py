import os
import sys
import cupy as np
from Maininjector import main

if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Usage: python3 Tester.py SEQUENCE eps_hb_vals eps_yukawa_vals")
    
    sequence = sys.argv[1]
    
    # Split the comma-separated lists and convert to floats
    epsilon_hb_values = [float(x) for x in sys.argv[2].split(',')]
    eps_yukawa_values = [float(x) for x in sys.argv[3].split(',')]
    
    # Fixed parameters
    gridshape = (225, 225, 60)
    max_iter = 2000
    rhop0 = 0.11
    vchi_ps = 0.0
    vchi_pp = 0.2
    bjerrum = 0.7
    salt_fraction = 0.005
    initial_gamma_values = [0.1]
    decay_yukawa = 0.5
    decay_es = 2

    # Loop over all combinations of eps_hb and eps_yukawa
    for epsilon_hb in epsilon_hb_values:
        for eps_yukawa in eps_yukawa_values:
            outdir = f"{sequence}/bj{bjerrum}_vps{vchi_ps}_epsy{eps_yukawa}_epshb{epsilon_hb}"
            os.makedirs(outdir, exist_ok=True)

            for gamma in initial_gamma_values:
                gamma_outdir = f"{outdir}/g{gamma}"
                os.makedirs(gamma_outdir, exist_ok=True)
                
                print(f"\nRunning {sequence} with eps_hb={epsilon_hb}, eps_yukawa={eps_yukawa}, gamma={gamma}")
                
                results = main(
                    sequence, epsilon_hb, vchi_pp, vchi_ps, eps_yukawa, 
                    decay_yukawa, bjerrum, decay_es, rhop0, max_iter, 
                    gamma, salt_fraction, gridshape, gamma_outdir
                )

                PartialsHystory, DeltaRhoHystory, LDVCs, LDVCmaxs, LDVCmins, Broken, PS = results
