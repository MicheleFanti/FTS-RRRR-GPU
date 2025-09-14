import os
import itertools
import cupy as np
from Maininjector import main

if __name__ == "__main__":
    '''    sequences = [
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA",
    "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
    ]'''

    '''sequences = ["GNNQQNY",
    "VQIVYK",
    "KLVFFA"
    ]'''

    sequences = ["QKLVFFAE"]#, "GVVHGVTTVA", "SLSLLSLSLLS"]
    
    gridshape = (200, 200, 60)  
    box_lengths = (225, 225) 
    max_iter = 2000  
    dx = 1.5
    epsilon_hb = 0.0
    alpha_pb = 0.0
    rhop0_values = [0.11]
    vchi_ps_values = [0.0]
    vchi_pp_values = [0]
    bjerrum_values = [0.0]
    eps_yukawa_values = [1, 0.05]
    salt_fractions = [0.005]
    initial_gamma_values = [0.1]
    decay_yukawa = 5
    decay_es = 20

    for sequence in sequences:
        param_combinations = list(itertools.product(
            rhop0_values, salt_fractions, vchi_ps_values, 
            vchi_pp_values, bjerrum_values, eps_yukawa_values
        ))
        print(f"\nTrying all parameter combinations for sequence: {sequence} ({len(param_combinations)} combinations)")
        for i, (rhop0, salt_fraction, vchi_ps, vchi_pp, bjerrum, eps_yukawa) in enumerate(param_combinations):
            gamma_success = None
            current_gamma_index = 0
            while current_gamma_index < len(initial_gamma_values):
                gamma = initial_gamma_values[current_gamma_index]
                outdir = f"EPSILONHB=0_0/{sequence}/bj{bjerrum}_vps{vchi_ps}_eps{eps_yukawa}/g{gamma}"
                os.makedirs(outdir, exist_ok=True)
                print(f"\n[{i+1}/{len(param_combinations)}] {sequence}, rhop0={rhop0}, salt={salt_fraction}, gamma={gamma}")
                results = main(sequence, epsilon_hb, vchi_ps, eps_yukawa, decay_yukawa, bjerrum, decay_es, rhop0, max_iter, gamma, salt_fraction, gridshape, outdir)
                PartialsHystory, DeltaRhoHystory, LDVCs, LDVCmaxs, LDVCmins, Broken, PS = results
                
            if gamma_success is None:
                print(f"  -> All gamma values failed for this parameter set.")
