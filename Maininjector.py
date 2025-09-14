import cupy as np
from Tools import *
from AAHydrophobic import *
from UpdateChemicalPotential import *
from PropagationStoch import *
import os
import time 

def main(sequence, epsilon_hb, vchi_ps, eps_yukawa, decay_yukawa, bjerrum_length, decay_es, rhop0, max_iter, gamma, salt_fraction, gridshape, outdir):
    b_length = 0.38/3
    n_quad_per_rod = 3
    dx = 0.38/(3*n_quad_per_rod)
    l_hb = 0.18
    
    Nx, Ny, Nang = gridshape
    box_lengths = tuple(d * dx for d in gridshape[:2])
    V = np.prod(np.array(box_lengths))
    dV = V / (Nx * Ny)
    spat_weights = dV*np.ones((Nx, Ny))/V
    x = np.linspace(-box_lengths[0]/2,box_lengths[0]/2,Nx)
    y = np.linspace(-box_lengths[1]/2,box_lengths[1]/2,Ny)
    X,Y = np.meshgrid(x,y,indexing='ij')
    grid = np.array([X,Y])

    l_chain = ((3* len(sequence)-1)/3)*0.38+2*0.15*len(sequence)

    u_vectors, ang_weights = lebedev_grid(Nang)
    u_vectors /= np.linalg.norm(u_vectors, axis=1)[:, None]
    ang_weights /= np.sum(ang_weights)
    
    residue_class_per_segment, res_classes, rho0_all, rhobb_class, rhosc_class= build_initials(sequence, l_chain, rhop0, gridshape, spat_weights, ang_weights)
    charges, lambdas = gather_charges_and_lambdas(sequence)
    es_charges = get_es_charges(sequence)
    
    V_hydro, _ = make_3D_kernel_fft(yukawa_realspace, grid, eps_yukawa, decay_yukawa)
    V_es, _ = make_3D_kernel_fft(yukawa_realspace, grid,  bjerrum_length, decay_es)
    V_hb, _    = make_3D_kernel_fft(gaussian_realspace, grid,  epsilon_hb, l_hb)
    A_hb = build_angular_kernel(Nang, u_vectors, np.pi, ang_weights)
    geom_kernel = build_exp_angular_kernel(Nang, u_vectors, (2/3)*np.pi, ang_weights)

    DeltaRhoHystory = []
    PartialsHystory = {k: [0] * (max_iter+1) for k in rho0_all}
    c_gamma = gamma
    LDVCs = []
    LDVCmins = []
    LDVCmaxs = []
    rhoS = {}
    rho0_sv = {}
    total_charge = sum(es_charges.get(res, 0.0) for res in sequence)
    eq_iters = 1500
    prod_iters = 500
    save_interval = 3
    max_iter = eq_iters + prod_iters


    def computerhoS_zero(rho_class, rhop0, total_charge):
        rhoP_field = np.zeros(gridshape[:2])
        for key in rho0_all:
            rhoP_field += np.sum(rho_class[key]*ang_weights[None, None, ...], axis=-1)
        k = 1.0 - rhop0
        rho_s = salt_fraction * k
        counterion_fraction = np.abs(sum(es_charges[key] * Counter(sequence)[key] * rho0_all[key] for key in rho_class if es_charges[key] != 0))
        if counterion_fraction + rho_s > k:
            raise ValueError("Polymer charge + salt exceeds free fraction.")
            
        if total_charge >= 0:
            rho_minus = rho_s + counterion_fraction
            rho_plus = rho_s
        else:
            rho_plus = rho_s + counterion_fraction
            rho_minus = rho_s
        rho_neutral = k - rho_plus - rho_minus

        rho0_sv['plus'] = rho_plus
        rho0_sv['minus'] = rho_minus
        rho0_sv['neutral'] = rho_neutral

        scaling = 1.0 - rhoP_field
        rhoS['plus'] = rho_plus* scaling
        rhoS['minus'] = rho_minus * scaling
        rhoS['neutral'] = rho_neutral * scaling
        print(f'---> INITIAL DENSITIES -> plus: {rho_plus:.4f}, minus: {rho_minus:.4f}, neutral: {rho_neutral:.4f}, salt fraction: {rho_s:.4f}\n\n')
        return rhoS
    
    computerhoS_zero({**rhobb_class, **rhosc_class}, rhop0, total_charge)  
    Broken = False  
    PS = False
    c_gamma = gamma
    if outdir is None:
        outdir = f"density_outputs_{sequence}"
    os.makedirs(outdir, exist_ok=True)

    # subfolders
    logs_folder = os.path.join(outdir, "logs")
    plots_folder = os.path.join(outdir, "density_plots")
    npz_folder  = os.path.join(outdir, "saved_densities")
    propagator_folder  = os.path.join(outdir, "propagators")

    for folder in [logs_folder, plots_folder, npz_folder, propagator_folder]:
        os.makedirs(folder, exist_ok=True)

    # log file
    ldvc_log_filename = os.path.join(logs_folder, f"LDVCs_log_vchi_ps_{vchi_ps}_rhop0_{rhop0}_gamma_{gamma}.txt")
    with open(ldvc_log_filename, "w") as f:
        header = "iter\tLDVC_mean\tLDVC_max\tLDVC_min\t" + \
                "\t".join([f"Partial_{key}" for key in res_classes]) + \
                "\tDeviation\tQ\n"
        f.write(header)

    # initial plot before iterations
    plot_densities(sequence, {**rhobb_class, **rhosc_class}, rhoS, gridshape, -1,
                0.0, vchi_ps, gamma, 0.0, rhop0,
                eps_yukawa, bjerrum_length, ang_weights,
                plots_folder)
    ldvc_log_filename = os.path.join(logs_folder, f"LDVCs_log_vchi_pp_{0.0}_vchi_ps_{vchi_ps}_rhop0_{rhop0}_gamma_{gamma}.txt")
    with open(ldvc_log_filename, "w") as f:
        header = "iter\tLDVC_mean\tLDVC_max\tLDVC_min\t" + \
                "\t".join([f"Partial_{key}" for key in res_classes]) + \
                "\tDeviation_bb\tDeviation_solv\tDeviation_other\tQ\n"
        f.write(header)
    

    h_as = compute_has(sequence, charges, rhobb_class, res_classes, ang_weights, gridshape)
    c_field = compute_c(sequence, es_charges, rhobb_class, rhoS, res_classes, ang_weights, gridshape)
    
    mixer = SCFTUpdater(res_classes, gridshape, 0.0, vchi_ps,epsilon_hb, bjerrum_length, charges, lambdas, es_charges, n_history=5)
    w_chains, w_sidechains, w_solvent, xi = mixer.zero_update(rhobb_class, rhoS, rhosc_class, gamma, ang_weights, spat_weights, V_hydro, V_hb, A_hb, V_es, h_as, c_field, box_lengths, gridshape)

    q_prev_fw_list = np.zeros((len(list(residue_class_per_segment)), n_quad_per_rod, *gridshape), dtype = np.complex128)
    q_prev_bw_list = np.zeros((len(list(residue_class_per_segment)), n_quad_per_rod, *gridshape), dtype = np.complex128)
    q_prev_fwsc_list = {key: np.zeros((n_quad_per_rod, *gridshape), dtype = np.complex128) for key in ['Nsc', 'Csc']}
    q_prev_bwsc_list = {key: np.zeros((n_quad_per_rod, *gridshape), dtype = np.complex128) for key in ['Nsc', 'Csc']}
    
    for it in range(1, max_iter+1):
        start = time.time()
        h_as = compute_has(sequence, charges, rhobb_class, res_classes, ang_weights, gridshape)
        c_field = compute_c(sequence, es_charges, rhobb_class, rhoS, res_classes, ang_weights, gridshape)

        w_chains, w_sidechains, w_solvent, xi, deviations = mixer.linear_descent(xi, w_chains, w_solvent, w_sidechains, rhobb_class, rhoS, rhosc_class, gamma,ang_weights,spat_weights, V_hydro, V_hb, A_hb, V_es, h_as, c_field, box_lengths, gridshape)
        if it == 1:
            rhobb_class, rhosc_class, Q, q_prev_fw_list, q_prev_bw_list,q_prev_fwsc_list, q_prev_bwsc_list = propagate_closed(residue_class_per_segment, rho0_all, w_chains, w_sidechains, ang_weights, spat_weights, u_vectors, gridshape, b_length, n_quad_per_rod, D_theta = 0.48**(-1), Lx=dx*Nx, Ly = dx*Ny, dt=0.01, qf_previous= q_prev_fw_list, qb_previous= q_prev_bw_list, qf_prev_sc=q_prev_fwsc_list, qb_prev_sc=q_prev_bwsc_list, geom_kernel = geom_kernel, mode = 'deterministic')
        else:
            rhobb_class, rhosc_class, Q, q_prev_fw_list, q_prev_bw_list,q_prev_fwsc_list, q_prev_bwsc_list = propagate_closed(residue_class_per_segment, rho0_all, w_chains, w_sidechains,ang_weights, spat_weights, u_vectors, gridshape, b_length, n_quad_per_rod, D_theta = 0.48**(-1), Lx=dx*Nx, Ly = dx*Ny, dt=0.01, qf_previous = q_prev_fw_list, qb_previous = q_prev_bw_list, qf_prev_sc=q_prev_fwsc_list, qb_prev_sc=q_prev_bwsc_list,geom_kernel = geom_kernel, mode = 'thermal')
        for solvents in ['plus', 'minus', 'neutral']:
            rhoS[solvents] =  rho0_sv[solvents]*np.exp(-w_solvent[solvents])/(np.sum(spat_weights*np.exp(-w_solvent[solvents])))

        total_diff, partial_diffs = compute_constraint_violations(rho0_all, {**rhobb_class, **rhosc_class}, spat_weights, ang_weights, V)
        DeltaRhoHystory.append(total_diff)
        
        local_density_violation_no_rhoSneutral = np.zeros(gridshape[:2], dtype = np.float64)
        for key in {**rhobb_class, **rhosc_class}:
            local_density_violation_no_rhoSneutral += np.sum({**rhobb_class, **rhosc_class}[key] * ang_weights, axis=-1)
        LDVC_mean = np.mean(local_density_violation_no_rhoSneutral)
        LDVC_max = np.max(local_density_violation_no_rhoSneutral)
        LDVC_min = np.min(local_density_violation_no_rhoSneutral)

        partials_values = [partial_diffs[idx] for idx in range(len(res_classes))]
        deviation1, deviation2, deviation3 = deviations  # unpack tuple

        with open(ldvc_log_filename, "a") as f:
            f.write(f"{it}\t{LDVC_mean:.8f}\t{LDVC_max:.8f}\t{LDVC_min:.8f}\t")
            f.write("\t".join([f"{val:.8f}" for val in partials_values]))
            f.write(f"\t{deviation1:.8f}\t{deviation2:.8f}\t{deviation3:.8f}\t{Q:.8f}\n")
            
        if any(np.isnan({**rhobb_class, **rhosc_class}[key]).any() for key in {**rhobb_class, **rhosc_class}) or any(np.isnan(rhoS[key]).any() for key in rhoS):
            Broken = True
            print(f'NaN detected at iteration {it}, simulation broken. GAMMA = {c_gamma:.4f},')
            break
        if it % 1 == 0:
            print(f'Iter {it} | Elapsed: {time.time()- start}|LDVC mean/max/min={LDVC_mean:.4f}/{LDVC_max:.4f}/{LDVC_min:.4f}, gamma={c_gamma:.4f}, vchi_ps = {vchi_ps}')
        if it % 25 == 0:
            plot_densities(sequence, {**rhobb_class, **rhosc_class}, rhoS, gridshape, it,
                        0.0, vchi_ps, gamma, 0.0, rhop0,
                        eps_yukawa, bjerrum_length, ang_weights,
                        plots_folder)
        if it > eq_iters and ((it - eq_iters) % save_interval == 0):
            save_idx = it

            # --- Save densities ---
            save_fname = os.path.join(npz_folder, f"dens_iter_{save_idx:04d}.npz")
            try:
                save_dict = {}
                for k, arr in {**rhobb_class, **rhosc_class}.items():
                    save_dict[f"rho_{k}"] = arr.astype(np.float32) if np.isrealobj(arr) else arr
                for s_key, s_arr in rhoS.items():
                    save_dict[f"rhoS_{s_key}"] = s_arr.astype(np.float32) if np.isrealobj(s_arr) else s_arr

                save_dict["iteration"] = np.array([it])
                save_dict["Q"] = np.array([Q])
                save_dict["LDVC_mean"] = np.array([LDVC_mean])

                np.savez_compressed(save_fname, **save_dict)
                print(f"Saved densities at iter {it} -> {save_fname}")
            except Exception as e:
                print(f"Warning: failed to save densities at iter {it}: {e}")

            # --- Save propagators ---
            prop_fname = os.path.join(propagator_folder, f"prop_iter_{save_idx:04d}.npz")
            try:
                prop_dict = {
                    "q_fw": [q.astype(np.complex128) for q in q_prev_fw_list],
                    "q_bw": [q.astype(np.complex128) for q in q_prev_bw_list],
                }
                np.savez_compressed(prop_fname, **prop_dict)
                print(f"Saved propagators at iter {it} -> {prop_fname}")
            except Exception as e:
                print(f"Warning: failed to save propagators at iter {it}: {e}")

    if not Broken:
        plot_densities(sequence, {**rhobb_class, **rhosc_class}, rhoS, gridshape, it, 0.0, vchi_ps, c_gamma, 0.0, rhop0, eps_yukawa, bjerrum_length, outdir)
        rhoS_all = np.concatenate([rhoS[key].ravel() for key in rhoS])
        if np.abs(np.max(rhoS_all) - np.min(rhoS_all)) > 0.1 * np.min(rhoS_all):
            PS = True
    return PartialsHystory, DeltaRhoHystory, LDVCs, LDVCmaxs, LDVCmins, Broken, PS
