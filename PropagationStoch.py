import cupy as np
from cupy.fft import fft, ifft, fft2, ifft2, fftfreq
from collections import Counter

def strang_step_wlc(q, w, ds, KX, KY, UX, UY, ang_mul_half):
    q = ifft(fft(q, axis=2, norm='ortho') * ang_mul_half[None,None,:], axis=2, norm='ortho')
    phase_half = np.exp(-1j*(KX[...,None]*UX[None,None,:] + KY[...,None]*UY[None,None,:])*(ds/2))
    q = ifft2(fft2(q, axes=(0,1), norm='ortho') * phase_half, axes=(0,1), norm='ortho')
    q = q * np.exp(-w * ds)
    q = ifft2(fft2(q, axes=(0,1), norm='ortho') * phase_half, axes=(0,1), norm='ortho')
    q = ifft(fft(q, axis=2, norm='ortho') * ang_mul_half[None,None,:], axis=2, norm='ortho')
    return q

def strang_step_wlc_backward(q, w, ds, KX, KY, UX, UY, ang_mul_half):
    q = ifft(fft(q, axis=2, norm='ortho') * ang_mul_half[None,None,:], axis=2, norm='ortho')
    phase_half = np.exp(1j*(KX[...,None]*UX[None,None,:] + KY[...,None]*UY[None,None,:])*(ds/2))
    q = ifft2(fft2(q, axes=(0,1), norm='ortho') * phase_half, axes=(0,1), norm='ortho')
    q = q * np.exp(-w * ds)
    q = ifft2(fft2(q, axes=(0,1), norm='ortho') * phase_half, axes=(0,1), norm='ortho')
    q = ifft(fft(q, axis=2, norm='ortho') * ang_mul_half[None,None,:], axis=2, norm='ortho')
    return q

def propagate_forward_wlc(q0_spatial, w, theta_grid, length, n_substeps, D_theta, Lx, Ly, mu_forward, dt, q_prev, mode):
    Nx, Ny = q0_spatial.shape[:2]
    Ntheta = len(theta_grid)
    ds = length / n_substeps
    KX = np.meshgrid(2*np.pi*fftfreq(Nx, d=Lx/Nx), 2*np.pi*fftfreq(Ny, d=Ly/Ny), indexing='ij')[0]
    KY = np.meshgrid(2*np.pi*fftfreq(Nx, d=Lx/Nx), 2*np.pi*fftfreq(Ny, d=Ly/Ny), indexing='ij')[1]
    UX = np.cos(theta_grid)
    UY = np.sin(theta_grid)
    m = np.arange(Ntheta)
    m[m > Ntheta//2] -= Ntheta
    ang_mul_half = np.exp(-D_theta * (m**2) * (ds/2))
    #breakpoint()
    if q0_spatial.ndim == 2:
        q_curr = np.repeat(q0_spatial[:, :, None], Ntheta, axis=2).astype(np.complex128)
    else:
        q_curr = q0_spatial.astype(np.complex128)
    q_full = np.zeros((n_substeps, Nx, Ny, Ntheta), dtype=np.complex128)
    w_arr = w if w.ndim==3 else np.repeat(w[:,:,None], Ntheta, axis=2)
    for i in range(n_substeps):
        if mode == 'thermal':
            q_curr = strang_step_wlc(q_curr-dt * (q_curr - q_prev[i]) + dt * mu_forward[..., i], w_arr, ds, KX, KY, UX, UY, ang_mul_half)
        elif mode == 'deterministic':
            q_curr = strang_step_wlc(q_curr, w_arr, ds, KX, KY, UX, UY, ang_mul_half)
        q_full[i] = q_curr
    return np.real(q_full)

def propagate_backward_wlc(q0_spatial, w, theta_grid, length, n_substeps, D_theta, Lx, Ly, mu_backward, dt, q_prev, mode):
    Nx, Ny = q0_spatial.shape[:2]
    Ntheta = len(theta_grid)
    ds = length / n_substeps
    KX = np.meshgrid(2*np.pi*fftfreq(Nx, d=Lx/Nx), 2*np.pi*fftfreq(Ny, d=Ly/Ny), indexing='ij')[0]
    KY = np.meshgrid(2*np.pi*fftfreq(Nx, d=Lx/Nx), 2*np.pi*fftfreq(Ny, d=Ly/Ny), indexing='ij')[1]
    UX = np.cos(theta_grid)
    UY = np.sin(theta_grid)
    m = np.arange(Ntheta)
    m[m > Ntheta//2] -= Ntheta
    ang_mul_half = np.exp(-D_theta * (m**2) * (ds/2))


    if q0_spatial.ndim == 2:
        q_curr = np.repeat(q0_spatial[:, :, None], Ntheta, axis=2).astype(np.complex128)
    else:
        q_curr = q0_spatial.astype(np.complex128)
    q_full = np.zeros((n_substeps, Nx, Ny, Ntheta), dtype=np.complex128)
    w_arr = w if w.ndim==3 else np.repeat(w[:,:,None], Ntheta, axis=2)

    for i in range(n_substeps):
        if mode == 'thermal':
            q_curr = strang_step_wlc_backward(q_curr- dt * (q_curr - q_prev[i]) + np.sqrt(dt) * mu_backward[..., i], w_arr, ds, KX, KY, UX, UY, ang_mul_half)
        elif mode == 'deterministic':
            q_curr = strang_step_wlc_backward(q_curr, w_arr, ds, KX, KY, UX, UY, ang_mul_half)
        q_full[i] = q_curr
    return np.real(q_full)

def propagate_closed(backbone_seq, rho0_per_class, w_per_class, w_sc, ang_weights, spat_weights, u_grid, gridshape, length_rod, n_quad_per_rod, D_theta, Lx, Ly, dt, qf_previous, qb_previous, qf_prev_sc, qb_prev_sc, geom_kernel, mode):
    seq = list(backbone_seq)
    Nx, Ny, Ntheta = gridshape
    N = len(seq)
    sequence_length = int((N+1)/3)


    qf_list = [None]*N
    q_init_spatial = np.ones(gridshape[:2], dtype = np.complex128)
    theta_grid = np.linspace(0, 2*np.pi, len(u_grid))
    eta_1_full = np.random.randn(Nx, Ny, N, n_quad_per_rod) * np.sqrt(2)
    eta_2_full = np.random.randn(Nx, Ny, N, n_quad_per_rod) * np.sqrt(2)
    eta_sc1_full = {key: np.random.randn(Nx, Ny, n_quad_per_rod) * np.sqrt(2) for key in ['Nsc', 'Csc']}
    eta_sc2_full = {key: np.random.randn(Nx, Ny, n_quad_per_rod) * np.sqrt(2) for key in ['Nsc', 'Csc']}


    #Propagate sidechains:
    q_sc_forward = {key: np.zeros((n_quad_per_rod, *gridshape), dtype=np.complex128) for key in ['Nsc', 'Csc']}

    for idx in ['Nsc', 'Csc']:
        eta_1 = eta_sc1_full[idx]
        eta_2 = eta_sc2_full[idx]
        mu_forward = (eta_1 + 1j*eta_2)/np.sqrt(2)
        mu_forward = np.broadcast_to(mu_forward[:, :, None, :], (Nx, Ny, Ntheta, n_quad_per_rod))
        q_sc_forward[idx] = propagate_forward_wlc(np.ones(gridshape[:2]), w_sc[idx], theta_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly, mu_forward, dt, qf_prev_sc[idx], mode)

    #Propagate main backbone forward
    for idx in range(N):
        eta_1 = eta_1_full[:, :, idx, :]   # pick residue idx
        eta_2 = eta_2_full[:, :, idx, :]
        mu_forward = (eta_1 + 1j*eta_2)/np.sqrt(2)
        mu_forward = np.broadcast_to(mu_forward[:, :, None, :], (Nx, Ny, Ntheta, n_quad_per_rod))
        res = seq[idx]
        if idx == 0: 
            q_init_spatial = np.tensordot(q_sc_forward['Nsc'][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))
        elif idx % 3 == 0: 
            q_init_spatial *= np.tensordot(q_sc_forward['Nsc'][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))
        elif idx % 3 == 2: 
            q_init_spatial *= np.tensordot(q_sc_forward['Csc'][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))
        qf_list[idx] = propagate_forward_wlc(q_init_spatial, w_per_class[res], theta_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly, mu_forward, dt, qf_previous[idx], mode)
        q_init_spatial = np.tensordot(qf_list[idx][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))
    qb_list = [None]*N
    q_prev_spatial = np.ones(gridshape, dtype = np.complex128)
    q_sc_bb = {key: np.zeros(gridshape, dtype = np.complex128) for key in ['Nsc', 'Csc']}
    for idx in range(N-1, -1, -1):
        eta_1 = eta_1_full[:, :, idx, :]
        eta_2 = eta_2_full[:, :, idx, :]
        eta_1 = eta_1[:, :, ::-1]
        eta_2 = eta_2[:, :, ::-1]

        mu_backward = (eta_2 + 1j*eta_1)/np.sqrt(2)
        mu_backward = np.broadcast_to(mu_backward[:, :, None, :], (Nx, Ny, Ntheta, n_quad_per_rod))
        
        res = seq[idx]
        if idx == N-1:
            q_prev_spatial = np.tensordot(q_sc_forward['Csc'][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))
        elif idx % 3 == 0:
            q_prev_spatial *= np.tensordot(q_sc_forward['Nsc'][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))
        elif idx % 3 == 2:
            q_prev_spatial *= np.tensordot(q_sc_forward['Csc'][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))

        qb_list[idx] = propagate_backward_wlc(q_prev_spatial, w_per_class[res], theta_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly, mu_backward, dt, qb_previous[idx], mode)
        q_prev_spatial = np.tensordot(qb_list[idx][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))

        
        if idx == 0:
            q_sc_bb['Nsc'] += np.tensordot(q_prev_spatial * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))
        elif idx == N-1:
            q_sc_bb['Csc'] += np.tensordot(qf_list[N-1][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))
        elif idx % 3 == 2:
            q_sc_bb['Csc'] += np.tensordot(q_prev_spatial * ang_weights[None,None,:], geom_kernel, axes=([2],[0])) * np.tensordot(qf_list[idx-1][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))
        elif idx % 3 == 0:
            q_sc_bb['Nsc'] += np.tensordot(q_prev_spatial * ang_weights[None,None,:], geom_kernel, axes=([2],[0])) * np.tensordot(qf_list[idx-1][-1] * ang_weights[None,None,:], geom_kernel, axes=([2],[0]))
    q_sc_bb_full = {key: np.zeros((n_quad_per_rod, *gridshape), dtype = np.complex128) for key in ['Nsc', 'Csc']}
    for idx in ['Nsc', 'Csc']:
        eta_1 = eta_sc1_full[idx][:, : , ::-1]
        eta_2 = eta_sc2_full[idx][:, : , ::-1]
        mu_backward = (eta_2 + 1j*eta_1)/np.sqrt(2)
        mu_backward = np.broadcast_to(mu_backward[:, :, None, :], (Nx, Ny, Ntheta, n_quad_per_rod))
        q_sc_bb_full[idx] = propagate_backward_wlc(q_sc_bb[idx], w_sc[idx], theta_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly, mu_backward, dt, qb_prev_sc[idx], mode)
    
    temp_f = np.sum(qf_list[-1][-1] * ang_weights[None,None,:], axis=-1)
    
    Q = np.sum(temp_f * spat_weights)
    Q_check = np.sum(np.sum(qb_list[0][-1] * ang_weights[None,None,:], axis=-1) * spat_weights)
    
    print(f"Q = {Q}, Q_check = {Q_check}, rel.err = {(Q - Q_check)/Q_check}")
    
    rho_bb = {res: np.zeros(gridshape, dtype=np.float64) for res in set(seq)}
    rho_sc = {res: np.zeros(gridshape, dtype=np.float64) for res in ['Nsc', 'Csc']}
    
    for idx in range(N):
        res = seq[idx]
        qf = qf_list[idx]
        qb = qb_list[idx]
        for s in range(n_quad_per_rod):
            rho_bb[res] += np.real((rho0_per_class[res]/(n_quad_per_rod*Counter(seq)[res]*Q_check))*np.real(qf[s] * qb[s]))
    for idx in rho_sc:
        for s in range(n_quad_per_rod):
            rho_sc[idx] += (rho0_per_class[idx]/(n_quad_per_rod*sequence_length*(Q_check)))*np.real(q_sc_forward[idx][s] * q_sc_bb_full[idx][s])

    return rho_bb, rho_sc, Q, qf_list, qb_list, q_sc_forward, q_sc_bb_full


'''def compute_persistence_length_function(q_bb_fw, q_bb_bw, u_grid, n_quad_per_rod, N, Q, spat_weights):
    #We compute the tangent-tangent correlation fixing s at the start and sliding via that
    for idxN in range(N):
        for idxs in range(n_quad_per_rod):
            tgt_tgt_corr += (1/Q)*ds*np.sum(q_bb_fw[idxN][idxs]*q_bb_bw[idxN][idxs]*spat_weights, axis = (0,1))*
'''