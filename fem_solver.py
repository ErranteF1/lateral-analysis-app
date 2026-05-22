"""
fem_solver.py  —  Hermitian Beam-on-Winkler-Foundation FEM Solver
==================================================================
Internal unit system: kip, in throughout.

Theory: RSPile Laterally Loaded Piles — Theory Manual (Rocscience, 2022)
        Hetenyi (1946)  governing ODE for beam-column on Winkler foundation

Key features
------------
- Euler-Bernoulli beam, 2 DOFs/node: lateral deflection y [in], rotation θ [rad]
- Hermitian cubic shape functions
- 2-point Gauss quadrature for Winkler soil stiffness and shear recovery
- Secant stiffness iteration (p-y method engine)
- Layered soil profile: arbitrary number of layers, each with its own p-y model
- Fixed or free pile head; free/pinned/fixed tip
- Returns all internal forces and displacements at every node

Sign conventions
----------------
    z  → positive downward (0 = head, L = tip)
    y  → positive in direction of applied lateral force H
    θ  = dy/dz
    M  = EI · d²y/dz²   (positive = tension on +z face)
    V  = EI · d³y/dz³   (V(0) = H_head by construction)
    p  → acts opposite to y (resists deflection); dV/dz = −p

All kip-in unless noted.
"""

import numpy as np
import warnings
from py_models import get_soil_response, compute_pult, _XI_GP, _W_GP

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — GAUSS POINT AND SHAPE FUNCTION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def hermitian_shape_N(zeta, Le):
    """
    Hermitian cubic shape functions at normalised coordinate zeta = s/Le ∈ [0,1].
    DOF order: [y_i, θ_i, y_j, θ_j].
    Returns ndarray N[4].
    """
    z2 = zeta ** 2
    z3 = zeta ** 3
    return np.array([
        1.0 - 3.0 * z2 + 2.0 * z3,          # N1 — deflection at node i
        Le  * zeta * (1.0 - zeta) ** 2,      # N2 — rotation    at node i
        3.0 * z2 - 2.0 * z3,                 # N3 — deflection at node j
        Le  * z2   * (zeta - 1.0)            # N4 — rotation    at node j
    ])


def _gauss_points_on_element(Le):
    """
    2-point Gauss data on element [0, Le].
    Returns (s_gp [in],  zeta_gp [-])  each shape (2,).
    Mapping: ξ ∈ [−1,1] → s = (Le/2)·(ξ+1)
    """
    s_gp    = (Le / 2.0) * (_XI_GP + 1.0)
    zeta_gp = s_gp / Le
    return s_gp, zeta_gp


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ELEMENT STIFFNESS MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def beam_element_stiffness(EI, Le):
    """
    4×4 Euler-Bernoulli beam stiffness matrix.
    DOF order: [y_i, θ_i, y_j, θ_j].  Units: kip/in, kip, kip·in, ...
    """
    c = EI / Le ** 3
    L = Le
    return c * np.array([
        [ 12.0,    6.0*L,   -12.0,    6.0*L],
        [  6.0*L,  4.0*L**2, -6.0*L,  2.0*L**2],
        [-12.0,   -6.0*L,    12.0,   -6.0*L],
        [  6.0*L,  2.0*L**2, -6.0*L,  4.0*L**2]
    ])


def soil_element_stiffness_gauss(k_gp1, k_gp2, Le):
    """
    4×4 Gauss-integrated Winkler soil stiffness matrix.

        K_soil,e = (Le/2) · Σ_g  w_g · N(ζ_g)^T ⊗ N(ζ_g) · k_sec(ζ_g)

    Parameters
    ----------
    k_gp1 : float  [kip/in²]  secant stiffness at Gauss point 1
    k_gp2 : float  [kip/in²]  secant stiffness at Gauss point 2
    Le    : float  [in]        element length

    Returns
    -------
    K_soil : ndarray (4,4)  [kip/in]
    """
    Jac    = Le / 2.0
    K_soil = np.zeros((4, 4))
    s_gp, zeta_gp = _gauss_points_on_element(Le)
    for g, k_s in enumerate([k_gp1, k_gp2]):
        N = hermitian_shape_N(zeta_gp[g], Le)
        K_soil += Jac * _W_GP[g] * k_s * np.outer(N, N)
    return K_soil


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LAYERED SOIL INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def _find_layer(z, layers):
    """
    Return the layer dict whose [z_top_in, z_bot_in) bracket contains depth z.
    Falls back to the last layer for z beyond the defined profile.

    layers: list of dicts with keys 'z_top_in', 'z_bot_in', 'model', 'params'
    """
    for layer in layers:
        if layer['z_top_in'] <= z < layer['z_bot_in']:
            return layer
    return layers[-1]   # extend last layer to pile tip


def soil_response_layered(y, z, layers):
    """
    Evaluate p-y response at depth z using the correct soil layer.
    Returns (p [kip/in], k_sec [kip/in²], k_tan [kip/in²]).
    """
    lyr = _find_layer(z, layers)
    return get_soil_response(y, z, lyr['params'], lyr['model'])


def pult_layered(z, layers):
    """Return p_ult [kip/in] at depth z for post-processing."""
    lyr = _find_layer(z, layers)
    return compute_pult(z, lyr['params'], lyr['model'])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SECANT STIFFNESS UPDATE (PER-ELEMENT, 2 GAUSS POINTS)
# ══════════════════════════════════════════════════════════════════════════════

def _initial_k_sec_gp(n_elements, z_nodes, layers):
    """
    k_sec at y = 0 for all Gauss points → shape (n_elements, 2).
    Calls soil_response_layered(y=0, z_g) for each Gauss point.
    """
    k_gp = np.zeros((n_elements, 2))
    for e in range(n_elements):
        Le_e = z_nodes[e + 1] - z_nodes[e]
        s_gp, _ = _gauss_points_on_element(Le_e)
        for g in range(2):
            z_g = z_nodes[e] + s_gp[g]
            _, k_s, _ = soil_response_layered(0.0, z_g, layers)
            k_gp[e, g] = k_s
    return k_gp


def _update_k_sec_gp(u, n_elements, z_nodes, layers):
    """
    k_sec at the converging deflection field u → shape (n_elements, 2).
    Interpolates y at each Gauss point via Hermitian cubic field.
    """
    k_gp = np.zeros((n_elements, 2))
    for e in range(n_elements):
        Le_e   = z_nodes[e + 1] - z_nodes[e]
        u_e    = np.array([u[2*e], u[2*e+1], u[2*(e+1)], u[2*(e+1)+1]])
        s_gp, zeta_gp = _gauss_points_on_element(Le_e)
        for g in range(2):
            N   = hermitian_shape_N(zeta_gp[g], Le_e)
            y_g = np.dot(N, u_e)
            z_g = z_nodes[e] + s_gp[g]
            _, k_s, _ = soil_response_layered(y_g, z_g, layers)
            k_gp[e, g] = k_s
    return k_gp


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GLOBAL ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def assemble_global_stiffness(n_elements, n_dof, EI_array, k_sec_gp, Le_array):
    """
    Assemble K = K_beam + K_soil using Gauss-integrated soil matrices.

    Parameters
    ----------
    n_elements  : int
    n_dof       : int
    EI_array    : float or (n_elements,) [kip·in²]
    k_sec_gp    : ndarray (n_elements, 2)  [kip/in²]  secant at 2 GPs per element
    Le_array    : float or (n_elements,) [in]
    """
    EI_v = np.broadcast_to(np.atleast_1d(EI_array), (n_elements,))
    Le_v = np.broadcast_to(np.atleast_1d(Le_array), (n_elements,))
    k_gp = np.asarray(k_sec_gp)

    K = np.zeros((n_dof, n_dof))
    for e in range(n_elements):
        K_e = (beam_element_stiffness(EI_v[e], Le_v[e]) +
               soil_element_stiffness_gauss(k_gp[e, 0], k_gp[e, 1], Le_v[e]))
        dof = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for r in range(4):
            for c in range(4):
                K[dof[r], dof[c]] += K_e[r, c]
    return K


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BOUNDARY CONDITIONS AND LOAD VECTOR
# ══════════════════════════════════════════════════════════════════════════════

def build_fixed_dofs(n_elements, head_fixed, tip_condition):
    """
    Return list of globally constrained DOF indices.

    head_fixed     : bool    True = fixed head (θ(0) = 0 → DOF 1 fixed)
    tip_condition  : str     'free' | 'pinned' | 'fixed'
    """
    fixed = []
    if head_fixed:
        fixed.append(1)    # θ at node 0

    n = n_elements         # last node index
    if tip_condition == 'pinned':
        fixed.append(2 * n)
    elif tip_condition == 'fixed':
        fixed.extend([2 * n, 2 * n + 1])
    return fixed


def build_load_vector(n_dof, H_kip, M_kip_in=0.0):
    """F[0] = H (shear at head) [kip],  F[1] = M (moment at head) [kip·in]."""
    F = np.zeros(n_dof)
    F[0] = H_kip
    F[1] = M_kip_in
    return F


def apply_boundary_conditions(K_global, F, fixed_dofs):
    """Zeroing-and-diagonal method; preserves symmetry."""
    Km = K_global.copy()
    Fm = F.copy()
    for d in fixed_dofs:
        Km[d, :] = 0.0
        Km[:, d] = 0.0
        Km[d, d] = 1.0
        Fm[d] = 0.0
    return Km, Fm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — NONLINEAR SOLVER (SECANT STIFFNESS ITERATION)
# ══════════════════════════════════════════════════════════════════════════════

def solve_nonlinear(pile_params, soil_layers, H_kip, M_kip_in=0.0,
                    max_iter=150, tol=1.0e-6, relax=1.0, verbose=False):
    """
    Secant stiffness p-y iteration.

    Algorithm
    ---------
    1. k_sec_gp ← initial_k_sec(y=0)
    2. For it = 1 .. max_iter:
         K   ← assemble(EI, k_sec_gp, Le)
         u   ← solve(K_bc, F_bc)
         u   ← relax·u + (1−relax)·u_prev
         Δu  ← max|y_new − y_prev|  (lateral DOFs only)
         if Δu < tol: converged
         k_sec_gp ← update_k_sec(u)

    Parameters
    ----------
    pile_params : dict
        'L_in'       float   pile length [in]
        'EI_kip_in2' float   bending stiffness [kip·in²]
        'n_elements' int     FEM mesh density
        'head_fixed' bool    True = fixed head (θ=0)
        'tip_cond'   str     'free' | 'pinned' | 'fixed'

    soil_layers : list of dicts  (see _find_layer docstring)
    H_kip       : float   lateral shear at head [kip]
    M_kip_in    : float   moment at head [kip·in]

    Returns
    -------
    dict with keys:
        'converged', 'n_iter', 'conv_hist',
        'u', 'z_nodes', 'Le', 'k_sec_gp'
    """
    L        = float(pile_params['L_in'])
    EI       = float(pile_params['EI_kip_in2'])
    n_el     = int(pile_params['n_elements'])
    head_fix = bool(pile_params.get('head_fixed', False))
    tip_cond = str(pile_params.get('tip_cond', 'free'))

    n_nodes  = n_el + 1
    n_dof    = 2 * n_nodes
    Le       = L / n_el
    z_nodes  = np.linspace(0.0, L, n_nodes)
    fixed    = build_fixed_dofs(n_el, head_fix, tip_cond)
    F        = build_load_vector(n_dof, H_kip, M_kip_in)

    k_sec_gp = _initial_k_sec_gp(n_el, z_nodes, soil_layers)
    u_prev   = np.zeros(n_dof)
    conv_hist = []
    converged = False

    if verbose:
        print(f"  Solver: H={H_kip:.3f} kip  M={M_kip_in:.3f} kip·in  "
              f"n_el={n_el}  tol={tol:.1e}")

    for it in range(1, max_iter + 1):
        K      = assemble_global_stiffness(n_el, n_dof, EI, k_sec_gp, Le)
        Km, Fm = apply_boundary_conditions(K, F, fixed)
        u_raw  = np.linalg.solve(Km, Fm)
        u_new  = relax * u_raw + (1.0 - relax) * u_prev

        delta = float(np.max(np.abs(u_new[0::2] - u_prev[0::2])))
        conv_hist.append(delta)

        if delta < tol:
            converged = True
            break

        k_sec_gp = _update_k_sec_gp(u_new, n_el, z_nodes, soil_layers)
        u_prev   = u_new.copy()
    else:
        warnings.warn(
            f"Secant iteration did not converge in {max_iter} iterations "
            f"(max|Δy|={conv_hist[-1]:.3e} in > tol={tol:.1e} in).",
            RuntimeWarning, stacklevel=2)

    return {
        'converged':  converged,
        'n_iter':     it,
        'conv_hist':  conv_hist,
        'u':          u_new,
        'z_nodes':    z_nodes,
        'Le':         Le,
        'k_sec_gp':   k_sec_gp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — INTERNAL FORCE RECOVERY
# ══════════════════════════════════════════════════════════════════════════════

def recover_internal_forces(sol, pile_params, soil_layers, H_kip):
    """
    Recover M(z) and V(z) at all nodes from converged displacement field.

    Moment: M = EI·d²y/dz² via Hermitian B2 operator (averaged at shared nodes).
    Shear:  V[e+1] = V[e] − (Le/2)·Σ_g p(y(ζ_g), z_g)
            (integrates actual p-y curve response at Gauss points)

    Returns  M_nodes [kip·in],  V_nodes [kip]
    """
    u       = sol['u']
    z_nodes = sol['z_nodes']
    Le      = sol['Le']
    EI      = float(pile_params['EI_kip_in2'])
    n_el    = len(z_nodes) - 1
    n_nodes = n_el + 1

    # ── Moment ──────────────────────────────────────────────────────────────
    B2_L = np.array([-6.0 / Le**2, -4.0 / Le,  6.0 / Le**2, -2.0 / Le])
    B2_R = np.array([ 6.0 / Le**2,  2.0 / Le, -6.0 / Le**2,  4.0 / Le])

    M_L = np.zeros(n_el)
    M_R = np.zeros(n_el)
    for e in range(n_el):
        u_e    = np.array([u[2*e], u[2*e+1], u[2*(e+1)], u[2*(e+1)+1]])
        M_L[e] = EI * np.dot(B2_L, u_e)
        M_R[e] = EI * np.dot(B2_R, u_e)

    M_nodes    = np.zeros(n_nodes)
    M_nodes[0] = M_L[0]
    for e in range(1, n_el):
        M_nodes[e] = 0.5 * (M_R[e - 1] + M_L[e])
    M_nodes[-1] = M_R[-1]

    # ── Shear (integrate actual p(y,z) at Gauss points) ──────────────────
    Jac     = Le / 2.0
    V_nodes = np.zeros(n_nodes)
    V_nodes[0] = H_kip

    for e in range(n_el):
        z_i   = z_nodes[e]
        u_e   = np.array([u[2*e], u[2*e+1], u[2*(e+1)], u[2*(e+1)+1]])
        s_gp, zeta_gp = _gauss_points_on_element(Le)
        p_sum = 0.0
        for g in range(2):
            N   = hermitian_shape_N(zeta_gp[g], Le)
            y_g = np.dot(N, u_e)
            z_g = z_i + s_gp[g]
            p_g, _, _ = soil_response_layered(y_g, z_g, soil_layers)
            p_sum += _W_GP[g] * p_g
        V_nodes[e + 1] = V_nodes[e] - Jac * p_sum

    return M_nodes, V_nodes


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def extract_results(sol, pile_params, soil_layers, H_kip, M_kip_in,
                    case_name=''):
    """
    Full post-processing: returns all profiles and key values in kip-in.
    Caller converts to display units.

    Returns dict with all arrays in kip-in and z in inches.
    """
    u       = sol['u']
    z_nodes = sol['z_nodes']   # [in]
    n_nodes = len(z_nodes)

    y_nodes = u[0::2]          # lateral deflection [in]
    t_nodes = u[1::2]          # rotation           [rad]

    M_nodes, V_nodes = recover_internal_forces(sol, pile_params, soil_layers, H_kip)

    # Nodal p and p_ult
    p_nodes    = np.zeros(n_nodes)
    pult_nodes = np.zeros(n_nodes)
    for n in range(n_nodes):
        p_nodes[n], _, _ = soil_response_layered(y_nodes[n], z_nodes[n], soil_layers)
        pult_nodes[n]    = pult_layered(z_nodes[n], soil_layers)

    # Mobilisation ratio p/p_ult
    ppult = np.where(pult_nodes > 0.0, p_nodes / pult_nodes, 0.0)

    # Key scalar values
    imax    = np.argmax(np.abs(M_nodes))
    M_max   = float(np.max(np.abs(M_nodes)))
    z_Mmax  = float(z_nodes[imax])

    return {
        'name':        case_name,
        'converged':   sol['converged'],
        'n_iter':      sol['n_iter'],
        'conv_hist':   sol['conv_hist'],
        # profiles (kip-in)
        'z_in':        z_nodes,
        'y_in':        y_nodes,
        'theta_rad':   t_nodes,
        'M_kip_in':    M_nodes,
        'V_kip':       V_nodes,
        'p_kip_in':    p_nodes,
        'pult_kip_in': pult_nodes,
        'p_over_pult': ppult,
        # scalars
        'y_head_in':   float(y_nodes[0]),
        'y_tip_in':    float(y_nodes[-1]),
        'M_head_kip_in': float(M_nodes[0]),
        'M_max_kip_in':  M_max,
        'z_Mmax_in':     z_Mmax,
        'V_head_kip':    float(V_nodes[0]),
        'V_tip_kip':     float(V_nodes[-1]),
        'H_applied_kip': H_kip,
        'M_applied_kip_in': M_kip_in,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TOP-LEVEL ANALYSIS RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(pile_params, soil_layers, load_cases,
                 max_iter=150, tol=1.0e-6, relax=1.0):
    """
    Run the full nonlinear p-y analysis for a list of load cases.

    Parameters
    ----------
    pile_params  : dict  (see solve_nonlinear)
    soil_layers  : list of layer dicts  (kip-in params)
    load_cases   : list of dicts {'name', 'H_kip', 'M_kip_in'}
    max_iter     : int
    tol          : float  [in]   convergence criterion on max|Δy|
    relax        : float  [0,1]  under-relaxation factor (1 = no relaxation)

    Returns
    -------
    list of result dicts (one per load case), each from extract_results()
    """
    results = []
    for lc in load_cases:
        sol = solve_nonlinear(
            pile_params, soil_layers,
            lc['H_kip'], lc.get('M_kip_in', 0.0),
            max_iter=max_iter, tol=tol, relax=relax)
        res = extract_results(
            sol, pile_params, soil_layers,
            lc['H_kip'], lc.get('M_kip_in', 0.0),
            case_name=lc.get('name', ''))
        results.append(res)
    return results
