"""
py_models.py  —  Clay p-y Curve Models
========================================
Internal unit system: kip, in throughout.

Implemented models (RSPile Theory Manual, Rocscience 2022):
  1. 'matlock'           Matlock (1970)       soft clay with free water    §5.1
  2. 'reese_stiff_water' Reese et al. (1975)  stiff clay with free water   §5.2
  3. 'welch_reese'       Welch & Reese (1972) stiff clay without free water §5.3

Interface
---------
    p, k_sec, k_tan = get_soil_response(y_in, z_in, params, model)

    y_in  : float  [in]       lateral pile deflection (signed)
    z_in  : float  [in]       depth below ground surface
    params: dict              model-specific parameters (kip-in, see below)
    model : str               one of the keys above

Returns
-------
    p     : float  [kip/in]   soil resistance (+ in load direction, resists displacement)
    k_sec : float  [kip/in²]  secant modulus  = |p| / |y|
    k_tan : float  [kip/in²]  tangent modulus = dp/dy

Common parameter keys (all kip-in)
------------------------------------
    'cu_ksi'         float    undrained shear strength at depth z  [kip/in²]
    'ca_ksi'         float    average undrained shear strength     [kip/in²]  (≈ cu for uniform)
    'eps50'          float    axial strain at 50% deviatoric stress [-]
    'gamma_kip_in3'  float    effective unit weight                 [kip/in³]
    'J'              float    Matlock empirical factor (default 0.5)
    'ks_kip_in3'     float    initial stiffness gradient for Reese  [kip/in³]
    'loading'        str      'static' or 'cyclic' (Reese only)
    'b_in'           float    pile diameter                         [in]

Profiles: any scalar parameter can alternatively be supplied as a (N,2) array
    [[z0_in, val0], [z1_in, val1], ...] for depth-varying input (linear interp).

Unit conversion reminders (for callers)
----------------------------------------
    cu  : 1 ksf  = 1/144  kip/in²  ≈ 6.944e-3 kip/in²
    γ'  : 1 pcf  = 1/(1000×1728) kip/in³ ≈ 5.787e-7 kip/in³
    ks  : 1 pci  = 1/1000 kip/in³
    z   : 1 ft   = 12 in
    b   : already in inches
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
_Y_ZERO_TOL  = 1.0e-12   # [in]  |y| below this treated as zero
_Y_INIT_FRAC = 1.0e-2    # regularisation: k_sec(y=0) evaluated at _Y_INIT_FRAC·y50
                          # This is a NUMERICAL CONVENIENCE, not the theoretical tangent.

# 2-point Gauss quadrature constants (also used by fem_solver)
_INV_SQRT3 = 1.0 / np.sqrt(3.0)
_XI_GP     = np.array([-_INV_SQRT3, +_INV_SQRT3])
_W_GP      = np.array([1.0, 1.0])

# A-factors for Reese stiff clay — digitised from Figure 5.3 (Reese & Van Impe, 2011)
# x-axis = z/b;  clamped to 0.88 at large z/b for both curves
_AS_ZB  = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 100.0])
_AS_VAL = np.array([0.23, 0.27, 0.32, 0.38, 0.46, 0.60, 0.73, 0.87,  0.88])  # static

_AC_ZB  = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 100.0])
_AC_VAL = np.array([0.17, 0.19, 0.22, 0.26, 0.30, 0.38, 0.48, 0.56, 0.65, 0.80, 0.88])  # cyclic


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _get_param(z, params, key):
    """
    Return scalar soil parameter at depth z [in].
    Accepts:  scalar value  or  (N,2) ndarray [[z0,v0],[z1,v1],...] (linear interp).
    """
    if key not in params:
        raise KeyError(f"Required parameter '{key}' not found in soil params dict.")
    v = params[key]
    if np.ndim(v) == 0:
        return float(v)
    prof = np.asarray(v, dtype=float)
    return float(np.interp(z, prof[:, 0], prof[:, 1]))


def _sign_and_abs(y):
    return (1.0 if y >= 0.0 else -1.0), abs(y)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — MATLOCK (1970)  Soft Clay with Free Water  §5.1
# ══════════════════════════════════════════════════════════════════════════════
def _matlock_soft_clay(y, z, params):
    """
    RSPile §5.1  (Matlock, 1970)

    Reference deflection (CORRECTED):
        y50 = 2.5 · ε50 · b                                         [in]

    Ultimate resistance (minimum of shallow/deep):
        p_ult_s = [3 + (γ'/cu)·z + (J/b)·z] · cu · b               (Eq.9)
        p_ult_d = 9 · cu · b                                         (Eq.10)

    Curve:
        p = 0.5·p_ult·(y/y50)^(1/3)   for y ≤ 8·y50
        p = p_ult                       for y > 8·y50
    """
    cu    = _get_param(z, params, 'cu_ksi')
    eps50 = float(params['eps50'])
    gamma = float(params['gamma_kip_in3'])
    b     = float(params['b_in'])
    J     = float(params.get('J', 0.5))

    y50   = 2.5 * eps50 * b
    y_lim = 8.0 * y50

    p_ult_s = (3.0 + (gamma / cu) * z + (J / b) * z) * cu * b
    p_ult_d = 9.0 * cu * b
    p_ult   = max(min(p_ult_s, p_ult_d), 0.0)

    y_sign, abs_y = _sign_and_abs(y)

    # y ≈ 0: numerical regularisation (k_sec evaluated at y_eval = 1% of y50)
    if abs_y <= _Y_ZERO_TOL:
        y_ev  = _Y_INIT_FRAC * max(y50, 1e-15)
        p_ref = 0.5 * p_ult * (y_ev / y50) ** (1.0 / 3.0) if y50 > 0.0 else 0.0
        k_sec = p_ref / y_ev if y_ev > 0.0 else 0.0
        return 0.0, k_sec, (1.0 / 3.0) * k_sec

    if abs_y <= y_lim:
        p_mag = 0.5 * p_ult * (abs_y / y50) ** (1.0 / 3.0)
        k_sec = p_mag / abs_y
        k_tan = (1.0 / 3.0) * k_sec   # exact: d/dy [C·y^(1/3)] = (1/3)·p/y
    else:
        p_mag = p_ult
        k_sec = p_ult / abs_y
        k_tan = 0.0

    return y_sign * p_mag, k_sec, k_tan


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — REESE et al. (1975)  Stiff Clay with Free Water  §5.2
# ══════════════════════════════════════════════════════════════════════════════
def _reese_stiff_clay_water(y, z, params):
    """
    RSPile §5.2  (Reese et al., 1975)

    Reference deflection:
        y50 = ε50 · b    (no 2.5 factor — stiff overconsolidated clay)

    Ultimate resistance parameter (Eqs. 11–12):
        pc_shallow = 2·ca·b + γ'·b·z + 2.83·ca·z    (Eq.11)
        pc_deep    = 11·cu·b                          (Eq.12)
        pc         = min(pc_shallow, pc_deep)

    A-factor: depth-dependent from Figure 5.3 (Reese & Van Impe, 2011)
        As (static loading) or Ac (cyclic loading)

    Curve segments:
        1. Linear initial:  p = Esi·y           (0 ≤ y ≤ y_k,  Esi = ks·z)
        2. Parabolic:       p = 0.5·pc·(y/y50)^½  (y_k ≤ y ≤ 6A·y50)
        3. Declining:       p = p_B + Ess·(y−6A·y50)  (Ess = −0.0625·pc/y50)
        4. Plateau:         p = p_C             (y > 18A·y50)

    Notes:
    - At z = 0, Esi = 0 → curve starts directly with the parabolic branch.
    - The descending slope Ess is negative (softening). k_tan set to 0 there
      for secant-iteration stability.
    """
    cu      = _get_param(z, params, 'cu_ksi')
    ca      = _get_param(z, params, 'ca_ksi')
    eps50   = float(params['eps50'])
    gamma   = float(params['gamma_kip_in3'])
    b       = float(params['b_in'])
    ks      = float(params['ks_kip_in3'])
    loading = str(params.get('loading', 'static')).lower()

    y50 = eps50 * b   # [in]

    # pc  (Eqs. 11–12)
    pc_s = 2.0 * ca * b + gamma * b * z + 2.83 * ca * z
    pc_d = 11.0 * cu * b
    pc   = max(min(pc_s, pc_d), 0.0)

    # A factor
    zb = z / b if b > 0.0 else 0.0
    A  = (np.interp(zb, _AS_ZB, _AS_VAL) if loading == 'static'
          else np.interp(zb, _AC_ZB, _AC_VAL))

    # Initial stiffness (zero at surface)
    Esi = ks * z   # [kip/in²]

    # Key deflection breakpoints
    y_A = A * y50
    y_B = 6.0 * A * y50
    y_C = 18.0 * A * y50

    # Key resistance values
    p_B = 0.5 * pc * np.sqrt(max(6.0 * A, 0.0))          # at peak
    Ess = -0.0625 * pc / y50 if y50 > 0.0 else 0.0        # declining slope [kip/in²]
    p_C = max(p_B + Ess * (y_C - y_B), 0.0)               # residual plateau

    y_sign, abs_y = _sign_and_abs(y)

    # y ≈ 0 ──────────────────────────────────────────────────────────────────
    if abs_y <= _Y_ZERO_TOL:
        if Esi > 0.0:
            # Linear initial portion exists; k_sec = Esi is well-defined
            return 0.0, Esi, Esi
        else:
            # z ≈ 0: parabolic start; regularise
            y_ev  = _Y_INIT_FRAC * max(y50, 1e-15)
            p_ref = 0.5 * pc * np.sqrt(y_ev / y50) if y50 > 0.0 else 0.0
            k_sec = p_ref / y_ev if y_ev > 0.0 else 0.0
            return 0.0, k_sec, 0.5 * k_sec

    # transition point (linear → parabola)
    if Esi > 0.0 and y50 > 0.0 and pc > 0.0:
        y_k = 0.25 * pc ** 2 / (Esi ** 2 * y50)
    else:
        y_k = 0.0
    y_k = min(y_k, y_A)

    # Evaluate p ─────────────────────────────────────────────────────────────
    if abs_y <= y_k:
        p_mag = Esi * abs_y
        k_sec = Esi
        k_tan = Esi
    elif abs_y <= y_B:
        p_mag = 0.5 * pc * np.sqrt(abs_y / y50)
        k_sec = p_mag / abs_y
        k_tan = 0.5 * k_sec           # d/dy[C·y^½] = (1/2)·p/y
    elif abs_y <= y_C:
        p_mag = max(p_B + Ess * (abs_y - y_B), 0.0)
        k_sec = p_mag / abs_y if abs_y > 0.0 else 0.0
        k_tan = 0.0                   # softening; treat as zero for stability
    else:
        p_mag = p_C
        k_sec = p_C / abs_y if abs_y > 0.0 else 0.0
        k_tan = 0.0

    return y_sign * p_mag, k_sec, k_tan


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — WELCH & REESE (1972)  Stiff Clay WITHOUT Free Water  §5.3
# ══════════════════════════════════════════════════════════════════════════════
def _welch_reese_stiff_clay_nowater(y, z, params):
    """
    RSPile §5.3  (Welch & Reese, 1972)

    Reference deflection:
        y50 = ε50 · b   (same formula as Matlock without 2.5 factor)

    Ultimate resistance (Eqs. 13–14):
        p_ult_s = [3 + (γ'/ca)·z + (J/b)·z] · ca · b   (Eq.13)
        p_ult_d = 9 · cu · b                              (Eq.14)
        p_ult   = min(p_ult_s, p_ult_d)

    Curve (quarter-root, Figure 5.4):
        p = 0.5·p_ult·(y/y50)^(1/4)   for y ≤ 16·y50
        p = p_ult                       for y > 16·y50
    """
    cu    = _get_param(z, params, 'cu_ksi')
    ca    = _get_param(z, params, 'ca_ksi')
    eps50 = float(params['eps50'])
    gamma = float(params['gamma_kip_in3'])
    b     = float(params['b_in'])
    J     = float(params.get('J', 0.5))

    y50   = eps50 * b
    y_lim = 16.0 * y50

    p_ult_s = (3.0 + (gamma / ca) * z + (J / b) * z) * ca * b
    p_ult_d = 9.0 * cu * b
    p_ult   = max(min(p_ult_s, p_ult_d), 0.0)

    y_sign, abs_y = _sign_and_abs(y)

    if abs_y <= _Y_ZERO_TOL:
        y_ev  = _Y_INIT_FRAC * max(y50, 1e-15)
        p_ref = 0.5 * p_ult * (y_ev / y50) ** (1.0 / 4.0) if y50 > 0.0 else 0.0
        k_sec = p_ref / y_ev if y_ev > 0.0 else 0.0
        return 0.0, k_sec, (1.0 / 4.0) * k_sec

    if abs_y <= y_lim:
        p_mag = 0.5 * p_ult * (abs_y / y50) ** (1.0 / 4.0)
        k_sec = p_mag / abs_y
        k_tan = (1.0 / 4.0) * k_sec   # d/dy[C·y^(1/4)] = (1/4)·p/y
    else:
        p_mag = p_ult
        k_sec = p_ult / abs_y
        k_tan = 0.0

    return y_sign * p_mag, k_sec, k_tan


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════════════════
_MODEL_MAP = {
    'matlock':             _matlock_soft_clay,
    'reese_stiff_water':   _reese_stiff_clay_water,
    'welch_reese':         _welch_reese_stiff_clay_nowater,
}

MODEL_LABELS = {
    'matlock':            'Matlock (1970) — Soft Clay w/ Water',
    'reese_stiff_water':  'Reese et al. (1975) — Stiff Clay w/ Water',
    'welch_reese':        'Welch & Reese (1972) — Stiff Clay w/o Water',
}

MODEL_KEYS = list(_MODEL_MAP.keys())


def get_soil_response(y, z, params, model='matlock'):
    """
    Central dispatch.  Returns (p [kip/in], k_sec [kip/in²], k_tan [kip/in²]).
    """
    fn = _MODEL_MAP.get(model)
    if fn is None:
        raise NotImplementedError(
            f"Soil model '{model}' not implemented. "
            f"Available: {MODEL_KEYS}")
    return fn(y, z, params)


def compute_pult(z, params, model):
    """
    Return p_ult [kip/in] at depth z for mobilisation ratio post-processing.
    Returns 0.0 for unknown models.
    """
    b  = float(params.get('b_in', 1.0))
    cu = _get_param(z, params, 'cu_ksi')

    if model == 'matlock':
        gamma = float(params['gamma_kip_in3'])
        J     = float(params.get('J', 0.5))
        p_s   = (3.0 + (gamma / cu) * z + (J / b) * z) * cu * b
        return max(min(p_s, 9.0 * cu * b), 0.0)

    elif model == 'reese_stiff_water':
        ca    = _get_param(z, params, 'ca_ksi')
        gamma = float(params['gamma_kip_in3'])
        p_s   = 2.0 * ca * b + gamma * b * z + 2.83 * ca * z
        return max(min(p_s, 11.0 * cu * b), 0.0)

    elif model == 'welch_reese':
        ca    = _get_param(z, params, 'ca_ksi')
        gamma = float(params['gamma_kip_in3'])
        J     = float(params.get('J', 0.5))
        p_s   = (3.0 + (gamma / ca) * z + (J / b) * z) * ca * b
        return max(min(p_s, 9.0 * cu * b), 0.0)

    return 0.0


def get_py_curve(z, params, model, n_pts=300):
    """
    Return (y_in, p_kipin) arrays for the p-y curve at depth z.
    Useful for plotting p-y curves at selected depths.
    """
    b    = float(params.get('b_in', 1.0))
    eps50 = float(params.get('eps50', 0.005))

    if model == 'matlock':
        y50   = 2.5 * eps50 * b
        y_max = max(12.0 * y50, 0.5)
    elif model in ('reese_stiff_water', 'welch_reese'):
        y50   = eps50 * b
        # For Reese, extend to cover the full softening + plateau
        if model == 'reese_stiff_water':
            from py_models import _AS_ZB, _AS_VAL
            zb  = z / b if b > 0 else 0.0
            A   = np.interp(zb, _AS_ZB, _AS_VAL)
            y_max = max(22.0 * A * y50, 0.5)
        else:
            y_max = max(20.0 * y50, 0.5)
    else:
        y_max = 1.0

    y_arr = np.linspace(0.0, y_max, n_pts)
    p_arr = np.array([get_soil_response(yv, z, params, model)[0] for yv in y_arr])
    return y_arr, p_arr
