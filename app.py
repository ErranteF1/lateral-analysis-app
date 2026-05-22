"""
app.py  —  Lateral Pile Analysis — Clay  (Streamlit)
=====================================================
Local interactive application for single-pile lateral load analysis in clay.

Soil models:
    1. Matlock (1970)       — Soft Clay with Free Water
    2. Reese et al. (1975)  — Stiff Clay with Free Water
    3. Welch & Reese (1972) — Stiff Clay without Free Water

Units — ALL USER-FACING I/O IN IMPERIAL:
    Length / depth     ft
    Pile diameter      in
    Pile stiffness E   ksi
    EI display         kip·ft²
    Force              kip
    Moment             kip·ft
    Stress / strength  ksf
    Unit weight        pcf
    k_s (Reese)        pci  (lb/in³)
    Deflection         in
    Rotation           degrees
    Soil reaction p    kip/ft

Internal solver: kip-in throughout (conversions handled at I/O boundary).

Run with:
    streamlit run app.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

# ── locate py_models and fem_solver relative to this file ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from py_models import (get_soil_response, get_py_curve, compute_pult,
                       MODEL_LABELS, MODEL_KEYS)
from fem_solver import run_analysis

# ══════════════════════════════════════════════════════════════════════════════
# UNIT CONVERSIONS  (all to/from internal kip-in)
# ══════════════════════════════════════════════════════════════════════════════
# 1 ft     = 12 in
# 1 ksf    = 1 kip/ft² = (1/144) kip/in²
# 1 pcf    = 1 lb/ft³  = (1/1000)kip / (12³ in³) = 5.787e-7 kip/in³
# 1 pci    = 1 lb/in³  = (1/1000) kip/in³
# 1 kip·ft = 12 kip·in
# 1 kip·ft²= 144 kip·in²
# 1 kip/ft = (1/12) kip/in

FT2IN     = 12.0
KSF2KSI   = 1.0 / 144.0            # kip/ft² → kip/in²
PCF2KIPIN3 = 1.0 / (1000.0 * 1728.0)  # lb/ft³  → kip/in³
PCI2KIPIN3 = 1.0 / 1000.0          # lb/in³  → kip/in³
KIPFT2KIPIN = 12.0                  # kip·ft  → kip·in
KIPFT22KIPIN2 = 144.0              # kip·ft² → kip·in²

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LateralPile — Clay Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:1.6rem; font-weight:700; color:#1a3a5c; margin-bottom:4px;}
    .sub-header  {font-size:0.88rem; color:#555; margin-bottom:18px;}
    .result-card {background:#f0f4f9; border-radius:6px; padding:12px 16px;
                  border-left:4px solid #1a6ecf; margin-bottom:8px;}
    .warn-card   {background:#fff8e1; border-radius:6px; padding:10px 14px;
                  border-left:4px solid #f0a500; margin-bottom:8px;}
    .ok-card     {background:#e8f5e9; border-radius:6px; padding:10px 14px;
                  border-left:4px solid #2e7d32; margin-bottom:8px;}
    div[data-testid="stExpander"] summary {font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULT_LAYERS = pd.DataFrame({
    'Layer':    [1, 2],
    'Top (ft)': [0.0, 15.0],
    'Bot (ft)': [15.0, 40.0],
    'Model':    ['Matlock (1970) — Soft Clay w/ Water',
                 'Welch & Reese (1972) — Stiff Clay w/o Water'],
    'su (ksf)': [1.0, 3.0],
    'ε50':      [0.020, 0.005],
    "γ' (pcf)": [60.0, 62.4],
    'J':        [0.5, 0.5],
    'ks (pci)': [500.0, 2000.0],
    'Loading':  ['Static', 'Static'],
})

_DEFAULT_LOADS = pd.DataFrame({
    'Case':     ['LC-1', 'LC-2'],
    'H (kip)':  [100.0, 200.0],
    'M (kip·ft)':[0.0,   0.0],
})

_DEFAULT_COMBOS = pd.DataFrame({
    'Combo': ['COMB-1'],
    'LC-1':  [1.0],
    'LC-2':  [1.0],
})

def _init_state():
    if 'layers_df' not in st.session_state:
        st.session_state.layers_df = _DEFAULT_LAYERS.copy()
    if 'loads_df' not in st.session_state:
        st.session_state.loads_df = _DEFAULT_LOADS.copy()
    if 'combos_df' not in st.session_state:
        st.session_state.combos_df = _DEFAULT_COMBOS.copy()
    if 'results' not in st.session_state:
        st.session_state.results = None

_init_state()

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: BUILD SOIL LAYERS LIST (kip-in) FROM DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
_LABEL_TO_KEY = {v: k for k, v in MODEL_LABELS.items()}

def _df_to_layers(df, b_in):
    """Convert layers DataFrame (imperial) → list of kip-in layer dicts."""
    layers = []
    for _, row in df.iterrows():
        model_key = _LABEL_TO_KEY.get(row['Model'], 'matlock')
        cu_ksi    = float(row['su (ksf)']) * KSF2KSI
        gamma     = float(row["γ' (pcf)"]) * PCF2KIPIN3
        ks        = float(row['ks (pci)']) * PCI2KIPIN3

        params = {
            'cu_ksi':        cu_ksi,
            'ca_ksi':        cu_ksi,        # ca = cu (uniform shear strength)
            'eps50':         float(row['ε50']),
            'gamma_kip_in3': gamma,
            'J':             float(row['J']),
            'ks_kip_in3':    ks,
            'loading':       str(row['Loading']).lower(),
            'b_in':          b_in,
        }
        layers.append({
            'z_top_in': float(row['Top (ft)']) * FT2IN,
            'z_bot_in': float(row['Bot (ft)']) * FT2IN,
            'model':    model_key,
            'params':   params,
        })
    # Sort by depth and ensure tip layer extends to infinity
    layers.sort(key=lambda l: l['z_top_in'])
    layers[-1]['z_bot_in'] = 1e12
    return layers


def _df_to_load_cases(loads_df, combos_df=None):
    """
    Convert load case + combination dataframes → list of kip-in load dicts.
    Returns (individual_cases, combo_cases).
    """
    lc_list = []
    for _, row in loads_df.iterrows():
        lc_list.append({
            'name':      str(row['Case']),
            'H_kip':     float(row['H (kip)']),
            'M_kip_in':  float(row['M (kip·ft)']) * KIPFT2KIPIN,
        })

    combo_list = []
    if combos_df is not None and len(combos_df) > 0:
        for _, row in combos_df.iterrows():
            H_c = 0.0
            M_c = 0.0
            for lc in lc_list:
                f = float(row.get(lc['name'], 0.0))
                H_c += f * lc['H_kip']
                M_c += f * lc['M_kip_in']
            combo_list.append({
                'name':     str(row['Combo']),
                'H_kip':    H_c,
                'M_kip_in': M_c,
            })
    return lc_list, combo_list


# ══════════════════════════════════════════════════════════════════════════════
# SECTION EI CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
def _compute_EI(section_type, D_in, E_ksi, t_in=None, I_in4=None):
    """
    Compute EI [kip·in²] from section type.
    section_type: 'Solid Circle' | 'Steel Pipe' | 'User-Defined I'
    """
    if section_type == 'Solid Circle':
        I = np.pi * D_in ** 4 / 64.0
    elif section_type == 'Steel Pipe':
        D_in_inner = D_in - 2.0 * (t_in or 0.0)
        I = np.pi * (D_in ** 4 - D_in_inner ** 4) / 64.0
    else:  # User-Defined I
        I = float(I_in4 or 1.0)
    return E_ksi * I   # kip/in² × in⁴ = kip·in²


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="main-header">🏗️ LateralPile</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Clay Analysis · Imperial Units</div>',
                unsafe_allow_html=True)
    st.divider()

    # ── A. PILE PROPERTIES ─────────────────────────────────────────────────
    st.subheader("A. Pile Properties")

    L_ft = st.number_input("Pile length  (ft)", min_value=5.0, max_value=500.0,
                            value=40.0, step=1.0)
    D_in = st.number_input("Pile diameter  (in)", min_value=1.0, max_value=240.0,
                            value=18.0, step=0.5)
    E_ksi = st.number_input("Modulus E  (ksi)", min_value=1.0, max_value=50000.0,
                             value=3600.0, step=100.0,
                             help="Concrete ≈ 3600 ksi | Steel ≈ 29000 ksi")

    section_type = st.selectbox("Section type",
                                 ['Solid Circle', 'Steel Pipe', 'User-Defined I'])
    t_in   = None
    I_in4  = None
    if section_type == 'Steel Pipe':
        t_in = st.number_input("Wall thickness  (in)", min_value=0.1, max_value=D_in/2,
                                value=min(0.5, D_in/4), step=0.05)
    elif section_type == 'User-Defined I':
        I_in4 = st.number_input("Moment of inertia I  (in⁴)",
                                 min_value=0.01, value=1000.0, step=10.0)

    EI_auto = _compute_EI(section_type, D_in, E_ksi, t_in, I_in4)

    override_EI = st.checkbox("Override EI manually")
    if override_EI:
        EI_kip_ft2 = st.number_input("EI override  (kip·ft²)",
                                      min_value=1.0, value=float(EI_auto / KIPFT22KIPIN2),
                                      step=1000.0)
        EI_kip_in2 = EI_kip_ft2 * KIPFT22KIPIN2
    else:
        EI_kip_in2 = EI_auto
        EI_kip_ft2 = EI_kip_in2 / KIPFT22KIPIN2

    st.info(f"**EI = {EI_kip_ft2:,.0f} kip·ft²** ({EI_kip_in2:,.0f} kip·in²)")

    st.divider()

    # ── B. BOUNDARY CONDITIONS ────────────────────────────────────────────
    st.subheader("B. Boundary Conditions")

    head_cond = st.radio("Pile head",
                          ["Free head  (H + M applied)",
                           "Fixed head (rotation restrained)"],
                          index=0)
    head_fixed = (head_cond.startswith("Fixed"))

    tip_cond = st.selectbox("Pile tip",
                              ['free', 'pinned', 'fixed'],
                              help=("free = no restraint  |  "
                                    "pinned = y=0  |  fixed = y=0, θ=0"))

    st.divider()

    # ── SOLVER SETTINGS ───────────────────────────────────────────────────
    with st.expander("⚙️ Solver Settings", expanded=False):
        n_elements = st.slider("Number of elements", 20, 120, 60, step=5)
        max_iter   = st.slider("Max iterations", 50, 300, 150, step=25)
        tol_in     = st.select_slider("Convergence tol (in)",
                                       options=[1e-4, 1e-5, 1e-6, 1e-7],
                                       value=1e-6,
                                       format_func=lambda x: f"{x:.0e}")
        relax      = st.slider("Under-relaxation factor", 0.3, 1.0, 1.0, step=0.05,
                                help="1.0 = no relaxation; reduce for convergence issues")

    # ── RUN BUTTON ────────────────────────────────────────────────────────
    st.divider()
    run_clicked = st.button("▶  Run Analysis", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL — TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">Lateral Pile Analysis — Clay</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Beam-column on nonlinear Winkler foundation '
            '· p-y method · Imperial units</div>', unsafe_allow_html=True)

tab_soil, tab_loads, tab_combos, tab_results, tab_py = st.tabs([
    "🌍 Soil Profile",
    "⚡ Load Cases",
    "🔗 Load Combinations",
    "📊 Results",
    "📈 p-y Curves",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SOIL PROFILE
# ══════════════════════════════════════════════════════════════════════════════
with tab_soil:
    st.subheader("Soil Layer Profile")
    st.caption(
        "Define layers from ground surface down to pile tip.  "
        "Each layer uses one clay p-y model.  "
        "**su** = undrained shear strength.  "
        "**ks** only required for Reese Stiff Clay w/ Water (Table 5.3 in RSPile manual).")

    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("➕ Add layer"):
            new_row = pd.DataFrame([{
                'Layer':    len(st.session_state.layers_df) + 1,
                'Top (ft)': float(st.session_state.layers_df['Bot (ft)'].iloc[-1]),
                'Bot (ft)': float(st.session_state.layers_df['Bot (ft)'].iloc[-1]) + 10.0,
                'Model':    MODEL_LABELS['matlock'],
                'su (ksf)': 1.0,
                'ε50':      0.020,
                "γ' (pcf)": 60.0,
                'J':        0.5,
                'ks (pci)': 500.0,
                'Loading':  'Static',
            }])
            st.session_state.layers_df = pd.concat(
                [st.session_state.layers_df, new_row], ignore_index=True)

        if st.button("🗑️ Remove last"):
            if len(st.session_state.layers_df) > 1:
                st.session_state.layers_df = st.session_state.layers_df.iloc[:-1]

    layers_edited = st.data_editor(
        st.session_state.layers_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            'Layer':    st.column_config.NumberColumn('Layer #', disabled=True),
            'Top (ft)': st.column_config.NumberColumn('Top (ft)', format="%.1f"),
            'Bot (ft)': st.column_config.NumberColumn('Bot (ft)', format="%.1f"),
            'Model':    st.column_config.SelectboxColumn(
                'p-y Model',
                options=list(MODEL_LABELS.values()),
                required=True),
            'su (ksf)': st.column_config.NumberColumn('su (ksf)', format="%.3f",
                                                        help="Undrained shear strength"),
            'ε50':      st.column_config.NumberColumn('ε50', format="%.4f",
                                                        help="Strain at 50% deviatoric stress"),
            "γ' (pcf)": st.column_config.NumberColumn("γ' (pcf)", format="%.1f",
                                                         help="Effective unit weight"),
            'J':        st.column_config.NumberColumn('J', format="%.2f",
                                                        help="Matlock factor (default 0.5)"),
            'ks (pci)': st.column_config.NumberColumn('ks (pci)', format="%.0f",
                                                        help="Initial stiffness (Reese only)"),
            'Loading':  st.column_config.SelectboxColumn('Loading',
                                                           options=['Static', 'Cyclic']),
        },
        key="layers_editor",
    )
    st.session_state.layers_df = layers_edited

    # ── Reference table ───────────────────────────────────────────────────
    with st.expander("📋 Typical parameter ranges"):
        st.markdown("""
| Soil | Consistency | su (ksf) | ε50 | γ' (pcf) |
|------|-------------|---------|-----|----------|
| Soft clay  | Very Soft–Soft  | 0.25–1.0 | 0.020 | 55–62 |
| Med. clay  | Medium          | 1.0–2.0  | 0.010 | 57–63 |
| Stiff clay | Stiff–Very Stiff| 2.0–4.0  | 0.005–0.007 | 60–65 |
| Hard clay  | Hard            | >4.0     | 0.004–0.005 | 62–68 |

**ks (pci)** for Reese stiff clay — typical values (RSPile Table 5.3, converted):
| su range | Static ks (pci) | Cyclic ks (pci) |
|----------|----------------|----------------|
| 1.0–2.1 ksf (50–100 kPa) | 497 | 202 |
| 2.1–4.2 ksf (100–200 kPa)| 994 | 405 |
| 4.2–8.3 ksf (200–400 kPa)|1989 |1989 |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LOAD CASES
# ══════════════════════════════════════════════════════════════════════════════
with tab_loads:
    st.subheader("Load Cases")
    st.caption(
        "Define individual load cases.  "
        "**H** = lateral shear at pile head.  "
        "**M** = applied moment at pile head (+ = same sense as load-induced moment).")

    col_la, col_lb = st.columns([3, 1])
    with col_lb:
        if st.button("➕ Add load case"):
            n = len(st.session_state.loads_df) + 1
            new_lc = pd.DataFrame([{
                'Case':       f'LC-{n}',
                'H (kip)':    50.0,
                'M (kip·ft)': 0.0,
            }])
            st.session_state.loads_df = pd.concat(
                [st.session_state.loads_df, new_lc], ignore_index=True)
        if st.button("🗑️ Remove last ", key='rm_lc'):
            if len(st.session_state.loads_df) > 1:
                st.session_state.loads_df = st.session_state.loads_df.iloc[:-1]

    loads_edited = st.data_editor(
        st.session_state.loads_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            'Case':       st.column_config.TextColumn('Case Name'),
            'H (kip)':    st.column_config.NumberColumn('H (kip)', format="%.2f"),
            'M (kip·ft)': st.column_config.NumberColumn('M (kip·ft)', format="%.2f"),
        },
        key="loads_editor",
    )
    st.session_state.loads_df = loads_edited


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LOAD COMBINATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_combos:
    st.subheader("Load Combinations")
    st.caption(
        "Each combination is analyzed as a single nonlinear event: "
        "H_combo = Σ factor × H_LC.  Superposition of *loads* (not results).")

    lc_names = list(st.session_state.loads_df['Case'].astype(str))

    # Rebuild combos_df columns if load case names changed
    existing_cols  = list(st.session_state.combos_df.columns)
    expected_cols  = ['Combo'] + lc_names
    if set(existing_cols) != set(expected_cols):
        new_combos = pd.DataFrame(columns=expected_cols)
        new_combos['Combo'] = ['COMB-1']
        for cn in lc_names:
            new_combos[cn] = 1.0
        st.session_state.combos_df = new_combos

    col_ca, col_cb = st.columns([3, 1])
    with col_cb:
        if st.button("➕ Add combination"):
            n = len(st.session_state.combos_df) + 1
            new_c = {cn: 0.0 for cn in lc_names}
            new_c['Combo'] = f'COMB-{n}'
            st.session_state.combos_df = pd.concat(
                [st.session_state.combos_df,
                 pd.DataFrame([new_c])], ignore_index=True)
        if st.button("🗑️ Remove last  ", key='rm_combo'):
            if len(st.session_state.combos_df) > 1:
                st.session_state.combos_df = st.session_state.combos_df.iloc[:-1]

    combo_col_cfg = {'Combo': st.column_config.TextColumn('Combo Name')}
    for cn in lc_names:
        combo_col_cfg[cn] = st.column_config.NumberColumn(f'× {cn}', format="%.3f")

    combos_edited = st.data_editor(
        st.session_state.combos_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config=combo_col_cfg,
        key="combos_editor",
    )
    st.session_state.combos_df = combos_edited

    # Preview table
    if len(st.session_state.combos_df) > 0:
        lc_list_prev, combo_prev = _df_to_load_cases(
            st.session_state.loads_df, st.session_state.combos_df)
        preview_rows = []
        for c in combo_prev:
            preview_rows.append({
                'Combo': c['name'],
                'H (kip)': f"{c['H_kip']:.2f}",
                'M (kip·ft)': f"{c['M_kip_in']/KIPFT2KIPIN:.2f}",
            })
        if preview_rows:
            st.caption("Combined loads:")
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if run_clicked:
    with st.spinner("Running nonlinear p-y analysis …"):
        try:
            # Build internal data structures
            b_in       = float(D_in)
            layers_kip = _df_to_layers(st.session_state.layers_df, b_in)
            lc_list, combo_list = _df_to_load_cases(
                st.session_state.loads_df, st.session_state.combos_df)

            all_cases = lc_list + combo_list

            if not all_cases:
                st.error("No load cases or combinations defined.")
            else:
                pile_params = {
                    'L_in':       float(L_ft) * FT2IN,
                    'EI_kip_in2': EI_kip_in2,
                    'n_elements': n_elements,
                    'head_fixed': head_fixed,
                    'tip_cond':   tip_cond,
                }
                results = run_analysis(
                    pile_params, layers_kip, all_cases,
                    max_iter=max_iter, tol=tol_in, relax=relax)
                st.session_state.results = results
                st.session_state.pile_params_run = pile_params
                st.session_state.layers_run = layers_kip
                st.success(f"✅  Analysis complete — {len(results)} case(s) solved.")
        except Exception as e:
            st.error(f"Analysis error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# HELPER PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
_COLORS = ['#1a6ecf', '#e84a2e', '#2e9c45', '#f0a500', '#8b3dab',
           '#0d9488', '#c026d3', '#dc2626', '#2563eb', '#16a34a']


def _depth_plot(axes, z_in_arr, data_arr, xlabel, color, label='', lw=2.0, ls='-'):
    """Depth-versus-value plot (z downward)."""
    ax = axes
    z_ft = z_in_arr / FT2IN
    ax.plot(data_arr, z_ft, color=color, lw=lw, ls=ls, label=label)
    ax.fill_betweenx(z_ft, data_arr, 0, color=color, alpha=0.10)
    ax.axvline(0, color='#888', lw=0.8, ls='--')
    ax.invert_yaxis()
    ax.set_ylim(z_ft[-1], z_ft[0])
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(True, alpha=0.25, ls=':')
    if label:
        ax.legend(fontsize=7, loc='lower right')


def _make_results_figure(results_list, selected_indices, pile_params):
    """
    Build 5-panel depth plot (deflection, rotation, moment, shear, p/p_ult)
    for the selected result indices, overlaid on one figure.
    """
    fig, axes = plt.subplots(1, 5, figsize=(17, 8), sharey=False)
    titles  = ['Deflection\ny  (in)', 'Rotation\nθ  (°)',
               'Moment\nM  (kip·ft)', 'Shear\nV  (kip)', 'Mobilisation\np/pult  (-)']

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.25, ls=':')
        ax.axvline(0, color='#888', lw=0.7, ls='--')
        if i == 0:
            ax.set_ylabel('Depth  z  (ft)', fontsize=9)

    for idx_c, idx in enumerate(selected_indices):
        res   = results_list[idx]
        col   = _COLORS[idx_c % len(_COLORS)]
        lbl   = res['name']
        z_ft  = res['z_in'] / FT2IN

        datasets = [
            res['y_in'],
            np.degrees(res['theta_rad']),
            res['M_kip_in'] / KIPFT2KIPIN,    # kip·in → kip·ft
            res['V_kip'],
            res['p_over_pult'],
        ]

        for i, (ax, data) in enumerate(zip(axes, datasets)):
            ax.plot(data, z_ft, color=col, lw=2.0, label=lbl if i == 0 else '')
            ax.fill_betweenx(z_ft, data, 0, color=col, alpha=0.08)
            ax.invert_yaxis()
            ax.set_ylim(z_ft[-1], z_ft[0])
            if i > 0:
                ax.set_yticklabels([])

    axes[0].legend(fontsize=7.5, loc='lower right')
    # p/pult: add reference lines
    axes[4].axvline(1.0, color='red', lw=1.2, ls='--', alpha=0.7)
    axes[4].set_xlim(left=0)

    plt.tight_layout(pad=1.2)
    return fig


def _make_py_figure(layers_kip, z_depths_ft, b_in):
    """p-y curves at selected depths."""
    z_depths_in = [z * FT2IN for z in z_depths_ft]
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis

    for j, z_in in enumerate(z_depths_in):
        lyr = None
        for l in layers_kip:
            if l['z_top_in'] <= z_in:
                lyr = l
        if lyr is None:
            continue
        y_arr, p_arr = get_py_curve(z_in, lyr['params'], lyr['model'])
        col = cmap(j / max(len(z_depths_in) - 1, 1))
        ax.plot(y_arr, p_arr * FT2IN,   # kip/in → kip/ft
                color=col, lw=2.0, label=f'z = {z_in/FT2IN:.1f} ft  [{MODEL_LABELS[lyr["model"]][:25]}]')

        eps50 = float(lyr['params']['eps50'])
        if lyr['model'] == 'matlock':
            y50 = 2.5 * eps50 * b_in
        else:
            y50 = eps50 * b_in
        ax.axvline(y50, color=col, lw=0.7, ls=':', alpha=0.5)

    ax.set_xlabel('Deflection  y  (in)', fontsize=10)
    ax.set_ylabel('Soil resistance  p  (kip/ft)', fontsize=10)
    ax.set_title('p-y Curves at Selected Depths', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.25, ls=':')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return fig


def _make_convergence_figure(results_list, selected_indices):
    """Convergence history for selected cases."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx_c, idx in enumerate(selected_indices):
        res  = results_list[idx]
        col  = _COLORS[idx_c % len(_COLORS)]
        hist = res['conv_hist']
        ax.semilogy(range(1, len(hist)+1), hist,
                    'o-', color=col, ms=3.5, lw=1.8, label=res['name'])
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('max |Δy|  (in)', fontsize=10)
    ax.set_title('Solver Convergence', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.25, ls=':')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if st.session_state.results is None:
        st.info("▶  Click **Run Analysis** in the sidebar to compute results.")
    else:
        results = st.session_state.results

        # ── Case selector ─────────────────────────────────────────────────
        case_names = [r['name'] for r in results]
        selected = st.multiselect("Select cases to plot",
                                   case_names, default=case_names[:min(3, len(case_names))])
        selected_idx = [case_names.index(n) for n in selected if n in case_names]

        if not selected_idx:
            st.warning("Select at least one case to display.")
        else:
            # ── Summary table ─────────────────────────────────────────────
            st.subheader("Summary")
            rows = []
            for idx in selected_idx:
                r = results[idx]
                conv_str = "✅ Yes" if r['converged'] else f"⚠️ No (>{r['n_iter']} iter)"
                rows.append({
                    'Case':         r['name'],
                    'H (kip)':      f"{r['H_applied_kip']:.2f}",
                    'M (kip·ft)':   f"{r['M_applied_kip_in']/KIPFT2KIPIN:.2f}",
                    'y(0) (in)':    f"{r['y_head_in']:.4f}",
                    'M_max (kip·ft)': f"{r['M_max_kip_in']/KIPFT2KIPIN:.2f}",
                    'z(Mmax) (ft)': f"{r['z_Mmax_in']/FT2IN:.2f}",
                    'V(0) (kip)':   f"{r['V_head_kip']:.3f}",
                    'V(L) (kip)':   f"{r['V_tip_kip']:.4f}",
                    'y(L) (in)':    f"{r['y_tip_in']:.5f}",
                    'Iter':         r['n_iter'],
                    'Converged':    conv_str,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Convergence notes ─────────────────────────────────────────
            for idx in selected_idx:
                r = results[idx]
                if not r['converged']:
                    st.markdown(
                        f'<div class="warn-card">⚠️ <b>{r["name"]}</b>: did not converge '
                        f'after {r["n_iter"]} iterations. '
                        f'Try reducing relaxation factor or increasing max iterations.</div>',
                        unsafe_allow_html=True)

            st.divider()

            # ── 5-panel depth plots ───────────────────────────────────────
            st.subheader("Depth Profiles")
            fig_main = _make_results_figure(
                results, selected_idx,
                st.session_state.pile_params_run)
            st.pyplot(fig_main, use_container_width=True)
            plt.close(fig_main)

            st.divider()

            # ── Convergence plot ──────────────────────────────────────────
            st.subheader("Convergence History")
            fig_conv = _make_convergence_figure(results, selected_idx)
            st.pyplot(fig_conv, use_container_width=True)
            plt.close(fig_conv)

            st.divider()

            # ── Tabular data download ─────────────────────────────────────
            with st.expander("📥 Download results as CSV"):
                for idx in selected_idx:
                    r = results[idx]
                    df_out = pd.DataFrame({
                        'Depth (ft)':        r['z_in'] / FT2IN,
                        'Deflection y (in)': r['y_in'],
                        'Rotation theta (deg)': np.degrees(r['theta_rad']),
                        'Moment M (kip-ft)': r['M_kip_in'] / KIPFT2KIPIN,
                        'Shear V (kip)':     r['V_kip'],
                        'Soil rxn p (kip/ft)': r['p_kip_in'] * FT2IN,
                        'p_ult (kip/ft)':    r['pult_kip_in'] * FT2IN,
                        'p/p_ult':           r['p_over_pult'],
                    })
                    csv = df_out.to_csv(index=False)
                    st.download_button(
                        label=f"Download {r['name']}.csv",
                        data=csv,
                        file_name=f"{r['name'].replace(' ','_')}.csv",
                        mime='text/csv',
                        key=f"dl_{idx}",
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — p-y CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab_py:
    st.subheader("p-y Curves at Selected Depths")
    st.caption(
        "Preview the p-y curves that the solver will use.  "
        "Dashed vertical lines mark y50.  "
        "Depths are evaluated against the current soil profile.")

    max_depth_ft = float(L_ft)
    default_depths = [0.0, max_depth_ft*0.1, max_depth_ft*0.25,
                      max_depth_ft*0.5, max_depth_ft*0.75]
    default_depths = [round(d, 1) for d in default_depths]

    depths_input = st.text_input(
        "Depths to plot (ft, comma-separated)",
        value=", ".join(str(d) for d in default_depths))

    try:
        z_plot_ft = [float(x.strip()) for x in depths_input.split(',') if x.strip()]
        z_plot_ft = [z for z in z_plot_ft if 0.0 <= z <= max_depth_ft]
    except ValueError:
        z_plot_ft = default_depths

    if z_plot_ft:
        try:
            layers_preview = _df_to_layers(st.session_state.layers_df, float(D_in))
            fig_py = _make_py_figure(layers_preview, z_plot_ft, float(D_in))
            st.pyplot(fig_py, use_container_width=True)
            plt.close(fig_py)
        except Exception as e:
            st.error(f"Could not generate p-y curves: {e}")

    # ── Quick p-y table at a single depth ────────────────────────────────
    with st.expander("🔍 Inspect p-y values at one depth"):
        z_insp_ft = st.number_input("Depth (ft)", 0.0, float(L_ft), 5.0, step=0.5)
        z_insp_in = z_insp_ft * FT2IN
        try:
            layers_insp = _df_to_layers(st.session_state.layers_df, float(D_in))
            from fem_solver import _find_layer
            lyr_insp = _find_layer(z_insp_in, layers_insp)
            y_arr, p_arr = get_py_curve(z_insp_in, lyr_insp['params'], lyr_insp['model'],
                                         n_pts=20)
            df_insp = pd.DataFrame({
                'y (in)':      np.round(y_arr, 5),
                'p (kip/ft)':  np.round(p_arr * FT2IN, 5),
                'k_sec (kip/ft²)': [
                    round(get_soil_response(yv, z_insp_in, lyr_insp['params'],
                                            lyr_insp['model'])[1] * FT2IN, 3)
                    for yv in y_arr
                ],
            })
            st.caption(f"Model: {MODEL_LABELS[lyr_insp['model']]}  "
                       f"| su = {lyr_insp['params']['cu_ksi']*144:.2f} ksf  "
                       f"| ε50 = {lyr_insp['params']['eps50']:.4f}")
            st.dataframe(df_insp, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"{e}")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "**LateralPile — Clay v1.0**  ·  "
    "Theory: RSPile Laterally Loaded Piles Theory Manual (Rocscience, 2022)  ·  "
    "Matlock (1970)  ·  Reese et al. (1975)  ·  Welch & Reese (1972)  ·  "
    "Hermitian beam FEM  ·  Gauss-integrated Winkler stiffness  ·  Secant iteration")
