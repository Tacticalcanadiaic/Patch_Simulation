#!/usr/bin/env python3
"""
2.4 GHz inset-fed microstrip patch antenna - openEMS (FDTD) version.

Python/openEMS port of patch_2p4GHz_prototype.m: same board, substrate,
geometry, band targets and result checks, so the two solvers can be compared.

Differences from the MATLAB model:
  - Feed is a 50-ohm microstrip port at the board edge (closer to a real
    edge-launch SMA) instead of MATLAB's via probe 1.5 mm from the edge.
    |S11| and resonance compare directly; Zin is referenced to the port's
    measurement plane, so its phase will differ slightly from MATLAB.
  - FDTD gives the whole 2.0-3.0 GHz response from a single run, so there is
    no coarse/fine sweep.
  - Radiation pattern is shown as E- and H-plane cuts rather than a 3D plot.

Requires: openEMS and CSXCAD with Python bindings, numpy, matplotlib.
"""

import os
import time
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

# ================================ SETTINGS ================================
BAND_LOW  = 2.400e9                    # Wi-Fi / BLE band
BAND_HIGH = 2.4835e9
F_CENTER  = (BAND_LOW + BAND_HIGH) / 2
Z0        = 50.0
S11_LIMIT = -10.0                      # dB

F_START, F_STOP = 2.0e9, 3.0e9         # analysed range (one FDTD run)
N_FREQ = 1001

MESH_DIV        = 20      # cells per substrate wavelength at F_STOP
SUBSTRATE_CELLS = 4       # cells through the substrate thickness
AIR_MARGIN      = 50.0    # mm of air around the board
END_CRITERIA    = 1e-4    # stop at -40 dB energy decay (1e-5 for final check)
MAX_TIMESTEPS   = 200000

SHOW_GEOMETRY       = False   # open the model in AppCSXCAD before running
RUN_PATTERN         = True
RUN_MESH_STUDY      = True # re-runs the full simulation per mesh setting
MESH_STUDY_DIVS     = [15, 20, 30]
RUN_TOLERANCE_STUDY = False   # re-runs with epsR +/- tolerance
EPS_R_TOL           = 0.2

SIM_ROOT = os.path.join(tempfile.gettempdir(), "patch_2p4GHz")

# ===================== BOARD, SUBSTRATE, GEOMETRY (mm) =====================
GEOM = {
    "board_x":     50.0,
    "board_y":     50.0,
    "board_thk":   1.6,
    "patch_x":     37.303,   # patch width
    "patch_y":     32.100,   # patch length (resonant, along feed)
    "inset_depth": 8.000,
    "slot_gap":    0.750,
    "feed_width":  3.000,
}
EPS_R = 4.38
TAN_D = 0.02                 # generic FR-4; use laminate datasheet value

UNIT = 1e-3                  # drawing unit = mm


# ============================== MODEL BUILD ===============================
def layout(g):
    """Patch/feed positions, matching buildInsetPatch in the MATLAB script."""
    feed_margin = 3.0
    avail = g["board_y"] - g["patch_y"] - feed_margin
    if avail <= 0:
        raise ValueError("Patch too long for board.")
    if g["inset_depth"] >= g["patch_y"]:
        raise ValueError("Inset deeper than patch.")

    L = {}
    L["y_edge"]    = -g["board_y"] / 2                      # feed start
    L["patch_bot"] = L["y_edge"] + feed_margin + avail / 2
    L["patch_top"] = L["patch_bot"] + g["patch_y"]
    L["inset_top"] = L["patch_bot"] + g["inset_depth"]      # feed point
    L["slot_in"]   = g["feed_width"] / 2
    L["slot_out"]  = g["feed_width"] / 2 + g["slot_gap"]
    L["port_len"]  = 0.75 * (L["patch_bot"] - L["y_edge"])  # MSL port section
    return L


def thirds(edge, metal_side, res):
    """1/3-2/3 mesh rule at a metal edge (metal_side = +1 or -1)."""
    return [edge + metal_side * res / 3, edge - metal_side * 2 * res / 3]


def build_and_run(g, eps_r, tan_d, mesh_div, sim_path, want_nf2ff=True, show=False):
    L = layout(g)
    h = g["board_thk"]
    bx, by = g["board_x"], g["board_y"]
    px, fw = g["patch_x"], g["feed_width"]

    # FDTD engine and excitation
    fdtd = openEMS(NrTS=MAX_TIMESTEPS, EndCriteria=END_CRITERIA)
    f0 = (F_START + F_STOP) / 2
    fc = 1.2 * (F_STOP - F_START) / 2
    fdtd.SetGaussExcite(f0, fc)
    fdtd.SetBoundaryCond(["MUR"] * 6)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    mesh_res = C0 / F_STOP / UNIT / np.sqrt(eps_r) / mesh_div
    edge_res = mesh_res / 4

    # Materials
    pec = csx.AddMetal("PEC")
    kappa = tan_d * 2 * np.pi * F_CENTER * EPS0 * eps_r
    fr4 = csx.AddMaterial("FR4", epsilon=eps_r, kappa=kappa)

    fr4.AddBox([-bx/2, -by/2, 0], [bx/2, by/2, h], priority=0)
    pec.AddBox([-bx/2, -by/2, 0], [bx/2, by/2, 0], priority=10)          # ground

    # Patch = upper section + two lower sections beside the inset slots
    pec.AddBox([-px/2, L["inset_top"], h], [px/2, L["patch_top"], h], priority=10)
    pec.AddBox([-px/2, L["patch_bot"], h], [-L["slot_out"], L["inset_top"], h], priority=10)
    pec.AddBox([L["slot_out"], L["patch_bot"], h], [px/2, L["inset_top"], h], priority=10)
    # Feed line from end of port section to the inset point
    pec.AddBox([-fw/2, L["y_edge"] + L["port_len"], h], [fw/2, L["inset_top"], h], priority=10)

    # Mesh lines
    slot_mid = (L["slot_in"] + L["slot_out"]) / 2
    x = [-bx/2 - AIR_MARGIN, bx/2 + AIR_MARGIN, -bx/2, bx/2,
         -L["slot_out"], -slot_mid, -L["slot_in"], L["slot_in"], slot_mid, L["slot_out"]]
    x += thirds(-px/2, +1, edge_res) + thirds(px/2, -1, edge_res)

    y = [L["y_edge"] - AIR_MARGIN, by/2 + AIR_MARGIN, by/2, L["inset_top"]]
    y += list(np.linspace(L["y_edge"], L["y_edge"] + L["port_len"], 5))
    y += thirds(L["patch_bot"], +1, edge_res) + thirds(L["patch_top"], -1, edge_res)

    z = list(np.linspace(0, h, SUBSTRATE_CELLS + 1)) + [-AIR_MARGIN, h + AIR_MARGIN]

    mesh.AddLine("x", x)
    mesh.AddLine("y", y)
    mesh.AddLine("z", z)
    mesh.SmoothMeshLines("all", mesh_res, 1.4)

    # 50-ohm microstrip port at the board edge (mesh must exist first)
    port = fdtd.AddMSLPort(1, pec,
                           [-fw/2, L["y_edge"], h],
                           [ fw/2, L["y_edge"] + L["port_len"], 0],
                           "y", "z", excite=-1, Feed_R=Z0, priority=10)

    nf2ff = fdtd.CreateNF2FFBox() if want_nf2ff else None

    n = [len(mesh.GetLines(d)) for d in ("x", "y", "z")]

    if show:
        os.makedirs(sim_path, exist_ok=True)
        xml = os.path.join(sim_path, "geometry.xml")
        csx.Write2XML(xml)
        try:
            from CSXCAD import AppCSXCAD_BIN
            os.system(f'{AppCSXCAD_BIN} "{xml}"')
        except ImportError:
            print(f"AppCSXCAD not found - open {xml} manually.")

    os.makedirs(sim_path, exist_ok=True)   # openEMS won't create parent folders
    t0 = time.time()
    fdtd.Run(sim_path, cleanup=True)
    # Keep the solver/geometry objects alive for post-processing; otherwise
    # Python deletes them when this function returns and CalcPort fails.
    port.keep_alive = (fdtd, csx)
    return port, nf2ff, n, time.time() - t0


# ============================== POST-PROCESS ==============================
def analyze(port, sim_path):
    f = np.unique(np.concatenate([np.linspace(F_START, F_STOP, N_FREQ),
                                  [BAND_LOW, F_CENTER, BAND_HIGH]]))
    port.CalcPort(sim_path, f, ref_impedance=Z0)
    gamma = port.uf_ref / port.uf_inc
    mag = np.minimum(np.abs(gamma), 0.999999)
    return {
        "f": f,
        "gamma": gamma,
        "s11": 20 * np.log10(np.abs(gamma)),
        "vswr": (1 + mag) / (1 - mag),
        "zin": port.uf_tot / port.if_tot,
        "p_inc": port.P_inc,
        "p_acc": port.P_acc,
    }


def matched_band(f, s11, i_dip, limit):
    """Contiguous region around the dip where S11 <= limit."""
    if s11[i_dip] > limit:
        return np.nan, np.nan
    lo = i_dip
    while lo > 0 and s11[lo - 1] <= limit:
        lo -= 1
    hi = i_dip
    while hi < len(f) - 1 and s11[hi + 1] <= limit:
        hi += 1
    return f[lo], f[hi]


def metrics(r):
    f, s11 = r["f"], r["s11"]
    i_dip = int(np.argmin(s11))
    in_band = (f >= BAND_LOW) & (f <= BAND_HIGH)
    m = {
        "f_dip": f[i_dip],
        "s11_min": s11[i_dip],
        "at_edge": i_dip in (0, len(f) - 1),
        "worst_s11": s11[in_band].max(),
        "worst_vswr": r["vswr"][in_band].max(),
        "i_low": int(np.argmin(np.abs(f - BAND_LOW))),
        "i_ctr": int(np.argmin(np.abs(f - F_CENTER))),
        "i_high": int(np.argmin(np.abs(f - BAND_HIGH))),
    }
    m["f_lo"], m["f_hi"] = matched_band(f, s11, i_dip, S11_LIMIT)
    return m


def smith_axes(ax):
    """Draw a basic Smith chart grid."""
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="0.3", lw=1)
    ax.plot([-1, 1], [0, 0], color="0.6", lw=0.6)
    for rv in (0.2, 0.5, 1, 2, 5):
        c, rad = rv / (1 + rv), 1 / (1 + rv)
        ax.plot(c + rad * np.cos(t), rad * np.sin(t), color="0.75", lw=0.6)
    rr = np.concatenate([np.linspace(0, 10, 400), np.logspace(1, 3, 200)])
    for xv in (0.2, 0.5, 1, 2, 5):
        for s in (1, -1):
            z = rr + 1j * s * xv
            gm = (z - 1) / (z + 1)
            ax.plot(gm.real, gm.imag, color="0.75", lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")


# ================================== MAIN ==================================
def main():
    g = GEOM
    t_all = time.time()

    print("2.4 GHz PATCH - openEMS SIMULATION")
    print(f"Band           : {BAND_LOW/1e9:.4f} - {BAND_HIGH/1e9:.4f} GHz "
          f"(centre {F_CENTER/1e9:.4f} GHz)")
    print(f"Substrate      : epsR {EPS_R:.2f}, tan d {TAN_D:.3f}, {g['board_thk']:.2f} mm")
    print(f"Patch W x L    : {g['patch_x']:.3f} x {g['patch_y']:.3f} mm")
    print(f"Inset/gap/feed : {g['inset_depth']:.3f} / {g['slot_gap']:.3f} / "
          f"{g['feed_width']:.3f} mm")

    sim_path = os.path.join(SIM_ROOT, "main")
    port, nf2ff, n, t_run = build_and_run(g, EPS_R, TAN_D, MESH_DIV, sim_path,
                                          want_nf2ff=RUN_PATTERN, show=SHOW_GEOMETRY)
    print(f"Mesh           : {n[0]} x {n[1]} x {n[2]} = {np.prod(n)/1e3:.0f}k cells")
    print(f"FDTD run time  : {t_run:.0f} s")

    r = analyze(port, sim_path)
    m = metrics(r)
    f, s11, vswr, zin = r["f"], r["s11"], r["vswr"], r["zin"]

    # ---------------- results ----------------
    print("\nRESULTS")
    edge_note = "  (AT RANGE EDGE - not a true dip)" if m["at_edge"] else ""
    print(f"Resonance (S11 min) : {m['f_dip']/1e9:.4f} GHz{edge_note}")
    print(f"Offset from centre  : {(m['f_dip'] - F_CENTER)/1e6:+.1f} MHz")
    print(f"S11 at resonance    : {m['s11_min']:.2f} dB")
    if np.isnan(m["f_lo"]):
        print("-10 dB bandwidth    : none")
    else:
        bw = m["f_hi"] - m["f_lo"]
        print(f"-10 dB bandwidth    : {m['f_lo']/1e9:.4f} - {m['f_hi']/1e9:.4f} GHz "
              f"({bw/1e6:.1f} MHz, {100*bw/m['f_dip']:.2f}%)")

    print(f"\n{'Freq':<10} {'S11 (dB)':>9} {'VSWR':>8} {'Zin (ohm)':>22}")
    for k in (m["i_low"], m["i_ctr"], m["i_high"]):
        print(f"{f[k]/1e9:.4f} GHz {s11[k]:9.2f} {vswr[k]:8.2f} "
              f"{zin[k].real:10.2f} {zin[k].imag:+9.2f}j")
    print(f"\nWorst in-band S11   : {m['worst_s11']:.2f} dB")
    print(f"Worst in-band VSWR  : {m['worst_vswr']:.2f}")

    if abs(m["f_dip"] - F_CENTER) > 5e6:
        new_len = g["patch_y"] * m["f_dip"] / F_CENTER
        new_inset = g["inset_depth"] * new_len / g["patch_y"]
        print(f"\nSuggested next try  : patch length {new_len:.3f} mm, "
              f"inset {new_inset:.3f} mm")

    # ---------------- plots ----------------
    fg = f / 1e9

    plt.figure("S11")
    plt.plot(fg, s11, lw=2, label="$S_{11}$")
    plt.axvline(BAND_LOW/1e9, ls="--", color="0.4", label="Band edges")
    plt.axvline(BAND_HIGH/1e9, ls="--", color="0.4")
    plt.axhline(S11_LIMIT, ls=":", color="0.4", label="-10 dB")
    plt.plot(m["f_dip"]/1e9, m["s11_min"], "o", ms=8, mfc="none", mew=2, label="Minimum")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("$S_{11}$ (dB)")
    plt.title("$S_{11}$"); plt.grid(True); plt.legend()

    plt.figure("VSWR")
    plt.plot(fg, vswr, lw=1.8)
    plt.axvline(BAND_LOW/1e9, ls="--", color="0.4")
    plt.axvline(BAND_HIGH/1e9, ls="--", color="0.4")
    plt.axhline(2, ls=":", color="0.4")
    plt.ylim(1, 10)
    plt.xlabel("Frequency (GHz)"); plt.ylabel("VSWR")
    plt.title("VSWR"); plt.grid(True)

    plt.figure("Input impedance")
    plt.plot(fg, zin.real, lw=1.8, label="Resistance R")
    plt.plot(fg, zin.imag, lw=1.8, label="Reactance X")
    plt.axhline(Z0, ls="--", color="0.4")
    plt.axhline(0, ls=":", color="0.4")
    plt.axvline(BAND_LOW/1e9, ls="--", color="0.4")
    plt.axvline(BAND_HIGH/1e9, ls="--", color="0.4")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Impedance ($\\Omega$)")
    plt.title("Input Impedance (at port measurement plane)")
    plt.grid(True); plt.legend()

    fig, ax = plt.subplots(num="Smith chart")
    smith_axes(ax)
    gm = r["gamma"]
    ax.plot(gm.real, gm.imag, lw=1.8, label="$S_{11}$")
    i = int(np.argmin(s11))
    ax.plot(gm[i].real, gm[i].imag, "o", mfc="none", mew=2, label="Minimum")
    for k, lab in ((m["i_low"], "2.400"), (m["i_high"], "2.4835")):
        ax.plot(gm[k].real, gm[k].imag, "s", mfc="none", label=lab)
    ax.set_title("$S_{11}$ Smith Chart"); ax.legend(loc="upper left")

    # ---------------- radiation pattern ----------------
    if RUN_PATTERN:
        print(f"\nCalculating far field at {F_CENTER/1e9:.4f} GHz...")
        theta = np.arange(-180.0, 180.5, 2.0)
        phi = [0.0, 90.0]
        res = nf2ff.CalcNF2FF(sim_path, F_CENTER, theta, phi,
                              center=[0, 0, g["board_thk"] * UNIT])
        e_norm = res.E_norm[0]
        d_max = res.Dmax[0]
        p_rad = res.Prad[0]
        ic = m["i_ctr"]

        eff = p_rad / r["p_acc"][ic]
        pat_db = (20 * np.log10(e_norm / e_norm.max() + 1e-12)
                  + 10 * np.log10(d_max) + 10 * np.log10(eff))    # gain, dBi
        i0 = int(np.argmin(np.abs(theta)))
        i180 = int(np.argmin(np.abs(theta - 180)))

        print(f"Max directivity     : {10*np.log10(d_max):.2f} dBi")
        print(f"Radiation eff.      : {100*eff:.1f} %")
        print(f"Max gain            : {10*np.log10(d_max*eff):.2f} dBi")
        print(f"Max realized gain   : {10*np.log10(d_max*p_rad/r['p_inc'][ic]):.2f} dBi")
        print(f"Front-to-back ratio : {pat_db[i0, 1] - pat_db[i180, 1]:.1f} dB")

        floor = -30
        fig = plt.figure("Radiation pattern")
        axp = fig.add_subplot(projection="polar")
        axp.plot(np.deg2rad(theta), np.maximum(pat_db[:, 1], floor), lw=2,
                 label="E-plane (yz, phi=90)")
        axp.plot(np.deg2rad(theta), np.maximum(pat_db[:, 0], floor), "--", lw=2,
                 label="H-plane (xz, phi=0)")
        axp.set_theta_zero_location("N")
        axp.set_theta_direction(-1)
        axp.set_ylim(floor, np.ceil(pat_db.max()) + 2)
        axp.set_title(f"Gain (dBi) at {F_CENTER/1e9:.3f} GHz")
        axp.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.15))
    else:
        print("\n(Radiation pattern skipped - set RUN_PATTERN = True)")

    # ---------------- pass / fail ----------------
    res_ok = (not m["at_edge"]) and BAND_LOW <= m["f_dip"] <= BAND_HIGH
    pf = lambda ok: "PASS" if ok else "FAIL"
    print("\nDESIGN CHECK")
    print(f"Resonance inside band      : {pf(res_ok)}")
    print(f"S11 <= -10 dB across band  : {pf(m['worst_s11'] <= S11_LIMIT)}")
    print(f"VSWR <= 2 across band      : {pf(m['worst_vswr'] <= 2)}")

    # ---------------- optional studies ----------------
    if RUN_MESH_STUDY:
        print("\nMESH STUDY (cells per substrate wavelength)")
        for div in MESH_STUDY_DIVS:
            path = os.path.join(SIM_ROOT, f"mesh_{div}")
            p, _, nk, tk = build_and_run(g, EPS_R, TAN_D, div, path, want_nf2ff=False)
            mk = metrics(analyze(p, path))
            print(f"  div {div:3d} ({np.prod(nk)/1e3:5.0f}k cells): "
                  f"{mk['f_dip']/1e9:.4f} GHz, {mk['s11_min']:.2f} dB  [{tk:.0f} s]")
        print("  Converged when resonance changes < ~0.5% between steps.")

    if RUN_TOLERANCE_STUDY:
        print(f"\nTOLERANCE STUDY (epsR +/- {EPS_R_TOL:.2f})")
        for er in (EPS_R - EPS_R_TOL, EPS_R, EPS_R + EPS_R_TOL):
            path = os.path.join(SIM_ROOT, f"eps_{er:.2f}")
            p, _, _, _ = build_and_run(g, er, TAN_D, MESH_DIV, path, want_nf2ff=False)
            mk = metrics(analyze(p, path))
            print(f"  epsR {er:.2f} : {mk['f_dip']/1e9:.4f} GHz, {mk['s11_min']:.2f} dB")

    print(f"\nSIMULATION COMPLETE in {(time.time() - t_all)/60:.1f} min.")
    plt.show()


if __name__ == "__main__":
    main()#!/usr/bin/env python3

