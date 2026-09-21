# patch_antenna_inset.py
# The 2.4 GHz patch with a REAL inset feed: the feed line runs into a notch
# cut into the patch, connecting at the point where the impedance drops to
# 50 ohms. This matches how your KiCad board is actually built.
#
# The key knob is `feed_inset` (how deep the notch goes). That's what tunes
# the MATCH (how deep the S11 dip is). Sweep it to find the best value.

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

Sim_Path = os.path.join(os.path.dirname(__file__), 'patch_inset_sim')

# =====================================================================
#  DIMENSIONS (mm)
# =====================================================================

# --- Patch ---
patch_width  = 30.4
patch_length = 29.6

# --- Feed line + inset notch ---
feed_width = 3.0       # ~50 ohm line width
feed_gap   = 0.5       # air clearance on EACH side of the line inside the notch
feed_inset = 8.71       # <<<<< THE MATCHING KNOB: how deep the notch goes.
                       #        Sweep this (try 4, 6, 8, 10, 12) to deepen the dip.
feed_length = 15.0     # how far the line sticks out past the patch edge

notch_half_width = feed_width/2 + feed_gap     # half-width of the cut-out slot
notch_bottom_y   = patch_length/2 - feed_inset  # y where the line meets the patch
line_end_y       = patch_length/2 + feed_length # y where the line (and port) ends

# --- Substrate (FR4) ---
substrate_epsR      = 4.4
substrate_tan_delta = 0.02
substrate_thickness = 1.6
substrate_cells     = 4
substrate_kappa = substrate_tan_delta * 2*np.pi*2.4e9 * EPS0 * substrate_epsR
h = substrate_thickness   # shorthand, used a lot below

sub_x_min, sub_x_max = -25, 25
sub_y_min, sub_y_max = -25, line_end_y + 3

# --- Feed / port ---
feed_R = 50

# --- Air box ---
air_x_min, air_x_max = -50, 50
air_y_min, air_y_max = -45, line_end_y + 20
air_z_min, air_z_max = -40, 80

# =====================================================================
#  FDTD SETUP
# =====================================================================
f0 = 2.4e9
fc = 1.0e9

FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['MUR']*6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)
mesh_res = C0/(f0+fc)/1e-3/20

# =====================================================================
#  BUILD THE GEOMETRY
# =====================================================================
mesh.AddLine('x', [air_x_min, air_x_max])
mesh.AddLine('y', [air_y_min, air_y_max])
mesh.AddLine('z', [air_z_min, air_z_max])

# --- Patch with an inset notch, built from three boxes (see explanation) ---
patch = CSX.AddMetal('patch')
# left slab (full patch height, left of the notch)
patch.AddBox(priority=10,
             start=[-patch_width/2,   -patch_length/2, h],
             stop =[-notch_half_width, patch_length/2, h])
# right slab (full patch height, right of the notch)
patch.AddBox(priority=10,
             start=[ notch_half_width, -patch_length/2, h],
             stop =[ patch_width/2,     patch_length/2, h])
# center slab (only up to the notch bottom -> leaves the slot empty above it)
patch.AddBox(priority=10,
             start=[-notch_half_width, -patch_length/2, h],
             stop =[ notch_half_width, notch_bottom_y,  h])

# --- Feed line: from just inside the notch bottom out past the patch edge ---
# (starts 1mm below the notch bottom so it solidly joins the center slab)
patch.AddBox(priority=10,
             start=[-feed_width/2, notch_bottom_y - 1, h],
             stop =[ feed_width/2, line_end_y,         h])

FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2)

# The notch and its thin gaps are small, so force a fine mesh across them
mesh.AddLine('x', np.linspace(-notch_half_width, notch_half_width, 21))

# --- Substrate ---
substrate = CSX.AddMaterial('substrate', epsilon=substrate_epsR, kappa=substrate_kappa)
substrate.AddBox(priority=0,
                 start=[sub_x_min, sub_y_min, 0],
                 stop =[sub_x_max, sub_y_max, h])
mesh.AddLine('z', np.linspace(0, h, substrate_cells+1))

# --- Ground plane ---
gnd = CSX.AddMetal('gnd')
gnd.AddBox(priority=10,
           start=[sub_x_min, sub_y_min, 0],
           stop =[sub_x_max, sub_y_max, 0])
FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

# --- Lumped port at the end of the feed line ---
port = FDTD.AddLumpedPort(1, feed_R,
                          [-feed_width/2, line_end_y, 0],
                          [ feed_width/2, line_end_y, h],
                          'z', 1.0, priority=5, edges2grid='xy')

mesh.SmoothMeshLines('all', mesh_res, 1.4)

# =====================================================================
#  RUN
# =====================================================================
FDTD.Run(Sim_Path, cleanup=True)

# =====================================================================
#  POST-PROCESSING: S11 vs frequency
# =====================================================================
import matplotlib.pyplot as plt

f = np.linspace(max(1e9, f0-fc), f0+fc, 401)
port.CalcPort(Sim_Path, f)
s11    = port.uf_ref / port.uf_inc
s11_dB = 20.0*np.log10(np.abs(s11))

i_res = np.argmin(s11_dB)
print(f"\nInset = {feed_inset} mm  ->  Lowest S11 = {s11_dB[i_res]:.1f} dB at {f[i_res]/1e9:.3f} GHz\n")

plt.figure()
plt.plot(f/1e9, s11_dB, 'k-', linewidth=2)
plt.grid(True)
plt.ylabel('S11 (dB)')
plt.xlabel('Frequency (GHz)')
plt.title(f'Inset-fed patch (inset = {feed_inset} mm)')
plt.savefig(os.path.join(os.path.dirname(__file__), 's11_inset.png'), dpi=120)
plt.show()



# =====================================================================
#  INPUT IMPEDANCE: REAL AND IMAGINARY PARTS
# =====================================================================

Zin = port.uf_tot / port.if_tot
i_24 = np.argmin(np.abs(f - 2.4e9))

print("\n===================================")
print("IMPEDANCE AT 2.4 GHz")
print("===================================")
print(f"Frequency = {f[i_24]/1e9:.3f} GHz")
print(f"Zin       = {Zin[i_24]}")
print(f"Resistance = {np.real(Zin[i_24]):.3f} Ohms")
print(f"Reactance  = {np.imag(Zin[i_24]):.3f} Ohms")
print("===================================\n")
Z_real = np.real(Zin)
Z_imag = np.imag(Zin)

# Find impedance at 2.4 GHz
i_f0 = np.argmin(np.abs(f - f0))

print("Input impedance at 2.4 GHz:")
print(f"  R = {Z_real[i_f0]:.2f} ohms")
print(f"  X = {Z_imag[i_f0]:.2f} ohms")
print(f"  Zin = {Zin[i_f0]:.2f} ohms")

# -------------------------------------------------------------
# Plot real and imaginary impedance
# -------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(f/1e9, Z_real, label='Real(Zin)', linewidth=2)
plt.plot(f/1e9, Z_imag, label='Imag(Zin)', linewidth=2)

# 50-ohm reference
plt.axhline(50, linestyle='--', linewidth=1)

# Zero-reactance reference
plt.axhline(0, linestyle=':', linewidth=1)

plt.grid(True)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Input Impedance (Ohms)')
plt.title(f'Input Impedance (Inset = {feed_inset} mm)')
plt.legend()

plt.savefig(
    os.path.join(os.path.dirname(__file__), 'impedance_inset.png'),
    dpi=120
)

plt.show()
