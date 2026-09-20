# patch_antenna.py
# A 2.4 GHz microstrip patch antenna, rebuilt from the KiCad board.
#
# This builds on your dipole script. The four stages are the same
# (geometry -> mesh -> excitation/boundaries -> run), with three new
# ingredients: a dielectric SUBSTRATE, a GROUND plane, and a vertical
# lumped-port FEED. At the end it plots S11 (reflection vs frequency)
# so you can see where the antenna actually resonates.

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0   # speed of light, vacuum permittivity

# Put simulation output in a "patch_sim" folder next to this script
Sim_Path = os.path.join(os.path.dirname(__file__), 'patch_sim')

# =====================================================================
#  DIMENSIONS  (all in millimetres, taken straight from the KiCad board)
# =====================================================================

# --- The patch: the copper rectangle that does the radiating ---
patch_width  = 30.4    # x-direction size
patch_length = 23.4    # y-direction size

# --- The substrate: the FR4 board the patch sits on ---
# (Your real board is 70x67mm. We use a representative 50x50mm centred
#  on the patch: smaller = faster to simulate, still plenty of margin.)
substrate_epsR      = 4.4     # FR4 relative permittivity (how much it slows the wave)
substrate_tan_delta = 0.02    # FR4 loss tangent (how lossy the material is)
substrate_thickness = 1.6     # board thickness
substrate_width     = 50      # x-direction size
substrate_length    = 50      # y-direction size
substrate_cells     = 4       # number of mesh layers to slice the 1.6mm into

# Convert the loss tangent into the "conductivity" number openEMS wants
substrate_kappa = substrate_tan_delta * 2*np.pi*2.4e9 * EPS0 * substrate_epsR

# --- The feed ---
# The real board uses an inset microstrip feed. We approximate it with a
# vertical "probe": a lumped port going straight up from the ground plane
# to the patch, 2mm in from the feeding edge.
feed_pos_y = patch_length/2 - 2   # 2mm inset from the +y edge of the patch
feed_R     = 50                   # 50 ohm source, standard for RF

# --- The air box ---
# The empty space around the antenna where the wave travels before being
# absorbed at the walls. The antenna radiates upward (away from ground),
# so we leave extra room above it (the z line is asymmetric on purpose).
SimBox = np.array([160, 160, 120])

# =====================================================================
#  FDTD SETUP  (the solver engine + the excitation pulse)
# =====================================================================

f0 = 2.4e9    # centre frequency of the pulse
fc = 1.0e9    # bandwidth: the pulse covers f0 +/- fc, i.e. ~1.4 to 3.4 GHz

FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
# MUR = a thin absorbing boundary (the simpler cousin of PML). All 6 walls
# absorb, so radiated energy leaves the box instead of bouncing back in.
FDTD.SetBoundaryCond(['MUR']*6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)                 # our numbers are in millimetres
mesh_res = C0/(f0+fc)/1e-3/20           # target cell size: ~1/20th of a wavelength

# =====================================================================
#  BUILD THE GEOMETRY
# =====================================================================

# Seed the mesh with the outer air-box corners (more room above in z)
mesh.AddLine('x', [-SimBox[0]/2, SimBox[0]/2])
mesh.AddLine('y', [-SimBox[1]/2, SimBox[1]/2])
mesh.AddLine('z', [-SimBox[2]/3, SimBox[2]*2/3])

# --- Patch (top copper) ---
patch = CSX.AddMetal('patch')
patch.AddBox(priority=10,
             start=[-patch_width/2, -patch_length/2, substrate_thickness],
             stop =[ patch_width/2,  patch_length/2, substrate_thickness])
# Drop extra-fine mesh lines exactly on the patch edges (where the field
# is strongest). This is what makes the resonant frequency come out right.
FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2)

# --- Substrate (the FR4 slab) ---
# AddMaterial, not AddMetal: a dielectric. epsilon slows the wave, kappa adds loss.
substrate = CSX.AddMaterial('substrate', epsilon=substrate_epsR, kappa=substrate_kappa)
substrate.AddBox(priority=0,
                 start=[-substrate_width/2, -substrate_length/2, 0],
                 stop =[ substrate_width/2,  substrate_length/2, substrate_thickness])
# Slice the thin 1.6mm thickness into a few mesh layers so the fields
# inside the substrate are properly resolved.
mesh.AddLine('z', np.linspace(0, substrate_thickness, substrate_cells+1))

# --- Ground plane (bottom copper, a flat sheet at z=0) ---
gnd = CSX.AddMetal('gnd')
gnd.AddBox(priority=10,
           start=[-substrate_width/2, -substrate_length/2, 0],
           stop =[ substrate_width/2,  substrate_length/2, 0])
FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

# --- Feed (lumped port, straight up from ground to patch) ---
port = FDTD.AddLumpedPort(1, feed_R,
                          [0, feed_pos_y, 0],                    # start: on the ground
                          [0, feed_pos_y, substrate_thickness],  # stop: at the patch
                          'z', 1.0, priority=5, edges2grid='xy')

# Fill in the rest of the mesh between the fixed lines, up to mesh_res,
# letting cell size grow gradually (ratio 1.4) so it never jumps too fast.
mesh.SmoothMeshLines('all', mesh_res, 1.4)

# =====================================================================
#  RUN
# =====================================================================
FDTD.Run(Sim_Path, cleanup=True)

# =====================================================================
#  POST-PROCESSING: plot S11 (reflection) vs frequency
# =====================================================================
import matplotlib.pyplot as plt

f = np.linspace(max(1e9, f0-fc), f0+fc, 401)
port.CalcPort(Sim_Path, f)
s11    = port.uf_ref / port.uf_inc          # reflected wave / incident wave
s11_dB = 20.0*np.log10(np.abs(s11))

# Report the resonance (the frequency where least power is reflected)
i_res = np.argmin(s11_dB)
print(f"\nLowest S11 = {s11_dB[i_res]:.1f} dB at {f[i_res]/1e9:.3f} GHz\n")

plt.figure()
plt.plot(f/1e9, s11_dB, 'k-', linewidth=2)
plt.grid(True)
plt.ylabel('S11 (dB)')
plt.xlabel('Frequency (GHz)')
plt.title('Patch antenna reflection')
plt.savefig(os.path.join(os.path.dirname(__file__), 's11.png'), dpi=120)
plt.show()
