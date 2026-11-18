# step_sensitivity_masked.py — the difference between two 2-D plotfiles at
# one time, reported apart on the cells near a wave or a front and on the
# rest, so an order reduction localized at the waves shows as such
# (docs/PLAN_high_order.pdf, the register items of 18 Nov 2026).
#   python3 step_sensitivity_masked.py pltA pltB [alpha_jump=0.05] [dilate=3] [wave_jump=0]
# The mask: cells where the flash fired in either run, where alpha_1 jumps
# by more than alpha_jump to a neighbour, or (wave_jump > 0) where the
# pressure or the density jumps by more than wave_jump of its range to a
# neighbour; dilated by `dilate` cells.  Per level and variable: L1 and
# Linf of |A - B| over the range, on the unmasked cells, the masked cells,
# and all.  Reads the plotfiles directly (no AMReX Python needed).
import sys, glob, re, numpy as np
def pf_read(pf):
    H = open(pf + '/Header').read().split('\n'); nv = int(H[1]); names = H[2:2 + nv]; t = float(H[2 + nv + 1]); out = {}
    for d in sorted(glob.glob(pf + '/Level_*')):
        lev = int(d.split('_')[-1]); fabs = []
        for fn in sorted(glob.glob(d + '/Cell_D_*')):
            b = open(fn, 'rb').read(); p = 0
            while p < len(b):
                e = b.index(b'\n', p); h = b[p:e].decode(); p = e + 1
                m = re.search(r'\(\((-?\d+),(-?\d+)\) \((-?\d+),(-?\d+)\) \([\d,]+\)\)\s+(\d+)\s*$', h)
                lo0, lo1, hi0, hi1, nc = map(int, m.groups()); nx, ny = hi0 - lo0 + 1, hi1 - lo1 + 1
                a = np.frombuffer(b[p:p + 8 * nx * ny * nc], '<f8').reshape(nc, ny, nx); p += 8 * nx * ny * nc
                fabs.append((lo0, lo1, hi0, hi1, a))
        X = max(f[2] for f in fabs) + 1; Y = max(f[3] for f in fabs) + 1
        D = np.full((nv, Y, X), np.nan)
        for lo0, lo1, hi0, hi1, a in fabs: D[:, lo1:hi1 + 1, lo0:hi0 + 1] = a
        out[lev] = D
    return names, t, out
def dilate(m, n):
    for _ in range(n):
        m2 = m.copy(); m2[1:] |= m[:-1]; m2[:-1] |= m[1:]; m2[:, 1:] |= m[:, :-1]; m2[:, :-1] |= m[:, 1:]; m = m2
    return m
A, B = sys.argv[1], sys.argv[2]; ajump = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05; ndil = int(sys.argv[4]) if len(sys.argv) > 4 else 3
na, ta, Da = pf_read(A); nb, tb, Db = pf_read(B); assert na == nb
print(f'{A}\n  vs {B}\n  t = {ta:.6e} {tb:.6e}   mask: flash_rate != 0, |d alpha_1| > {ajump} or a P or rho jump above the fifth argument of the range, to a neighbour, dilated {ndil} cells')
for lev in sorted(Da):
    La, Lb = Da[lev], Db[lev]; ia = na.index('alpha_1')
    al = La[ia]; front = np.zeros_like(al, bool)
    if 'flash_rate' in na: front |= (La[na.index('flash_rate')] != 0.0) | (Lb[na.index('flash_rate')] != 0.0)
    for s in (1, -1):
        front[:, :] |= np.abs(al - np.roll(al, s, axis=1)) > ajump
        front[:, :] |= np.abs(al - np.roll(al, s, axis=0)) > ajump
    #  and the waves: a pressure or density jump to a neighbour above gjump of the range
    gjump = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    if gjump > 0:
        for v in ('pressure', 'density'):
            q = La[na.index(v)]; r = np.nanmax(q) - np.nanmin(q)
            for s in (1, -1):
                front[:, :] |= np.abs(q - np.roll(q, s, axis=1)) > gjump * r
                front[:, :] |= np.abs(q - np.roll(q, s, axis=0)) > gjump * r
    front = dilate(front, ndil); valid = ~np.isnan(al) & ~np.isnan(Lb[ia])
    fm, sm = front & valid, ~front & valid
    print(f'  level {lev}: {valid.sum()} cells, front {fm.sum()} ({100*fm.sum()/valid.sum():.1f} %), smooth {sm.sum()}')
    print(f'  {"var":12s} {"L1/range smooth":>16s} {"Linf/range smooth":>18s} {"L1/range front":>15s} {"Linf/range front":>17s} {"L1/range all":>13s}')
    for v in ('density', 'pressure', 'alpha_1', 'alpha2_rho2', 'x_velocity'):
        if v not in na: continue
        i = na.index(v); d = np.abs(La[i] - Lb[i]); r = np.nanmax(La[i]) - np.nanmin(La[i])
        if r <= 0: continue
        print(f'  {v:12s} {d[sm].mean()/r:16.2e} {d[sm].max()/r:18.2e} {d[fm].mean()/r if fm.any() else 0:15.2e} {d[fm].max()/r if fm.any() else 0:17.2e} {d[valid].mean()/r:13.2e}')
