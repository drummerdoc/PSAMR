#!/usr/bin/env python3
"""XC2D coarse-fine ray diff: a two-level AMR plotfile against a uniform-256 plotfile
at the same time, along the mid-height ray (docs/VERIFICATION.md 7.3).
    python3 ray_diff.py AMR_PLT REF256_PLT
Protocol.  The uniform run's rows j=128,129 (the two fine rows straddling y=0.5)
are the reference.  Where level 1 covers the ray the fine cells (row j=128) are
compared cell by cell with the uniform row 128.  Where only level 0 covers it,
each coarse cell is compared with the mean of the uniform 2x2 block beneath it.
Zones: fine interior (fine-covered, more than 2 fine cells from a coarse-fine
edge), coarse-fine edge (2 fine cells / 1 coarse cell on each side), coarse
elsewhere.  Reported per zone: max |d alpha_1|, max |d E_1| / max |E_1| on the
ray, max |dP| / max P.  Then the cells where one run holds a phase the other
has removed."""
import sys, glob, re, numpy as np
def pf_read(pf):
    H = open(pf + '/Header').read().split('\n'); nv = int(H[1]); names = H[2:2 + nv]
    t = float(H[2 + nv + 1]); out = {}
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
A, R = sys.argv[1], sys.argv[2]
na, ta, Da = pf_read(A); nr, tr, Dr = pf_read(R); assert na == nr
ix = {n: i for i, n in enumerate(na)}
print('%s (t=%.10e)  vs  %s (t=%.10e)  dt_rel=%.1e' % (A, ta, R, tr, abs(ta - tr) / tr))
L0, L1 = Da[0], Da.get(1); U = Dr[0]
if L1 is not None and L1.shape != U.shape:          # pad the level-1 array to the full fine index space
    P = np.full(U.shape, np.nan); P[:, :L1.shape[1], :L1.shape[2]] = L1; L1 = P
nxc = L0.shape[2]; nxf = U.shape[2]; jc = nxc // 2; jf = nxf // 2       # rows straddling y = 0.5
fine_cov = np.zeros(nxf, bool)
if L1 is not None: fine_cov = ~np.isnan(L1[ix['alpha_1'], jf, :])
# composite along the ray at fine resolution, and the reference at matching resolution
comp = {}; ref = {}
for v in ('alpha_1', 'alpha1_rho1_E1', 'pressure'):
    c = np.full(nxf, np.nan); r = np.full(nxf, np.nan)
    for i in range(nxf):
        if fine_cov[i]:
            c[i] = L1[ix[v], jf, i]; r[i] = U[ix[v], jf, i]
        else:
            ic = i // 2; c[i] = L0[ix[v], jc, ic]
            r[i] = U[ix[v], jf - 1:jf + 1, 2 * ic:2 * ic + 2].mean()
    comp[v] = c; ref[v] = r
# zones
edge = np.zeros(nxf, bool)
for i in range(1, nxf):
    if fine_cov[i] != fine_cov[i - 1]:
        edge[max(0, i - 2):min(nxf, i + 2)] = True
xs = (np.arange(nxf) + 0.5) * (1.0 / nxf); contact = np.abs(xs - 0.494) < 0.02   # the contact sits in the coarse region
zones = {'fine interior': fine_cov & ~edge, 'coarse-fine edge': edge, 'coarse, contact band': ~fine_cov & ~edge & contact, 'coarse elsewhere': ~fine_cov & ~edge & ~contact}
Emax = np.nanmax(np.abs(ref['alpha1_rho1_E1'])); Pmax = np.nanmax(ref['pressure'])
dx = 1.0 / nxf
print('fine-covered x ranges:', end=' ')
i = 0
while i < nxf:
    if fine_cov[i]:
        j = i
        while j < nxf and fine_cov[j]: j += 1
        print('[%.4f, %.4f]' % (i * dx, j * dx), end=' '); i = j
    else: i += 1
print()
print('%-18s %6s %12s %12s %12s' % ('zone', 'cells', 'max|d a1|', 'max|dE1|/Emax', 'max|dP|/Pmax'))
for z, m in zones.items():
    if not m.any(): print('%-18s %6d' % (z, 0)); continue
    da = np.nanmax(np.abs(comp['alpha_1'][m] - ref['alpha_1'][m]))
    de = np.nanmax(np.abs(comp['alpha1_rho1_E1'][m] - ref['alpha1_rho1_E1'][m])) / Emax
    dp = np.nanmax(np.abs(comp['pressure'][m] - ref['pressure'][m])) / Pmax
    print('%-18s %6d %12.2e %12.2e %12.2e' % (z, m.sum(), da, de, dp))
# phase presence disagreements
a_c, a_r = comp['alpha_1'], ref['alpha_1']
print('cells where the AMR composite is pure (alpha_1 exactly 0 or 1) and the reference is not:')
n = 0
for i in range(nxf):
    if (a_c[i] == 1.0 or a_c[i] == 0.0) and not (a_r[i] == 1.0 or a_r[i] == 0.0):
        n += 1
        if n <= 12: print('  x=%.5f  AMR alpha_1=%.1f  ref alpha_2=%.3e  (%s)' % ((i + 0.5) * dx, a_c[i], 1 - a_r[i], 'fine' if fine_cov[i] else 'coarse'))
print('  count:', n)
print('cells where the reference is pure and the AMR composite is not:')
n = 0
for i in range(nxf):
    if (a_r[i] == 1.0 or a_r[i] == 0.0) and not (a_c[i] == 1.0 or a_c[i] == 0.0):
        n += 1
        if n <= 12: print('  x=%.5f  ref alpha_1=%.1f  AMR alpha_2=%.3e  (%s)' % ((i + 0.5) * dx, a_r[i], 1 - a_c[i], 'fine' if fine_cov[i] else 'coarse'))
print('  count:', n)
