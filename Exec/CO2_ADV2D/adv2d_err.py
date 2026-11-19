#!/usr/bin/env python3
"""adv2d_err.py shift PLT PLT0 NX NY     the L1 error per variable of PLT against
                                        PLT0 translated by (NX, NY) cells, periodic
   adv2d_err.py coarsen PLT PLTFINE       the L1 difference per variable of PLT
                                        against PLTFINE block-averaged to PLT's grid

Single-level plotfiles.  With the translation an integer number of cells
(stop_time = travel / u0 with travel a multiple of Lx/N on every grid of a
series) the first form needs no interpolation and measures every variable,
conserved or not; the second is the self-convergence of a case with no exact
solution.  Pure numpy.  docs/PLAN_high_order.pdf follow-up 12.
"""
import sys, re, glob, numpy as np

def read(p):
    hdr = open(p + '/Header').read().split('\n')
    nv = int(hdr[1]); nm = hdr[2:2 + nv]
    n = [int(v) + 1 for v in re.findall(r'\((\d+),(\d+)', hdr[2 + nv + 6])[1]]
    d = p + '/Level_0'
    cH = open(glob.glob(d + '/Cell_H')[0]).read().split('\n')
    fn = [l.split()[1] for l in cH if l.startswith('FabOnDisk:')]
    of = [int(l.split()[-1]) for l in cH if l.startswith('FabOnDisk:')]
    bo = [tuple(int(x) for x in m.groups()) for m in
          (re.match(r'\(\((\d+),(\d+)\) \((\d+),(\d+)\)', ln.strip()) for ln in cH) if m]
    full = np.full((nv, n[1], n[0]), np.nan)
    for (il, jl, ih, jh), fnm, off in zip(bo, fn, of):
        dat = open(d + '/' + fnm, 'rb').read(); q = dat.find(b'\n', off) + 1
        bx = ih - il + 1; by = jh - jl + 1
        a = np.frombuffer(dat[q:q + 8 * nv * bx * by], dtype='<f8').reshape(nv, by, bx)
        full[:, jl:jh + 1, il:ih + 1] = a
    return nm, full

if __name__ == '__main__':
    mode = sys.argv[1]
    nm, a = read(sys.argv[2]); nm2, b = read(sys.argv[3])
    if mode == 'shift':
        sx, sy = int(sys.argv[4]), int(sys.argv[5])
        ref = np.roll(np.roll(b, sx, axis=2), sy, axis=1)
    elif mode == 'coarsen':
        r = b.shape[2] // a.shape[2]
        assert b.shape[1] // a.shape[1] == r and b.shape[2] == r * a.shape[2]
        ref = b.reshape(b.shape[0], a.shape[1], r, a.shape[2], r).mean(axis=(2, 4))
    else:
        sys.exit('mode shift or coarsen')
    for k, name in enumerate(nm):
        if name not in nm2: continue
        k2 = nm2.index(name)
        e = np.abs(a[k] - ref[k2]).mean()
        print(f'{name:16s} L1 = {e:.6e}   range {np.nanmin(ref[k2]):.6e} .. {np.nanmax(ref[k2]):.6e}')
