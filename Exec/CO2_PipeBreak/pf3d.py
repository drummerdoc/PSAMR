#!/usr/bin/env python3
"""pf3d.py mirror PLT            the mirror asymmetry of a single-level 3-D plotfile across the
                                 y and z mid-planes (odd slots: y_velocity/ymom in y, z_velocity/zmom
                                 in z), max |f - mirror(f)| and its ratio to the field's range
   pf3d.py diff PLT_A PLT_B      max |A - B| per variable and its ratio to A's range (level 0)
   pf3d.py axis PLT [VAR]        the jet-axis profile (y = z = mid) of VAR (default pressure)

Level 0 only, pure numpy, any dimension (the box regex reads 1-, 2- or 3-D).  docs/PLAN_jet3d.md."""
import sys, re, glob, numpy as np

def read(p):
    hdr = open(p + '/Header').read().split('\n'); nv = int(hdr[1]); names = hdr[2:2 + nv]
    t = float(hdr[2 + nv + 1])
    d = p + '/Level_0'
    cH = open(d + '/Cell_H').read().split('\n')
    fn = [l.split()[1] for l in cH if l.startswith('FabOnDisk:')]
    of = [int(l.split()[-1]) for l in cH if l.startswith('FabOnDisk:')]
    bo = []
    for ln in cH:
        m = re.match(r'\(\(([\-\d,]+)\) \(([\-\d,]+)\) \(([\d,]+)\)\)\s*$', ln.strip())
        if m: bo.append((tuple(int(x) for x in m.group(1).split(',')), tuple(int(x) for x in m.group(2).split(','))))
    dim = len(bo[0][0])
    hi = [max(b[1][k] for b in bo) + 1 for k in range(dim)]
    full = np.full([nv] + hi[::-1], np.nan)
    for (lo, hb), fnm, off in zip(bo, fn, of):
        dat = open(d + '/' + fnm, 'rb').read(); q = dat.find(b'\n', off) + 1
        n = [hb[k] - lo[k] + 1 for k in range(dim)]; cnt = nv
        for x in n: cnt *= x
        a = np.frombuffer(dat[q:q + 8 * cnt], dtype='<f8').reshape([nv] + n[::-1])
        sl = tuple([slice(None)] + [slice(lo[k], hb[k] + 1) for k in reversed(range(dim))])
        full[sl] = a
    return names, t, full, dim

def rng(a): return float(np.nanmax(a) - np.nanmin(a))

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'mirror':
        nm, t, F, dim = read(sys.argv[2])
        print('%s  t = %.6e  dim %d  shape %s' % (sys.argv[2], t, dim, F.shape[1:]))
        axes = {'y': (dim - 2, ('y_velocity', 'ymom'))}
        if dim == 3: axes['z'] = (dim - 3, ('z_velocity', 'zmom'))
        for ax, (k, odd) in axes.items():
            arr_axis = 1 + k   # F axes: (var, z, y, x) in 3-D -> y is axis 2, z is axis 1
            arr_axis = {'y': F.ndim - 2, 'z': F.ndim - 3}[ax]
            worst = 0.0; wname = ''
            print('  -- mirror in %s --' % ax)
            for i, n in enumerate(nm):
                a = F[i]; m = np.flip(a, axis=arr_axis - 1)
                if n in odd: m = -m
                d = float(np.nanmax(np.abs(a - m))); r = rng(a); rel = d / r if r > 0 else 0.0
                if rel > worst: worst, wname = rel, n
                print('  %-14s max|asym| %10.3e  range %10.3e  rel %8.2e' % (n, d, r, rel))
            print('  worst asymmetry in %s: %.2e (%s)' % (ax, worst, wname))
    elif mode == 'diff':
        na, ta, A, dim = read(sys.argv[2]); nb, tb, B, _ = read(sys.argv[3])
        assert na == nb and A.shape == B.shape
        print('%s vs %s  t = %.6e %.6e' % (sys.argv[2], sys.argv[3], ta, tb)); worst = 0.0
        for i, n in enumerate(na):
            d = float(np.nanmax(np.abs(A[i] - B[i]))); r = rng(A[i]); rel = d / r if r > 0 else 0.0; worst = max(worst, rel)
            print('  %-14s max|d| %10.3e  range %10.3e  rel %8.2e' % (n, d, r, rel))
        print('  worst relative-to-range: %.2e' % worst)
    elif mode == 'axis':
        nm, t, F, dim = read(sys.argv[2]); v = sys.argv[3] if len(sys.argv) > 3 else 'pressure'
        a = F[nm.index(v)]
        if dim == 3: prof = a[a.shape[0] // 2, a.shape[1] // 2, :]
        else: prof = a[a.shape[0] // 2, :]
        print(' '.join('%.6e' % x for x in prof))
    else:
        sys.exit('mode mirror | diff | axis')
