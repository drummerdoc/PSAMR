#!/usr/bin/env python3
"""contact_width.py PLOTFILE [...] [--var NAME] [--y Y] [--window XLO XHI] [--level L]

The 10-90 % width, in cells and in metres, of the first monotone crossing of
a variable (alpha_1 by default) along a row of the finest level: the row of
a 1-D plotfile, or the row nearest y = Y of a 2-D one (0.5 by default, the
demo3 axis).  The jump is the difference between the row's maximum and
minimum inside the window (the whole row by default); the crossing is the
one through the 50 % level nearest the window's middle, and the width is
the distance between the 10 % and 90 % crossings around it, linearly
interpolated.  Cells of the window not covered by the finest level read as
gaps and stop the search.  Pure numpy, no yt.  docs/PLAN_high_order.pdf
follow-up 11.
"""
import sys, re, glob, numpy as np

def read_row(p, var, y, lev):
    hdr = open(p + '/Header').read().split('\n')
    nv = int(hdr[1]); nm = hdr[2:2 + nv]
    if var not in nm:
        sys.exit(f'{p}: no variable {var}; the plotfile has {nm}')
    vi = nm.index(var); dim = int(hdr[2 + nv])
    t = float(hdr[2 + nv + 1]); finest = int(hdr[2 + nv + 2])
    lo = [float(v) for v in hdr[2 + nv + 3].split()]; hi = [float(v) for v in hdr[2 + nv + 4].split()]
    rr = [int(v) for v in hdr[2 + nv + 5].split()] if finest > 0 else []
    if lev is None: lev = finest
    n0 = [int(v) + 1 for v in re.findall(r'\((\d+),(\d+)' if dim == 2 else r'\((\d+)\)', hdr[2 + nv + 6])[1]] if dim == 2 \
         else [int(re.findall(r'\((\d+)\) \((\d+)\)', hdr[2 + nv + 6])[0][1]) + 1]
    n = list(n0)
    for l in range(lev): n = [v * rr[l] for v in n]
    d = f'{p}/Level_{lev}'
    cH = open(glob.glob(d + '/Cell_H')[0]).read().split('\n')
    fn = [l.split()[1] for l in cH if l.startswith('FabOnDisk:')]
    of = [int(l.split()[-1]) for l in cH if l.startswith('FabOnDisk:')]
    dx = (hi[0] - lo[0]) / n[0]
    if dim == 1:
        bo = [tuple(int(x) for x in m.groups()) for m in
              (re.match(r'\(\((\d+)\) \((\d+)\)', ln.strip()) for ln in cH) if m]
        row = np.full(n[0], np.nan)
        for (il, ih), fnm, off in zip(bo, fn, of):
            dat = open(d + '/' + fnm, 'rb').read(); q = dat.find(b'\n', off) + 1
            bx = ih - il + 1
            a = np.frombuffer(dat[q:q + 8 * nv * bx], dtype='<f8').reshape(nv, bx)
            row[il:ih + 1] = a[vi]
    else:
        bo = [tuple(int(x) for x in m.groups()) for m in
              (re.match(r'\(\((\d+),(\d+)\) \((\d+),(\d+)\)', ln.strip()) for ln in cH) if m]
        dy = (hi[1] - lo[1]) / n[1]
        jrow = min(max(int((y - lo[1]) / dy), 0), n[1] - 1)
        row = np.full(n[0], np.nan)
        for (il, jl, ih, jh), fnm, off in zip(bo, fn, of):
            if not (jl <= jrow <= jh): continue
            dat = open(d + '/' + fnm, 'rb').read(); q = dat.find(b'\n', off) + 1
            bx = ih - il + 1; by = jh - jl + 1
            a = np.frombuffer(dat[q:q + 8 * nv * bx * by], dtype='<f8').reshape(nv, by, bx)
            row[il:ih + 1] = a[vi][jrow - jl]
    x = lo[0] + (np.arange(n[0]) + 0.5) * dx
    return t, x, row, dx

def width(x, row, dx, win):
    m = np.ones(len(x), bool) if win is None else (x >= win[0]) & (x <= win[1])
    idx = np.where(m)[0]
    r = row[idx]; xx = x[idx]
    if np.isnan(r).any():
        return None, 'gap in the finest level inside the window'
    vmin, vmax = r.min(), r.max(); jump = vmax - vmin
    if jump <= 0: return None, 'no variation'
    l10, l50, l90 = vmin + 0.1 * jump, vmin + 0.5 * jump, vmin + 0.9 * jump
    s = np.sign(r - l50); cross = np.where(s[:-1] * s[1:] < 0)[0]
    if len(cross) == 0: return None, 'no crossing of the 50 % level'
    mid = 0.5 * (xx[0] + xx[-1]); c = cross[np.argmin(np.abs(xx[cross] - mid))]
    rising = r[c + 1] > r[c]
    def xcross(level, start, step):
        i = start
        while 0 <= i + step < len(r):
            a, b = r[i], r[i + step]
            if (a - level) * (b - level) <= 0 and a != b:
                return xx[i] + (level - a) / (b - a) * (xx[i + step] - xx[i])
            i += step
        return None
    lo_lvl, hi_lvl = (l10, l90) if rising else (l90, l10)
    xa = xcross(lo_lvl, c, -1); xb = xcross(hi_lvl, c + 1, 1)
    if xa is None or xb is None: return None, 'the 10 or 90 % level is not reached beside the crossing'
    return (abs(xb - xa) / dx, abs(xb - xa), 0.5 * (xa + xb), jump, 'rising' if rising else 'falling'), None

if __name__ == '__main__':
    args = sys.argv[1:]; var = 'alpha_1'; y = 0.5; win = None; lev = None; files = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--var': var = args[i + 1]; i += 2
        elif a == '--y': y = float(args[i + 1]); i += 2
        elif a == '--level': lev = int(args[i + 1]); i += 2
        elif a == '--window': win = (float(args[i + 1]), float(args[i + 2])); i += 3
        else: files.append(a); i += 1
    for p in files:
        t, x, row, dx = read_row(p, var, y, lev)
        w, why = width(x, row, dx, win)
        if w is None: print(f'{p}: t = {t:.6e}: {why}'); continue
        wc, wm, xc, jump, kind = w
        print(f'{p}: t = {t:.6e}  width 10-90 % = {wc:.2f} cells ({wm:.4e} m)  at x = {xc:.5f} m  jump {jump:.4e} ({kind})')
