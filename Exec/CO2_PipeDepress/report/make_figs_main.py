#!/usr/bin/env python3
"""make_figs_main.py: the pipe-depressurisation figures from main's own runs (14 Nov 2026).
Series: PSAMR on main (GERG with the classical-nucleation-theory onset, the default; GERG with
the fixed 10 % window, the former default; PR at the default) from runs/<tag>/trace.csv, plus
the predecessor code line's probe runs (the homogeneous-relaxation source in place of the
nucleator, and the morphology-length scans), whose traces are kept as runs/camr_<tag>/ and are
labelled as such.  Every trace is shifted so that its 40-bar drop at the sensor is t = 0.
Env: OUT (directory for the PDFs; default this directory)."""
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, '..'); OUT = os.environ.get('OUT', H)
S = [0.08, 0.18, 0.28, 0.484, 0.8, 1.599, 3.198]
C = {'meas': '#4a4a4a', 'gerg': '#2a78d6', 'win': '#eb6834', 'hrm': '#1baf7a', 'hrm2': '#eda100', 'D': '#e87ba4', 'pr': '#8e63c4'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e6', 'grid.linewidth': 0.5, 'legend.frameon': False, 'lines.linewidth': 1.4})
def meas(test): return np.loadtxt(os.path.join(R, 'data', 'Exp%s_pt.csv' % test), delimiter=',', skiprows=2)
def run(tag): return np.loadtxt(os.path.join(R, 'runs', tag, 'trace.csv'), delimiter=',', skiprows=1)
def shifted(t, p, drop=40.0):
    a = t[np.argmax(p < p[0] - drop)]; return (t - a) * 1e3, p
def panel(ax, test, sensors, series, tmax=5.0):
    M = meas(test)
    for k in sensors:
        tt, pp = shifted(M[:, 0], M[:, 1 + k]); w = (tt > -0.5) & (tt < tmax)
        ax.plot(tt[w], pp[w], color=C['meas'], lw=2.2, alpha=0.55, label='measured (PT, ±0.6 bar)' if k == sensors[0] else None)
        for tag, lab, col, ls in series:
            T = run(tag); tt, pp = shifted(T[:, 0], T[:, 1 + k]); w = (tt > -0.5) & (tt < tmax)
            ax.plot(tt[w], pp[w], color=col, ls=ls, label=lab if k == sensors[0] else None)
    ax.set_xlim(-0.5, tmax); ax.set_xlabel('time after the wave at the sensor [ms]'); ax.set_ylabel('pressure [bar]')
def fig_test(test, series, fname, title, ylim):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    for ax, k in zip(axs.flat, (0, 2, 4, 5)):
        panel(ax, test, [k], series); ax.set_title('%.2f m from the open end' % S[k], fontsize=9, loc='left'); ax.set_ylim(*ylim)
    for ax in axs[0]: ax.set_xlabel('')
    for ax in axs[:, 1]: ax.set_ylabel('')
    hs, ls = axs[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc='lower center', ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=10, x=0.02, ha='left'); fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(os.path.join(OUT, fname)); plt.close(fig)
fig_test('19', [('on19_o1', 'PSAMR, GERG (identical with the fixed window: cap-bound)', C['gerg'], '-'),
                ('pr19_o1', 'PSAMR, PR', C['pr'], '-'),
                ('camr_k19', r'predecessor line, PR, HRM source, $\theta$ = 2.5 ms (paper fit)', C['hrm'], '-'),
                ('camr_k19_5e-4', r'predecessor line, PR, HRM source, $\theta$ = 0.5 ms', C['hrm2'], '--')],
         'eccsel_test19.pdf', 'Test 19: 124.7 bar, 10.2 °C', (0, 45))
fig_test('24', [('on24_o1', 'PSAMR, GERG, nucleation onset from CNT (default)', C['gerg'], '-'),
                ('on24_o0', 'PSAMR, GERG, fixed 10 % window (former default)', C['win'], '-'),
                ('camr_s24_pr_D2e-3', r'predecessor line, PR, window nucleator, $D$ = 2 mm', C['D'], '-'),
                ('camr_k24', r'predecessor line, PR, HRM source, $\theta$ = 0.14 ms (paper fit)', C['hrm'], '-')],
         'eccsel_test24.pdf', 'Test 24: 115.6 bar, 35.8 °C', (20, 70))
fig_test('23', [('on23_o1', 'PSAMR, GERG, nucleation onset from CNT (default)', C['gerg'], '-'),
                ('on23_o0', 'PSAMR, GERG, fixed 10 % window (former default)', C['win'], '-')],
         'eccsel_test23.pdf', 'Test 23: 121.9 bar, 31.5 °C (held out)', (20, 70))
print('ok')
