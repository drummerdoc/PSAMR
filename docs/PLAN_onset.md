# A state-dependent nucleation onset: classical nucleation theory in the flash window

Branch `onset` off `main` at `1d7ffef4` (the steady birth step as the default). Port of the Mac line's
onset rule (CAMR `6b02f87`, `6f63c1b`, `40eec9a`; its decision record R-17). Every `[DECIDE-O*]` is the
maintainer's; the predictions of §4 are written before any run on main.

## 1. What is wrong and what this changes

The nucleator's metastability window opens at $(1 - m)\,P_{\mathrm{sat}}(T)$ with a fixed margin
$m$ = `ps_flash_metastable_margin` = 0.1, and the steady birth step holds the liquid at the top of
the window, so the first pressure level after the fast drop is $(1 - m/2)\,P_{\mathrm{sat}} \approx
0.95\,P_{\mathrm{sat}}$ whatever the state. Measured on the ECCSEL depressurizations (Log et al.
2024): about $0.98\,P_{\mathrm{sat}}$ warm (Test 24, the liquid crossing the dome at 298.5 K) and
about $0.91\,P_{\mathrm{sat}}$ cold (Test 19, 276.1 K). The paper's own analysis: the warm tests
nucleate near the homogeneous superheat limit of classical nucleation theory; the cold ones far
above it, heterogeneously at the wall.

This change makes $m$ a function of the liquid's temperature, $m(T)$, from classical nucleation
theory, with the present constant kept as a cap. Nothing else in the nucleator changes: the window
is still a smoothstep of width $m(T)\,P_{\mathrm{sat}}$ opening at $(1 - m(T))\,P_{\mathrm{sat}}$,
the committed fraction is still the implicit solve of `ps_flash_implicit`, the birth state is
still the saturation projection. A vapour-dominant cell (condensation) keeps the constant margin.

## 2. The onset law

Homogeneous nucleation rate per unit volume of liquid at $(T, P)$, $P < P_{\mathrm{sat}}(T)$:

$$J = N_\ell \sqrt{\frac{2\sigma}{\pi m_1}}\;\exp\!\left(-\frac{16\pi\,\sigma^3}{3\,k_B T\,\Delta P^2}\right),
\qquad \Delta P = \big(P_{\mathrm{sat}}(T) - P\big)\Big(1 - \frac{\rho_v}{\rho_\ell}\Big),$$

$N_\ell = \rho_\ell/m_1$ the liquid number density, $m_1$ the molecular mass, $\sigma(T)$ the surface
tension, and $(1 - \rho_v/\rho_\ell)$ the Poynting correction on the pressure difference (the vapour
inside a bubble sits below saturation when the liquid around it does). In words: a bubble of radius
$r$ pays a surface energy $\propto r^2$ and gains a volume energy $\propto r^3\,\Delta P$; the sum has a
hump of height $16\pi\sigma^3/(3\Delta P^2)$ at the critical radius $2\sigma/\Delta P$, and the rate is
a kinetic prefactor times the Boltzmann factor of that hump.

The onset pressure $P_{\mathrm{on}}(T)$ is where $J = J_{\mathrm{crit}} = 10^{12}$ m⁻³s⁻¹ (the
paper's threshold). Taking logarithms gives it in closed form, no iteration:

$$\Delta P_{\mathrm{on}}^2 = \frac{16\pi\,\sigma^3}{3\,k_B T\,\ln(\mathrm{pref}/J_{\mathrm{crit}})},
\qquad \mathrm{pref} = N_\ell\sqrt{2\sigma/\pi m_1},
\qquad m_{\mathrm{CNT}}(T) = \frac{\Delta P_{\mathrm{on}}}{(1 - \rho_v/\rho_\ell)\,P_{\mathrm{sat}}(T)},$$

$$m(T) = \min\big(m_{\mathrm{CNT}}(T),\ m_{\max}\big), \qquad m_{\max} = \texttt{ps\_flash\_metastable\_margin} = 0.1 .$$

The cap is the heterogeneous floor: a real pipe wall nucleates before the homogeneous limit, and
the cold test puts that onset near 9 %. Inputs, none new to the code but $\sigma$:
$P_{\mathrm{sat}}(T)$ from the backend's `Psat_of_T`; $\rho_\ell, \rho_v$ from `co2_sat_state` (the
20-point table in `hem_flash.H`, already device code); $\sigma(T) = 0.07863\,(1 - T/T_c)^{1.254}$
N/m (Mulero, Cachadiña and Parra, J. Phys. Chem. Ref. Data 41 (2012) 043105, stated 1–2 % on CO₂;
zero at and above $T_c$); $m_1$ from `PsPhaseAPI::molar_mass()`; $k_B$, $N_A$ constants.

Values (Span–Wagner auxiliary equations, computed on the Mac line 23 Sept 2026): at 276.1 K
$m_{\mathrm{CNT}}$ = 0.63 (capped to 0.10); 285 K 0.26 (capped); 296.5 K 0.041; 298.5 K 0.023;
300 K 0.014. The cap binds below about 293 K. Sensitivity: $\Delta P_{\mathrm{on}} \propto
\sigma^{3/2}$, and a decade in $J_{\mathrm{crit}}$ moves the onset by a few per cent.

Two guards in the formula, both inert by construction and named here because of the ground rule
on floors: (i) $m \ge 10^{-3}$ keeps the ramp band $m P_{\mathrm{sat}}$ finite; with
$m \propto (1 - T/T_c)^{1.88}$ it is reached only within about 1 K of $T_c$, where the dome window
$w_T$ (5 % of the dome width, 4.4 K) has already closed the flash, so it never changes a result.
(ii) $\ln(\mathrm{pref}/J_{\mathrm{crit}}) \le 0$ would need $\sigma \sim 10^{-57}$ N/m and cannot
occur in double precision; the branch returns the floor and exists so a non-finite input cannot
reach the square root.

## 3. The GPU design point

The margin depends on temperature only, so it could be tabulated once at start-up and copied to the
device. The Sonnet thread's note (12 Nov) put that against "a per-cell solve of 80 fixed passes" and
recommended the table. That described the Mac plan note (§2 of CAMR `6b02f87`, a 40-step bisection),
not the code that landed: `ps_flash_margin_cnt` is the closed form above. Per evaluation: one
`pow` (σ), one lookup in the 20-point saturation table (a 5-step binary search already on the
device), one `log`, two `sqrt`. No loop, no convergence test, no per-backend data, nothing to copy.
It is evaluated once per candidate cell on the entering state and once per evaluation of $W(s)$
inside the implicit solve (at most eight), each next to a full EOS query that costs far more.

`[DECIDE-O-GPU]` Per-cell closed form, as the Mac line has it (recommended and taken, 13 Nov 2026).
A table would add a per-backend artefact to maintain for no measurable saving.

Port differences from the Mac code: the selector is read on the host and carried to the kernels in
`PsKernelDials` (`eos.dials.flash_onset`), never a static inside the kernel; `co2_surface_tension`
and `ps_flash_margin_cnt` carry `AMREX_GPU_HOST_DEVICE`; π is a literal, not `M_PI`. Everything
else is the Mac code line for line.

## 4. Predictions (before any run on main)

Values at `PSAMR.ps_flash_onset = 1`, everything else at defaults (the steady birth step on, the
flattened relaxation solve, mode 6; the mode-5 control alongside as for every comparison with the
Mac line). Contraction-free llvm build on the Mac for the battery; GERG for the pipe tests.

O-P1. Inert at 0: `characterize.py` 21/21 identical to the build before the commit; `ps_cover.py`
8/8. Falsifier: any fingerprint differs.

O-P2. Battery at 1 (PR): B7 alone moves; its liquid enters the dome from 310 K, above the
temperature where the cap stops binding. B2 and B9, whose evaporation fronts are at 256–260 K, are
byte-identical (cap-bound), as are every A and C row and B3–B6, B8, B10–B12. At mode 5 the B7
`exact_suite` row equals the Mac BASE4 row, ρ / u / P rel-L2 0.1603 / 0.7418 / 0.3113, to four
digits, as the earlier stages' rows did; at mode 6 a shift of the same size (about −0.018 in P)
from the step-on row 0.1600 / 0.7469 / 0.3269. `verify_canonical.py` ALL PASS. Falsifier: B2, B9
or any A or C row changes; or B7 at mode 5 differs from BASE4 in the third digit.

O-P3. `ps_cover front` unchanged at both selector values (B2 is cap-bound, so the band check at
$m$ = 0.1 is still the right band).

O-P4. `flashing_front.py` (320 K supercritical liquid crossing the dome near 300 K, $m$ = 0.014):
the pins (ref 2.3428, leg 2.1022, v2e 1.4838, v2s 2.4081) move by less than 2 %, no abort. The Mac
line's moved by +0.2 to +0.5 %.

O-P5. Test 24 (GERG, the Mac, `pipe_depress.py` with the outlet declared an open end): the first
level at 0.8 m after the fast drop is 64.0 ± 0.5 bar, i.e. $(1 - m/2)\,P_{\mathrm{sat}}$ with
$m$ = 0.023 at the 298.5 K crossing; measured 63.8; the fixed window gives 60.0 on the Mac line.
Falsifier: below 62.5 or above 65.5 bar. The traces behind the level (the collapse to 45–50 bar at
2 ms) are not predicted to improve: that is the transfer rate, not the onset.

O-P6. Test 19 (GERG): byte-identical at 1 and 0 (the cap binds at every temperature the cold liquid
reaches); the level stays where the open-end run puts it. This says the cap does the work in the
cold test; it is not a success of the law.

O-P7. GPU (peart, gnu, CUDA): the plume deck has nucleation off (`ps_flash_tau = 0`), so the plume
run proves the build and that the kernel runs. The 2-D blow-down `inputs-sym-x-lo` at 1 and 0, GPU
and host: the on-versus-off row on the GPU equals the host's to a few per cent, and GPU-versus-host
is round-off in both legs, as for the birth step. If the blow-down liquid is cap-bound throughout,
both rows are zero and that is the result.

O-P8. No abort on any run; no new `sstar` refusals (the narrower window narrows the band the
implicit solve works in; its bracket holds by construction).

## 5. Commits

1. `docs/PLAN_onset.md` (this file).
2. `flash: ps_flash_onset (0)`: `co2_surface_tension`, `ps_flash_margin_cnt` in `hem_flash.H`; the
   selector through `PsKernelDials`; both window sites read $m(T)$ at 1; `RUNNING.md` row. Witness
   O-P1; O-P2–O-P4 recorded.
3. `docs: PLAN_onset §6`: the gates (O-P5–O-P8) recorded against their predictions.
4. `[DECIDE-O3]` the maintainer's flip: default 1, 0 kept as the opt-out until retired; MODEL §5.5,
   DESIGN_DECISIONS (next free R number on main), VERIFICATION, RUNNING updated.
5. After the flip: the 1-D set re-recorded against the Mac BASE4; the stored references re-issued
   once (stage D).

## 6. Results

### 6.1 The Mac (llvm, contraction-free), 13 Nov 2026, build at commit 2 on branch `onset`

GERG executable `PSAMR1d.llvm.TPROF.PS.GERG.ex`, `pipe_depress.py` with `PSAMR.ps_bc_open_end_xhi=1`,
`prob.p_amb=6e5` (the Mac line's ambient), `PSAMR.ps_flash_onset` 0 and 1, 640 cells, 6 ms.

| | measured | selector 0 (fixed margin) | selector 1 (CNT onset) | prediction |
|---|---|---|---|---|
| Test 24, level at 0.28 / 0.484 / 0.8 m [bar] | 60.9 / 62.7 / 63.8 | 59.6 / 59.8 / 60.0 | **63.8 / 63.9 / 64.0** | 64.0 ± 0.5 at 0.8 m |
| Test 24, level / $P_{\mathrm{sat}}(T)$ at the crossing (298.7 K) | ≈ 0.98 | 0.92 | 0.979–0.982 | $1 - m/2$, $m$ = 0.023 |
| Test 24, 0.8 m at +2, +3 ms | 61.9, 61.9 | 47.9, 46.9 | 49.5, 47.8 | not improved (transfer rate) |
| Test 19, every sensor and delay | 34.2 at 0.8 m | 35.6 | **35.6, byte-identical** (trace and last plotfile) | identical (cap binds) |

- **O-P1 holds**: `characterize.py` at selector 0 (`onset_off`) against the previous build's record (`c4_flip`,
  the steady-birth-step default): 21 identical, 0 changed.
- **O-P2, fingerprint part, holds**: selector 1 (`onset_on`) against 0: 20 identical, **B7 alone changed**
  (max rel P 2.06e-2, α₁ 7.70e-3, T 1.23e-3; the Mac line measured P 2.2e-2, α₁ 8e-3). B2 and B9 identical
  (cap-bound fronts), every A and C row identical.
- **O-P2, `exact_suite` part, holds**: at selector 1 and mode 5, B7 is ρ / u / P rel-L2 **0.1603 / 0.7418 / 0.3113**,
  the Mac BASE4 row to four digits; at mode 6 (the default) 0.1604 / 0.7416 / 0.3115. Every other B row equals the
  step-on record of 13 Nov (B2 0.0760 / 0.8037 / 0.1485, B9 0.0966 / 0.7749 / 0.1456). `verify_canonical.py` at
  selector 1: ALL CHECKS PASS (32). Transcripts in `~/src/PSAMR-trial/records/2026-11-13_onset/`.
- **O-P3 holds**: `ps_cover.py` 8/8 at the default; `front` unchanged (plateau 20.21 bar at 255.95 K, band
  18.10–20.23, max dP/P 2.7e-12 at `ps_flash_implicit=1`, 3.0e-1 at 0).
- **O-P5 holds** at the predicted number, the same digits as the Mac line's GERG run (63.8 / 63.9 / 64.0).
- **O-P6 holds**: `trace.csv` and `plt02308/Level_0` identical between the two Test 19 runs.
- **O-P8 holds** on both tests (`rc 0`, no abort).
- **O-P4 holds**: `flashing_front.py` on the PR executable, selector 0 → 1: `ref` 2.4038 → 2.4111 (+0.3 %),
  `v2s` 2.4090 → 2.4163 (+0.3 %); no abort. Found on the way: the stored `ref` pin 2.3428 is 2.6 % below the
  selector-0 run of this build, and a control with `ps_flash_implicit=0` (2.4034) shows the birth step is not
  the cause: the drift predates it (the saturation table, the gas constant or stage B). The `v2s` pin 2.4081 is
  current to 0.03 %. Both are re-issued once at stage D, not here.
