# The limiter's denormal slot and the silent face-flux zeroing

Branch `limiter-denormal` off `main` at `980117c2`. The maintainer's decision of 15 November
2026 on O-31 / [DECIDE-35], taken before this plan was written: the slot normalisation stops
inverting a scale that is not a normal number, and a correction that is still not finite is
dropped at its face and counted instead of zeroing the face flux. No other change. The
predictions of §4 are written before any run.

## 1. The defect

`PS_umeth.cpp`, the `correct` lambda (one scalar van Leer limiter per wave, MODEL §3.3). Each
slot $n$ of the two waves is normalised by $1/sc$ with $sc = |U_{L,n}| + |U_{R,n}|$, guarded
only against $sc = 0$. For a denormal $sc$ (below $2.2\times10^{-308}$) the reciprocal is
infinite, both inner products are NaN, $\theta$ and $\phi$ are NaN, every $\tilde F[n]$ is NaN,
and `flx = ps_finite_or(flx + Ft, 0)` replaces the whole face flux, the first-order part
included, by zero. In liquid at 10 MPa the missing pressure flux kicks each neighbour by
$p\,\Delta t/\Delta x$ (4587 kg m⁻² s⁻¹ at the XC2D coarse step 40) with density and total
energy unchanged. Nothing counts it: `flux_sanit` only counts the sanitiser in `ps_wp_face`
(`PS_umeth.cpp:118`), and the correction path's `ps_finite_or` has no counter.

Seen on the Mac XC2D run (VERIFICATION §7.3, DESIGN O-31): round-off noise of 1e-11 in
momentum walks down the $x$-low inflow wall from step 36, the x-momentum of cell (0,91) is the
denormal 2.4e-312 at step 39, and three level-0 cells, (0,90), (0,91), (1,91), are kicked at
step 40. The GPU run on peart has the quiet liquid exactly at rest and never reaches the path.

Why the GPU run does not show it, measured on 15 Nov 2026 before any code change, from the
step-by-step plotfiles of the Mac run and one contraction-free Mac build (`-ffp-contract=off`,
`TMP_BUILD_DIR=tmp_nc`, `USERSuffix=.NC`, run with the same command):

- The physical rarefaction from the diagonal reaches the wall column with a numerical precursor
  that decays by about 130 per cell ahead of the front (at step 40, j = 108 to 104: 2.9e-5,
  1.8e-7, 1.4e-9, 3e-11, 1e-12 in x-momentum). Both builds and the GPU agree on that precursor.
- Below the precursor the Mac default build carries a flat band of 1e-12 to 5e-11 in x-momentum
  over ten cells, one to fifty ulps of the per-step pressure flux $p\,\Delta t/\Delta x$, with
  density and total energy bit-identical to the quiet liquid in most of those cells. The band's
  leading cell advances exactly one cell per step (j = 100 at step 30, 91 at step 39) and its
  value is a product of small quantities (−1.3e-27 at (0,91) at step 40; 2.4e-312 at step 39),
  because the x-momentum a quiet cell first receives through a y-face is $\rho\,v\,u$, the product
  of two velocities that are themselves at round-off. The GPU file has no band: x-momentum is
  non-zero only from j = 104 up at step 40 and exactly zero below.
- The contraction-free Mac build has no band either: non-zero only from j = 104 up, no denormal
  at any step 30 to 40, no kick, and its step-40 plotfile agrees with the GPU file at round-off
  in every field on both levels (momentum 1.4e-13 of range on level 0, 3.9e-13 on level 1).
  Against the Mac default build it differs by the kick alone (0.63 of range in x-momentum).

So the difference between the two machines is floating-point contraction: with clang's default
contraction on arm64 a flux difference at a face whose two states differ below one ulp leaves
an ulp-sized residue that re-seeds the next cell every step, and the chain of products at its
leading edge runs down into the denormals; without contraction, and in the CUDA build, the same
expressions cancel exactly and the sub-ulp leak dies out. The noise itself is harmless (1e-11
in a field of 1e4) and is not what the fix is for. The defect is that the limiter inverts a
denormal when one arrives, and a denormal slot can arise in any build from a product of small
velocities; the CUDA build's immunity in this run is the compiler's cancellation, not a property
of the scheme.

## 2. The rule (the decision, restated)

Decision 1, the normalisation. A slot scale below the smallest normal number,
`std::numeric_limits<Real>::min()`, is treated as the existing $sc = 0$ case: the slot is
scaled by $|W_f|$ if $|W_f|$ is itself a normal number, otherwise the slot is skipped as
carrying no information. The threshold is a property of the arithmetic, not a model number;
it is the smallest value whose reciprocal is finite. Nothing changes for a normal $sc$. The
denormal case is counted, `lim_slot_denorm` (the maintainer's choice of 15 Nov, so that the
audit sees denormal slots arriving); the exact-zero case stays uncounted, since it is the norm
for an absent phase's slots.

Decision 2, the fallback, in two parts.
(a) Per wave: if $\theta$ is not finite that wave gets $\phi = 0$, so it enters the correction
at first order; counted `lim_wave_drop`.
(b) Per face, the backstop that replaces the silent zeroing: if any component of the summed
correction $\tilde F$ is not finite, the whole correction at that face is dropped
($\tilde F[n] = 0$ for every $n$) and the first-order flux is left as it is; counted
`lim_face_drop`. Dropping every component together keeps the wave-direction property the
scalar limiter protects. The `ps_finite_or` wrappers on `flx + Ft` and on the three stashed
slots go: after (b) the correction is finite by construction and the first-order flux was
sanitised on entry, so the only thing they could still hide is what (b) now counts.

In code, the three places:

1. In the slot loop, `sc > 0` becomes `sc >= numeric_limits<Real>::min()`; a positive `sc`
   below that calls `ps_guard::count_lim_slot_denorm()` and falls through to the phase-birth
   branch, whose test `|Wf| > 0` becomes `|Wf| >= numeric_limits<Real>::min()`; otherwise
   `continue` as now.
2. After `theta = wdotu / wdotw`: if `theta` is not finite, `phi = 0` and
   `ps_guard::count_lim_wave_drop()`; otherwise the van Leer formula as now.
3. After the sum assignments `Ft[URHO]`, `Ft[UEDEN]`: if any `Ft[n]` is not finite, set every
   `Ft[n]` to 0 and `ps_guard::count_lim_face_drop()`; then `flx += Ft[n]` and
   `ft = Ft[...]` without `ps_finite_or`.

Three counters follow the `flux_sanit` pattern: fields in `PsCounters` (`PS_counters.H`), reset
in `reset_guards`, `count_*` functions in `PS_guards.H` that are no-ops on a GPU build,
reported on the `[PS-GUARD]` line in `PSAMR_advance.cpp` beside `flux_sanit`.

Rejected, for the record: dividing by $sc$ instead of multiplying by its reciprocal (moves the
battery at round-off for no gain); normalising by $\max(sc, |W_f|)$ everywhere (a model change
to $\phi$ wherever a wave's slot exceeds the slot sum); a relative noise floor on the slot
(a threshold needing a derivation nobody has asked for).

## 3. The GPU design point

The kernel stays as it is: one `ParallelFor` per direction, the same loops, two extra
comparisons per slot and one `isfinite` per wave and per face. `std::numeric_limits<Real>::min()`
is a compile-time constant and is already used in device code (`hem_flash.H`); `std::isfinite`
is already used in the same translation unit on the device. The counters are host-only
accumulators and compile to no-ops on the device, as `flux_sanit` does. No new dial, no new
constant, no new state.

## 4. Predictions (before any run)

All comparisons with `~/src/PSAMR-trial/bench_nov14/pfdiff.py` (per level, max |difference| /
range per field). The control binaries are the current Mac builds of 15 Nov (the code has not
changed between `5f098da6` and `980117c2`), copied aside before the rebuild. The `[PS-GUARD]`
line prints at the `PSAMR.ps_diag_mass` cadence, so the XC2D and T-Blowdown runs, new and
control alike, carry `PSAMR.ps_diag_mass=1`.

1. 1-D battery, PR executable (`characterize.py`, `EXE=` named explicitly because the harness
   picks the newest `PSAMR1d.*.ex` by file time): 21/21 fingerprints identical to
   `characterization/reflux_comove.json`; all three counters 0.
   Falsifier: any change in any fingerprint. (A denormal slot in the battery would show as a
   change, and would itself be a finding.)
2. XC2D on the Mac (llvm, 4 ranks, `amr.max_level=1 max_step=40 amr.plot_int=1`, 2 s), new
   binary against the control binary, both from the same command:
   - plotfiles through step 39 bitwise identical (the first denormal slot sum is at step 39,
     the kick lands at step 40);
   - at step 40 the two runs differ only in momentum, and only in cells that touch the three
     faces seeing (0,91), on level 0;
   - at step 40 the new run against the GPU reference `Exec/CO2_XC2D/plt_gpu_00040`: level-0
     momentum below 1e-9 of range (0.385 before the fix), the three cells included; every other
     field and level 1 as before (1e-14 and 5.4e-13 of range);
   - `flux_sanit`, `lim_wave_drop` and `lim_face_drop` 0 throughout; `lim_slot_denorm` 0 through
     step 39 and at least 1 during step 40 (the x-momentum slot at the faces that see (0,91));
     in the contraction-free build all four 0 throughout. With decision 1 in place the denormal
     slot is skipped, so neither fallback should fire; a non-zero `lim_wave_drop` or
     `lim_face_drop` is a finding to record, not a failure of the fix.
   Falsifiers: momentum still kicked at step 40 (then the mechanism reading is wrong); any
   difference before step 40 (then decision 1 changes a normal-slot result, which it must not).
3. T-Blowdown `Exec/CO2_TBlowdown`, `inputs-x-amr`, 300 steps, new binary against the control
   binary: bitwise identical final plotfile; all three counters 0. The two go together: a
   denormal slot in the control run was inverted and its face flux zeroed silently, so a
   non-zero `lim_slot_denorm` comes with a difference at the faces counted (the fix acting on
   a kick the control took silently), which would be recorded as such. Falsifier: a difference
   with `lim_slot_denorm` at 0, or a fallback counter above 0.
4. GPU (peart, gnu, CUDA 12.0): bundle over, fetch, rebuild in `Exec/CO2_XC2D` at the new
   commit, 40 steps, the same command line as the 15 Nov GPU check. Against the new Mac run:
   round-off in every field, momentum included (below 1e-12 of range on level 0, below 1e-12
   on level 1). Against `plt_gpu_00040`: bitwise identical (the GPU never entered the changed
   branches, and the arithmetic on a normal slot is the same expression as before).
   Falsifiers: any difference from `plt_gpu_00040`; momentum against the Mac above 1e-9.
5. Not re-run: the two pipe-break windows and the 3-D jet. The hydro change is inert wherever
   no slot is denormal, which predictions 1 to 4 test directly.

## 5. Commits

1. This plan, and the renumbering of the 15 Nov entry O-28 [DECIDE-35] to O-31 in
   `DESIGN_DECISIONS.md`, `PLAN_reflux.md` §6 and `VERIFICATION.md` §7.3 (the number O-28 was
   already taken by the table-backend item).
2. The code (`PS_umeth.cpp`, `PS_guards.H`, `PS_counters.H`, `PSAMR_advance.cpp`); MODEL §3.3
   (the paragraph after the $\langle a,b\rangle_D$ formula) and the guard table row and the
   `[PS-GUARD]` counter list in §8.6; DESIGN O-31 from Decision to LANDED; VERIFICATION §7.3.
3. The measured record (§6 here, VERIFICATION §7.3, DESIGN O-31), after the GPU check.

## 6. Results

To be written after the runs. The question the handover left open, why the Mac run carries
the 1e-11 wall noise while the GPU run is exactly 0 there, is answered in §1: contraction.
The contraction-free Mac run (plotfiles `plt_nc_*`, binary `PSAMR2d.llvm.TPROF.MPI.NC.ex`)
is a second control for prediction 2: with the fix in place the default Mac build must still
differ from it only at round-off, and the kick must be gone.
