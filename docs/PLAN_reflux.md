# The coarse–fine volume-fraction correction without a limit

Branch `reflux-comove` off `main` at `51594f3a`. The maintainer's decision of 15 November 2026
on O-17 / [DECIDE-17]: the correction is reformulated so that the clamp to
$[10^{-8}, 1-10^{-8}]$ and the rate cap $|\Delta\alpha_1| \le 0.05$ are not needed. The
predictions of §4 are written before any run.

## 1. What the limit was covering

`PSAMR::reflux` (A-1) co-moves $\alpha_1$ with its refluxed mass at the pre-reflux phase
density, $\Delta\alpha_1 = \Delta m_1/\rho_1^{\mathrm{pre}}$, then clamps the result and caps
the step. Written out, the unclamped result is

$$\alpha_1^{\mathrm{new}} = \frac{m_1^{\mathrm{post}}}{\rho_1^{\mathrm{pre}}},$$

so the rule keeps $\rho_1$ exactly and lets $\rho_2 = m_2^{\mathrm{post}}/(1-\alpha_1^{\mathrm{new}})$
absorb the whole volume mismatch of the correction, whichever phase is the trace. The clamp
fires at the low end only when the refluxed mass of phase 1 is itself at or below the vanish
level, and at the high end when phase 1's volume demand at its old density exceeds the cell:
phase 2 is then squeezed to nothing even when its own refluxed mass is positive. In the 14 Nov
XC2D measurement (VERIFICATION §7.3) the clamp fired about 380 times in 40 coarse steps and three
liquid-dominated coarse cells next to the contact ended at $\alpha_1 = 1$ exactly where the uniform
run keeps a $1.2\times10^{-6}$ vapour trace: the high-end case, the trace folded away with its
mass still present. The rate cap never fired in any recorded run.

## 2. The rule

After the conservative reflux both partial masses are fixed. With
$V_k = m_k^{\mathrm{post}}/\rho_k^{\mathrm{pre}}$ the volume each phase would occupy at its
pre-reflux density,

$$\alpha_1^{\mathrm{new}} = \frac{V_1}{V_1 + V_2}, \qquad
  \rho_k^{\mathrm{new}} = \rho_k^{\mathrm{pre}}\,(V_1 + V_2)\ \text{for both } k.$$

Both phase densities change by the same factor. When the reflux moved volume consistently,
$V_1 + V_2 = 1$, both densities are exactly unchanged and the rule reduces to the present one.
When it did not, the mismatch is shared, so neither phase takes a density spike. For two
positive refluxed masses the result lies strictly inside $(0, 1)$: no clamp, no cap.

Cases, in the order the code tests them:

1. A phase absent before the reflux ($\alpha_1^{\mathrm{pre}}$ exactly 0 or 1): $\alpha_1$ is left
   as it is. Any mass the reflux gave the absent phase sits at $\alpha_k = 0$ and the vanish fold
   that runs on the coarse state right after the reflux folds it back into the present phase,
   conservatively. A reflux does not create a phase; birth belongs to the nucleator. Counted as
   `reflux_sliver` when such mass is non-zero.
2. Both phases present and a pre-reflux phase density not positive: skipped and counted
   (`reflux_badrho`). This is an inadmissible input state, not a limit on the correction; the
   validator reports it.
3. Both phases present, both densities positive: $\alpha_1 = V_1/(V_1+V_2)$ is written as computed.
   If a refluxed mass is zero or negative the result is at or beyond 0 or 1 and the vanish fold
   removes that phase, conservatively and audited; counted as `reflux_absent`. If both masses are
   non-positive the cell's total mass is non-positive; the correction is skipped and counted
   under `reflux_badrho` and the validator reports the cell.

The out-of-range value is handed to the fold rather than set to exactly 0 or 1 here, so that the
model keeps one phase-removal mechanism with one audit (A-2). The fold runs on the coarse state
in `post_timestep` after `avgDown` and the vanish threshold is `ps_presence_vanish`.

What goes: `ps_const::ALPHA_REFLUX_MIN` (its only user was this routine), the cap, the clamp, and
the counters `reflux_cap` and `reflux_clamp`. `ps_bl_reflux` keeps its meaning (nonzero = on).

## 3. The GPU design point

The kernel stays one `ParallelFor` over the coarse valid boxes reading the pre-reflux masses of
both phases; the snapshot grows from one to two components (`UM1RHO1`, `UM2RHO2`). The counters
are host-only accumulators, as the two they replace were, and compile to no-ops on the device.
No new dial, no new constant.

## 4. Predictions (before any run)

1. 1-D battery, PR and GERG, single level: 21/21 fingerprints identical. Falsifier: any change.
2. XC2D on the Mac (llvm, MPI, 40 coarse steps) against the 14 Nov uniform-256 reference, which is
   single-level and untouched by the change: the three coarse cells by the contact keep the
   vapour trace, within a factor of two of the uniform $1.2\times10^{-6}$; inside the refined
   band the agreement stays at $4\times10^{-11}$ in $\alpha_1$ and $3\times10^{-6}$ of the
   maximum in phase energy; at the coarse–fine edge the differences stay within a factor of two
   of 14 Nov either way; `reflux_absent` under 10 over the 40 steps; total mass and energy
   conserved to under $10^{-12}$ relative. Falsifiers: the contact cells still lose the trace
   (then the loss is in the mass and no $\alpha$ rule repairs it), the edge error worse by more
   than two, or an abort.
3. T-Blowdown AMR deck `inputs-x-amr`, old code against new on the host, 300 steps: no abort,
   mass drift at or below the recorded $1.6\times10^{-8}$, density and pressure differences to the
   old run below $10^{-4}$ relative. Falsifier: above $10^{-3}$.
4. GPU (peart, gnu, CUDA): XC2D 40 steps against the Mac host run at the same commit: agreement
   at round-off, below $10^{-9}$ relative in every field. Falsifier: above that, or an abort.
5. Not re-run: the two 2-D pipe-break windows (53 min each); the stage-D record is the control.

## 5. Commits

1. This plan.
2. The rule in `PSAMR::reflux`, the counters, the constant retired; MODEL §1.5 and §4.9 and the
   guard table, DESIGN A-1 and O-17, the XC2D README, VERIFICATION §7.3.
3. The measured record (§6 here and VERIFICATION §7.3).

## 6. Results

(filled after the runs)
