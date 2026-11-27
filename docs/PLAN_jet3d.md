# The 3-D jet demo at high order

FUTURE_WORK §2 and §0 item 11, taken up 27 November 2026 (Marc's word, after the hygiene pass). The predictions of §3 are written before any run. Ground rules 3, 7, 13 and 15 bind: solution quality, not survival; the second-order run first as the control; every claim against a measurement.

## 1. What exists

The deck `Exec/CO2_PipeBreak/inputs.satjet_demo3_3d`: the pipe-break reservoir and a circular jet of radius 0.05 m on the axis (y, z) = (0.5, 0.5) of a 2 x 1 x 1 m domain, base grid 128 x 64 x 64 (524 288 cells, the 2-D bench's count), three levels to 512 x 256 x 256 at the finest, the x-lo reservoir inlet with its soft lip and the choked-inlet cap, NSCBC outflow on the five other faces, mode 6 relaxation, the flash off (saturated reservoir), the Newtonian shear-layer viscosity, and an inlet fluctuation field (`prob.jet_noise_*`, a seeded lattice hash) for the demo's realism. Nothing in the wave-propagation interior, the fluctuation deposits, the in-cell term of the order-4 path or the NSCBC faces is dimension-specific (z-faces compiled and dispatched; `CFInterp` sweeps z, y, x), but no 3-D two-phase run has ever been made, so the 3-D flux path is unaudited. A 3-D CUDA build existed on peart in October (`PSAMR3d.gnu.MPI.CUDA.PS.PR.ex`, gamma-law era); the Mac build is new.

The instrument: `Exec/CO2_PipeBreak/pf3d.py`, a dimension-generic level-0 reader with the mirror asymmetry across the y and z mid-planes (the odd slots sign-flipped), the difference of two plotfiles and the jet-axis profile.

## 2. The runs

All on the base grid at level 0 (`amr.max_level = 0`), the inlet noise off (`prob.jet_noise_amp = 0`, `prob.jet_noise_tamp = 0`) so the mirror symmetry is a gate, `PSAMR.ps_validate = 1`, `PSAMR.ps_diag_mass = 1`, to `stop_time = 2.5e-3` s or 300 steps, whichever first; plotfiles every 50 steps.

- **J0, the control**: the deck's own second-order scheme (`ps_wp_order = 2`, Godunov), Mac, 4 ranks.
- **J1, the order-4 configuration**: `do_mol=1 rk_order=3 ps_wp_order=4` (the conditional defaults then give the stage projection with Strang and the two-stage sub-step; `cf_interp` is idle at one level), Mac, 4 ranks.
- **J2, the device**: J0 and J1 on peart (DIM=3, CUDA), against the Mac runs.
- **J3, the demo**: after J0 to J2 hold, J1's configuration with `amr.max_level = 1` and the noise on, on the device, plotfiles every 10 steps: the deliverable, with its cost per step and the J0-form run beside it.

## 3. Predictions and falsifiers

1. **Completion.** J0 and J1 run to the end with finite fields and no abort. Falsifier: an abort or a non-finite state in either.
2. **The validator.** Zero violations in J0 and in J1 at every diagnostic point (`[PS-VALIDATE]`). Falsifier: any count above zero (the 2-D demo at order 4 had none).
3. **Mirror symmetry.** Across both the y and the z mid-plane, J1 symmetric to round-off: the worst asymmetry below $10^{-10}$ of each field's range at every plotfile; J0 symmetric to at most $10^{-3}$ of range (the 2-D second-order demo broke its mirror by a thousand pascals where the order-4 run held round-off). Falsifier: J1 above $10^{-10}$, or J0 above $10^{-2}$. The two mid-planes agree with each other (the jet is round): the y and z asymmetries of J1 within a factor of ten of each other.
4. **Folds.** J1's vacuum folds, if any, in the first 20 steps only (the start-up of the rarefaction head, as on the AMR tube), none after step 30; mass repairs zero. Falsifier: folds after step 30 or any mass repair.
5. **Cost.** J1's time per step between 1.2 and 1.6 times J0's on the Mac (2-D measured 1.14 to 1.37; the third face direction and the in-cell term add). Falsifier: above 2.
6. **The jet.** At the final time J1's jet head (the last axis cell with $\alpha_1$ above 0.01) within 2 cells of J0's; J1's axis pressure profile with the steeper gradients (the Mach disk or the first compression narrower by cells, as in 2-D). Falsifier: the heads apart by more than 4 cells.
7. **The device.** J2 against the Mac at the 2-D pipe-break's level: below $10^{-2}$ of range in every field, the differences confined to the orifice region (within 10 cells of the inlet on the axis) at 200 steps, both orders; J1's device run symmetric to round-off like the Mac's. Falsifier: a global difference, or above $10^{-2}$, or a device asymmetry above $10^{-10}$ (that would be the device's own, since the Mac's is round-off).
8. **The demo.** J3 completes at two levels with the validator at zero and the fine level following the jet; cost per step recorded; no prediction on its picture.

## 4. Order and proof

J0, J1 on the Mac; the predictions 1 to 6 read; J2 when peart has the 3-D build; J3 last. Every number into §5 as it comes; a refuted prediction stays on the record as such.
