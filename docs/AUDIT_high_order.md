# Audit of the high-order branch (18 Nov 2026)

The branch `high-order` against `main` (e9eecf18): 35 files, about 2300 lines of source. Three independent reads (the relaxation kernel and its sweeps; the hydro reconstruction; the time stepping and coarse–fine fills), every claimed bug re-read at the code before it is listed here. Nothing is fixed by this document; each item names the fix. Line numbers are of the branch tip 82767b27.

## Verdict

The numerics are right. Every coefficient, stage time, fill stage index and reflux weight of SSPRK(3,3), SSPRK(4,3) and the IMEX pairing re-derives; the WENO5-Z and PPM formulas, the in-cell term's signs and the stencil reach are correct; every new dial defaults to the previous behaviour by inspection as well as by test; nothing host-only sits in device code. No defect was found that changes a result of any run in the records.

The weakness is volume. The same construction exists twice in four places, three stage loops are one table, a dial is read three ways, and about a third of the new comments carry the plan's history (who decided what, which deck measured it) into the code, where one of them is wrong. Applying the cleanups below removes roughly a third of the new kernel code and half of the new advance code without changing a bit of output, which the existing records can confirm.

## Bugs (none affects a recorded result)

| id | where | what | fix |
|---|---|---|---|
| B1 | `PSAMR.cpp:251`, `PSAMR_advance.cpp:694` | `ps_dial_int("rk_stages")` outside any `USE_PS_HYDRO` guard; `PS_dials.H` is empty without it, so a build without `USE_PS_HYDRO` (which `read_params` still caters for) no longer compiles. Every Exec sets it, so unexercised. The dial is also read three ways (two helper reads and a raw query), against the helper's one-read-site contract. | make `rk_stages` a PSAMR parameter next to `rk_order`, validate it in `read_params`, use the member. |
| B2 | `PSAMR.cpp:1921-1935`, `2004-2015` | the `FillPatch` fallback of `rk_fill`/`rk4_fill` at a half-stage time fills fine–fine ghosts by linear interpolation between old and new data, not with the stage state. Unreachable today (the coarse store always precedes the fine fill). The `static int said` is shared by all levels. | abort as `AmrLevel::FillRKPatch` does, or fill with `FillPatchTwoLevels` as `rk4_fill` does; drop the static. |
| B3 | `PSAMR.cpp` `rk_fill`, `rk4_fill` | stage ghost cells skip `clean_state` (the alpha clamp, mass repair, density floor, species normalisation `expand_state` applies); with the unlimited `quartic` interpolater overshoots in coarse–fine ghosts are real. Whether the ctoprim guards absorb it is not verified. | a decision: `clean_state(S, S.nGrow(), false)` at the end of both fills changes `rk_order=3` results and needs the Stage records re-run. |
| B4 | `PSAMR_advance.cpp:160` | `ps_order4_default` keys on `ps_wp_order == 4` alone; a deck with `ps_wp_order=4` and the default `rk_order=2` gets the Strang pre-step and a `ps_rk_project=1` nothing reads. | key on `rk_order == 3` as well, or require `rk_order=3` under `ps_wp_order=4` in `PS_umeth.cpp`. |
| B5 | `PSAMR_advance.cpp:206-222` | `ps_imex=3` is accepted with `ps_hydro=0` and silently ignored. | add `ps_hydro == 0` to the abort. |
| B6 | `hem_relax_x3f.H:545` | the IMEX predictor `q_off + dt rT0/(1 + dt/θ)` is not the SDIRK stage-2 form `(q_off + dt rT0)/(1 + dt/θ)` of line 615, which is the one consistent with the linearised residual; only an initial guess (the converged state is unaffected), and the comment "as step_once forms them" is wrong. | use the stage-2 form; fix the comment. |
| B7 | `hem_relax_x3f.H:393-402` | the inadmissible-start loop halves the total transfer toward zero, which with a nonzero offset is not "toward the base" as the comment says. | halve toward the offset (`z = q_off + (z - q_off)/2`, bitwise the old behaviour at zero offset) and say so. |

## Duplication

| id | where | what | fix |
|---|---|---|---|
| C1 | `hem_relax_x3f.H:258-267, 536-544, 574-584` | the base-rate expressions (rT, rM) three times. | one `rates(...)` lambda with the finiteness checks. |
| C2 | `hem_relax_x3f.H:547-554, 590-594, 602-607, 617-621` | "Newton over dtl, base_from, admissibility" four times. | one `solve(bb, dtl, z, t, nit, out)` lambda. |
| C3 | `PS_relax.H:531-546, 759-774` | the host status switch twice (the second drops `X3_SUBCAP`). | `ps_x3_count_status(status, nsub, nit)` under the host guard. |
| C4 | `PS_relax.H:820-826, 951-956` | the per-sweep counter resets twice. | `ps_relax_reset_sweep_counters()`. |
| C5 | `PS_relax.H:184, 218`, `hem_relax_x3.H:29` | the literal `1.0e-4` projection tolerance in three places; the "unchanged default" depends on them agreeing. | one default from the params struct. |
| C6 | `PS_recon.H:163-205` vs `PS_state.H:88-125` | `prim_from_cons` repeats the first half of `ps_cell_state` line for line. | split `ps_cell_state` into the pre-EOS `ps_cell_prims` and the EOS stage; `prim_from_cons` packs the former. |
| C7 | `PS_recon.H:255-261` vs `PS_umeth.cpp:962-968` | the contact-keep weight twice, with a dead `t < 0` clamp. | one function; drop the clamp. |
| C8 | `PS_umeth.cpp:338-373, 896-936` | `ps_wp_face` and `ps_wp_face_recon` differ only in the source array, and the three `ParallelFor`s are pasted under `if (o4)/else`. | one face function taking the source `Array4` and component offsets; one loop set. |
| C9 | `PS_umeth.cpp:208-218, 300-310` | the momentum-component ladder twice in one function, and the core re-derives the alpha clamp, normal velocity and density floor that `ps_cell_state` computed inside the flux call; the `pmix_out` out-pointer. | one `ps_cell_state` per side in the core, handed to the flux and the HLLC face; delete the out-pointer and the second ladder. |
| C10 | `PSAMR_advance.cpp:710-735, 861-882` | the `rhs` and `fold` lambdas pasted into both stage branches. | hoist above the branch. |
| C11 | `PSAMR_advance.cpp:791-818, 822-844, 884-904` | three hand-unrolled stage sequences of the same five lines per stage. | one table-driven loop over (a_s, b_s, c_s, t_s) with a fill hook and a post hook; the operation order kept so the output stays bitwise. About 130 lines become 25. |
| C12 | `PSAMR.cpp` `rk_fill`/`rk4_fill`, `rk_store_coarse`/`rk4_store_coarse` | two names for one job each, the level-0 branch and the fallback duplicated. | one of each dispatching on `rk_stages`. |
| C13 | `PSAMR.H:426`, `PSAMR_advance.cpp:711, 862` | `rk_reflux_fac`, a member used as a hidden argument with a −1 sentinel and a set/reset pair per call. | an argument of `construct_hydro_source`, default −1. |

## Comments

| id | where | what | fix |
|---|---|---|---|
| D1 | `PS_umeth.cpp:145` | "wpf comps 6..15": the recon block writes 6..17. | 6..17, list 16/17. |
| D2 | `PS_recon.H:36-40, 92`, `PS_umeth.cpp:1225` | a fallback cell "takes its raw averages": it takes their prim→cons round trip (absent-phase slots exactly zero). | say so. |
| D3 | `Source/Params/_cpp_parameters` | `rk_order` "ignored under do_mol=0": `read_params` aborts on 3 with `do_mol=0`. | fix the text. |
| D4 | `PSAMR.H:423-425` | `rk_reflux_fac` "(1/6, 1/6, 2/3)": under `rk_stages=4` it carries 1/6, 1/6, 1/6, 1/2. | "the stage's weight". |
| D5 | `hem_relax_x3f.H:521-529, 608-612`; `PS_relax.H:703-714`; `PSAMR_advance.cpp:125-222, 738-763`; `PS_recon.H:1-50`; `PS_umeth.cpp:1225-1259` | the IMEX block comment restates the struct comment; the stage-1 note restates the header; the dial block runs 100 lines before the first statement; the IMEX stepping comment is half history (the Butcher form, the XC2D numbers); the recon header restates the taper dial and Marc's decision; the Pass 3 comment carries the B12 anecdote. | keep what the code does and the sign conventions; the history, the decisions and the evidence live in the plan and DESIGN. |
| D6 | `PS_relax.H:706-710, 795-799` | `st` described as "bits of the first cause" and tested with `&`, but assigned once; holds one code. | "the code of the first cause", test with `==`. |
| D7 | `hem_relax_x3f.H:227-228`; `PS_relax.H:605-619`; `CFInterp.cpp:936`; `PSAMR_advance.cpp:125-132` | stale: the offsets are not "the second SDIRK stage's" only; "this sweep" is five sweeps under IMEX; "Stage 1" with no stage 2; the `ps_strang` comment omits the `ps_imex` override. | one line each. |

## Dead code and factorisation

| id | where | what | fix |
|---|---|---|---|
| F1 | `hem_relax_x3f.H:537, 544, 551, 603` | three dead `q_off = dm_off = 0` resets before returns, one redundant. | keep the one after stage 2. |
| F2 | `PS_relax.H:28`, `PS_recon.H:62` | unused includes (`AMReX_GpuContainers.H`, `PS_guards.H`). | remove. |
| F3 | `CFInterp.cpp:963-974` | a counter incremented then `ignore_unused`. | delete. |
| F4 | `PSAMR_advance.cpp:80-81` | `ignore_unused(amr_iteration, amr_ncycle)` though both are used. | delete. |
| F5 | `PSAMR_advance.cpp:793` | `stage(S_new, 1, c2)` passes stage 2's offsets to stage 1 (ignored); stage 1 runs a whole-grid fold, resync and projection even when no relaxation can produce a transfer. | a zero array; run stage 1 only under mode 6 with relaxation on. |
| F6 | `PSAMR_advance.cpp:736-821` | an `else` dangling across `#ifdef USE_PS_HYDRO`; the advance is about 950 lines. | a `const bool imex` under the ifdef, and the stage loop as its own member. |
| F7 | `PSAMR.H:175-178` | four public `rk4_*` members in a class whose data are protected. | one protected struct. |
| F8 | `PS_recon.H:86, 118, 274` | unused `wpow = 1` default arguments. | drop. |
| F9 | `PS_relax.H:725` vs `PSAMR_advance.cpp:779` | the stage transfer slot is zeroed per stage and by `setVal` per advance. | keep the per-stage zero (the self-contained one). |
| F10 | `PS_relax.H:716-776` | a stage cell does not pass `exit_code`, so `ps_coexist_action=1` aborts only in the fallback. | pass it, or document. |
| F11 | `PS_relax.H:804-807` | the fallback runs the whole-step operator from the explicit final state, which already carries 2/3 of stage 4's transfers, so those cells see part of the transfer twice; the plan calls the fallback first order and this is its price. | one sentence at the call. |

## Efficiency

| id | where | what | fix |
|---|---|---|---|
| E1 | `PSAMR_advance.cpp:683-686` | under `rk_order=3` the old-time sources are built and added to `S_new`, which every stage branch then overwrites; `rhs(1)` rebuilds them. | gate on `!(do_mol && rk_order == 3)`. |
| E2 | `PSAMR_advance.cpp:778-781, 704-707` | `imex_D`, `imex_st` and `rkk` defined (and the first zero-filled) on every advance of every level while `hydro_source`/`new_hydro_source` sit unused under `rk_order=3`. | persistent members on the `flash_src` pattern; only `imex_st` needs the zero. |
| E3 | `PSAMR.cpp:1981-1986, 2026-2038` | `rk4_store_coarse` copies five whole-level fabs into the fine level each coarse step; `rk4_fill` allocates a coarse scratch and copies the fine state per stage, which `FillPatchTwoLevels` does not need. | move the k's; a persistent scratch; `fmf{&S}`. |
| E4 | `PS_umeth.cpp:1003-1013` | `rho_bar`, `s_c`, `c_bar`, `vref` are formed inside the three-wave loop and even when `lim_ref_eps == 0`. | hoist. |
| E5 | `PS_umeth.cpp:289-296` | four face slots store `alpha_k P` products two scalars (`PL`, `PR`) would give; `pc` 18 against 16. | store the pressures, form the products in `incell` (bitwise). |
| E6 | `PS_recon.H:276-284` | each cell's primitive set is rebuilt five times per direction. | one pass converting `grow(bx,3)` to primitives, read by the stencil. |
| E7 | `PS_umeth.cpp`, `PS_hllc.H:478` | under order 4 each face solves the EOS state four times (two in the flux, two in the HLLC face); a reconstructed edge is unique to its face so the cache cannot help. | C9 halves it: one `ps_cell_state` per side handed to both. Not measured. |
| E8 | `PS_umeth.cpp:1261-1290` | the Pass 3 kernel carries the Kapila block and the `incell` lambda under a runtime `if (o4)` in every build, growing the default kernel's GPU register footprint. | split the kernel on `o4` (a template bool). |
| E9 | `PS_relax.H:716-776` | a cell already marked for the fallback still runs the full Newton at every later stage. | a design call: the projection only for marked cells changes the stage state the hydro sees. |
| E10 | `hem_relax_x3f.H:615-616` | the SDIRK stage-2 predictor uses the base rates where stage 1's converged transfer is a free better guess. | untested. |

## Applied (18 Nov 2026, commits 328dcbbd, 194910d8, 35df4cc2)

Three batches, each built and rerun on a reference set of 22 runs on the binaries of 82767b27 (B5 at eight settings: the order-2 Strang step with and without the step-doubling control, the pairing, the four-stage integrator, Heun, mode 2, Lie without the projection, the order-1 sub-step; the tube bare, order 4, the pairing, four stages; XC2D on four ranks at the same four; the fine box at order 4, four stages and the unlimited interpolater; the 21-case fingerprint) and compared file by file. Every run is bitwise except the two runs of the pairing with relaxation, which move at 4e-11 of range on the tube and 2e-8 in pressure on B5 after batch A and not again after B or C: that is B6, the predictor's initial guess, converging to the same solution within the Newton tolerance.

- Batch A (the kernel and its sweeps): B6, C1, C2, C3, C4, C5, D5, D6, D7, F1, F2, F9, F10, F11. B7's comment fixed, its arithmetic kept (the shrink toward the offset is untested).
- Batch B (the advance and the fills): B1, B2 (an abort), B4, B5, C10, C11, C12, C13, D3, D4, D5, D7, E1, E2, E3 (the coarse old data read in place, the scratch persistent, the fine copy gone; the derivatives still copied once per coarse step), F3, F4, F5, F6, F7.
- Batch C (the hydro): C6, C7, C8, C9 (the flux side; the HLLC face still forms its own state, E7's other half), D1, D2, D5, E4, E5, F2, F8, and the edge stores allocated under order 4 only.
- Not applied, as listed: B3 (a decision that changes `rk_order=3` results), E6, E8, E9, E10 (measurements or design calls), the HLLC half of E7.

The new source code of the branch is shorter by about 500 lines after the three batches, with no output changed.

## Order of work, as it was taken

1. B1, B5, D1, D3, D4, F2, F3, F4 and the comment lines of D5–D7: no output change, an afternoon.
2. C1–C5 and F1 (the kernel and sweeps), then C10, C11, C13 and E1, E2 (the advance): each proven bitwise on the existing records (B5 cfl 0.25, the tube at 300 steps, the fingerprint) before the next.
3. C6–C9, E4, E5 (the hydro): bitwise by construction, proven the same way.
4. B2, B4 and E3 (the fills): B4 changes a deck's behaviour only where it was silently wrong; B2 is unreachable; E3 is a measured saving.
5. B3, E6–E10: decisions or measurements, each its own item.
