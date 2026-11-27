#ifndef PS_COUNTERS_H
#define PS_COUNTERS_H
// =====================================================================
//  PS_counters.H — the single host-side store of every diagnostic
//  counter in the Pelanti-Shyue module.  One plain struct, PsCounters,
//  grouped by section (guards, face audit, promotion, presence, the
//  relaxation sweep, the hem kernels), reached through ps_counters().
//  Counting is done by the thin count_*() wrappers at the kernels (no-ops
//  on GPU builds) or by direct field access from host-only code; the
//  report sites in PSAMR_advance.cpp, PS_relaxation.H and PS_sources.H
//  read the fields, MPI-reduce them per field and reset their section.
//
//  Reset cadence, unchanged by the shared storage:
//    reset_guards / reset_face / reset_promote / reset_presence /
//    fold.reset       : the [PS-MASS] probe (ps_diag_mass) in PSAMR_advance
//    reset_mass       : the [PS-FLOOR] census (PSAMR_PS_DIAG builds)
//    n_gate_presence / pres.reset / x3_stats, mt_stats
//                     : per ps_apply_relaxation sweep
//    reset_mt_cause, mt_coexist.reset, reset_flash_refuse
//                     : per PS_sources report
//    the four *_count fields are cumulative and never reset.
//  reset() clears everything and is not on the time-step path.
//
//  Nothing here touches the numerics: every field is written after the
//  decision it records has been taken.
//
//  PS_COUNT(stmt) is the wrapper for a counting statement inside a
//  per-cell kernel (AMREX_GPU_HOST_DEVICE): it runs on CPU builds and
//  compiles away on GPU builds, where the store is not reachable from
//  device code and the counters read zero.  Multi-statement diagnostics
//  (extremes, re-solves) sit in #if !defined(AMREX_USE_GPU) blocks.
// =====================================================================
#ifdef USE_PS_HYDRO

#if defined(AMREX_USE_GPU)
#define PS_COUNT(...) ((void)0)
#else
#define PS_COUNT(...) do { __VA_ARGS__; } while (0)
#endif

//  Array extents; the enums that index them (PS_HLLC::PsFlCause,
//  hem::PsMtCause, hem::PsFlashRefuse) static_assert against these.
constexpr int PS_COUNTERS_FL_NCAUSE  = 10;
constexpr int PS_COUNTERS_MT_NCAUSE  = 14;
constexpr int PS_COUNTERS_FLR_NCAUSE = 21;

// ---------------------------------------------------------------------
//  Per-sweep thermal-leg instruments, reset and reported with the sweep.
//  [PS-PRES]: the |P1-P2| residual at chain exit, split by whether the
//  thermal leg moved a T by > 1 K, ran and stood, or was gate-skipped;
//  the post-thermal states cost two branch-locked solves, so they are
//  formed only under PSAMR.ps_pres_diag.  [PS-DT]: |T1-T2| at entry and
//  exit of the leg (completeness), fed by X3.
//  Docs/MODEL_AND_ALGORITHM.md §8.6.
// ---------------------------------------------------------------------
struct PsPresDiag {
    // thermal leg ran and moved a T by > 1 K
    long   n_run_moved = 0;
    double dp_moved_max = 0.0, dp_moved_sum = 0.0;
    // thermal leg ran, no T moved by > 1 K
    long   n_run_still = 0;
    double dp_still_max = 0.0, dp_still_sum = 0.0;
    // gate-skipped: residual is post-mechanical, read off q1/q2 (no extra solves)
    long   n_skipped = 0;
    double dp_skip_max = 0.0, dp_skip_sum = 0.0;
    long   n_postinval = 0;   // post-thermal state invalid: residual unmeasurable
    // [PS-DT] |T1-T2| at thermal-leg entry and exit, cells whose leg ran
    long   n_dt = 0;
    double dtpre_max = 0.0,  dtpre_sum = 0.0;
    double dtpost_max = 0.0, dtpost_sum = 0.0;
    void reset() { *this = PsPresDiag{}; }
};

// ---------------------------------------------------------------------
//  Fold-mass audit: every fold event and the phase mass it moved, by
//  cause, so phase-removal churn is visible ([PS-FOLD]).  Read and reset
//  with the [PS-MASS] report in PSAMR_advance.cpp.
//  Docs/MODEL_AND_ALGORITHM.md §4.4.
// ---------------------------------------------------------------------
struct PsFoldAudit {
    long   n_vanish = 0, n_tfloor = 0, n_vacuum = 0;
    long   n_cdeg = 0, n_edeg = 0;    // corridor-density and energy reaps
    double m_vanish = 0.0, m_tfloor = 0.0, m_vacuum = 0.0;   // |phase mass folded|
    double m_cdeg = 0.0, m_edeg = 0.0;
    void reset() { *this = PsFoldAudit{}; }
};

namespace hem {

// ---- Mass-transfer cost/convergence instrumentation (PSAMR.ps_mt_diag) ----
//  Answers whether the equilibrium Newton converges cheaply or burns
//  max_iter on most cells.
//  seen      : cells entering ps_mass_transfer_finite_cell
//  gated     : returned early on the alpha-floor tests
//  eqsolve   : reached the implicit equilibrium solve
//  eqfail    : that solve returned false (no commit)
//  iters     : total Gibbs-Newton iterations across all cells
//  itmax     : worst single-cell iteration count
struct PsMtStats {
    long seen = 0, gated = 0, eqsolve = 0, eqfail = 0, iters = 0, itmax = 0;
    long pr_iters = 0;      // inner pressure-relax Newton iterations
    // One-step path error of the finite-rate transfer as the dimensionless
    // contraction ratio |dm'|/|dm| (a second solve from the post-step state).
    // ratio < 1 = damped ringing; ratio >= 1 = a genuinely non-contracting
    // cell.  Filled only under PSAMR.ps_mt_diag >= 2 (costs a second
    // equilibrium solve per transferring cell).
    long   cr_samples = 0;      // cells where the ratio was formed
    long   cr_nonctr  = 0;      // of those, normalised ratio >= 1
    long   cr_refused = 0;      // re-solve refused: ratio not formed, not zeroed
    long   cr_relaxed = 0;      // frac ~ 1 (expected residual 0): excluded from
                                //   the normalised max, reported separately
    double cr_worst   = 0.0;    // largest raw |dm'|/|dm|
    double cr_worst_n = 0.0;    // largest normalised |dm'|/|dm_eq - dm|
    double cr_worst_r = 0.0;    // fully-relaxed regime: largest |dm'|/|dm_eq|
    void reset() { seen = gated = eqsolve = eqfail = iters = itmax = 0;
                   pr_iters = 0;
                   cr_samples = cr_nonctr = cr_refused = cr_relaxed = 0;
                   cr_worst = 0.0; cr_worst_n = 0.0; cr_worst_r = 0.0; }
};

//  Which side of the coexistence gate closed, and the extremes seen there.
//  Sentinels a real query cannot produce keep "never written" distinguishable
//  from "written with a physical value".
struct PsMtCoexist {
    long n_p1_lo = 0, n_p1_hi = 0, n_p2_lo = 0, n_p2_hi = 0;
    double T1_min =  1.0e300, T1_max = -1.0e300;   // over gate-closing cells
    double T2_min =  1.0e300, T2_max = -1.0e300;
    double a1_at_min = -1.0;                       // alpha_1 at the coldest cell
    //  Gibbs driving force in the refused cells (p1.g, p2.g are already
    //  formed at the gate).  grel compares against tol_g_rel = 1e-4, the
    //  screen these cells would meet next had the gate not closed.
    double gd_min   =  1.0e300, gd_max   = -1.0e300;   // signed g1-g2 [J/kg]
    double grel_min =  1.0e300, grel_max = -1.0e300;   // |g1-g2|/max(|g1|,|g2|,1)
    //  Liquid metastability on the flash's own criterion, P_1 / Psat(T_1):
    //  < 1 is a liquid stretched below saturation (superheated).
    double pr1_min  =  1.0e300, pr1_max  = -1.0e300;
    long   n_pr1    = 0;                               // cells where it was formed
    void reset() { n_p1_lo = n_p1_hi = n_p2_lo = n_p2_hi = 0;
                   T1_min = T2_min =  1.0e300;
                   T1_max = T2_max = -1.0e300;
                   a1_at_min = -1.0;
                   gd_min = grel_min = pr1_min =  1.0e300;
                   gd_max = grel_max = pr1_max = -1.0e300;
                   n_pr1 = 0; }
};

// ---- iso-pressure-relaxation instrumentation ----------------------------
//  Same question as PsMtStats, for ps_iso_pressure_relax_cell (modes 1
//  and 2): does the Newton converge, or fall through to the
//  validity-guarded bracketing scan?  Counted, not reported.
struct PsPrStats {
    long seen = 0, trivial = 0, nt_iters = 0, nt_ok = 0;
    long fb = 0, fb_samples = 0, fb_bisect = 0, fb_ok = 0, refused = 0;
    long barrier_lo = 0, barrier_hi = 0;   // Newton steps clamped to the density barrier (follow-up 17)
    long eos_fail_liq = 0, eos_fail_vap = 0;   // the phase whose state was invalid at an EOS failure (follow-up 21)
    long entry_subtrip_liq = 0, entry_subtrip_vap = 0;   // kernel entry states valid with a temperature below the triple point (follow-up 22)
    void reset() { seen = trivial = nt_iters = nt_ok = 0;
                   fb = fb_samples = fb_bisect = fb_ok = refused = 0;
                   barrier_lo = barrier_hi = 0; eos_fail_liq = eos_fail_vap = 0;
                   entry_subtrip_liq = entry_subtrip_vap = 0; }
};

//  [PS-X3]: the coupled operator's outcome, cause-split, including the
//  P1 = P2 constraint's failures split into entry (structural: the cell
//  cannot be put on the manifold, with the projector's PSR_* cause) and
//  path (expected: the line search probing inadmissible trials).
//  Counting only.
struct PsX3Stats {
    long n_calls = 0;        // cells entering the kernel
    long n_entry_refused = 0;// failed an entry gate or the entry recon
    long n_gate_stood = 0;   // eligibility question said "not mine"
    long n_ok = 0;           // full dt integrated
    long n_newton = 0;       // sub-steps that failed to converge (rc=1)
    long n_dead = 0;         // structurally-dead rate evaluation (rc=2)
    long n_subcap = 0;       // hit the 64-sub-step cap
    long n_tiny_nonconv = 0; // tiny-increment commits whose residual was
                             //   still unmet (> 1 % of the step scale)
    long n_proj_entry = 0;   // entry projection failures
    long n_proj_path = 0;    // path projection failures (line search)
    long n_pr_input = 0, n_pr_eos = 0, n_pr_slope = 0, n_pr_maxiter = 0;
    long n_sub = 0;          // cost: sub-steps attempted
    long n_nit = 0;          // cost: Newton iterations over all sub-steps
    long n_path = 0;         // cost: path evaluations (each one P1=P2 projection)
    //  [PS-IMEX] (PSAMR.ps_imex=3): cells the final sweep handed to the
    //  present operator over the whole step, by the first cause met:
    //  a Newton failure at a stage, a base refusal at a stage (entry
    //  refused, entry projection, dead rates), a stand at a stage, the
    //  final apply itself; and cells whose stage transfers were dropped
    //  because the final's explicit state failed the presence gate.
    long n_imex_fb_newton = 0, n_imex_fb_base = 0, n_imex_fb_stand = 0;
    long n_imex_fb_final = 0, n_imex_dropped = 0;
    void reset() { n_calls = n_entry_refused = n_gate_stood = n_ok = 0;
                   n_newton = n_dead = n_subcap = n_tiny_nonconv = 0;
                   n_proj_entry = n_proj_path = 0;
                   n_pr_input = n_pr_eos = n_pr_slope = n_pr_maxiter = 0;
                   n_sub = n_nit = n_path = 0;
                   n_imex_fb_newton = n_imex_fb_base = n_imex_fb_stand = 0;
                   n_imex_fb_final = n_imex_dropped = 0; }
};

} // namespace hem

// ---------------------------------------------------------------------
//  The store.  Field meanings are documented at the counting sites
//  (PS_guards.H, PS_hllc.H, PS_promote.H, PS_presence.H, PS_relaxation.H,
//  hem_pelanti_shyue.H).
// ---------------------------------------------------------------------
struct PsCounters {
    // ---- guards (PS_guards.H), [PS-GUARD] ----------------------------
    long   guard_seen        = 0;   // phase-pressure guard: calls
    long   guard_reject_low  = 0;   //   substitutions (there is no upper cap)
    long   rho_clamp_lo      = 0;   // G1 density clamp engaged
    long   ctop_seen         = 0;   // ctoprim reference audit
    long   ctop_sub          = 0;
    long   ctop_host_floor   = 0;
    long   flux_sanit        = 0;   // silent limiters
    long   lim_slot_denorm   = 0;   // second-order limiter (O-31): a slot scale below the smallest normal number, not inverted
    long   lim_wave_drop     = 0;   //   a wave whose theta is not finite, taken at first order
    long   lim_face_drop     = 0;   //   a face whose summed correction is not finite, the correction dropped whole
    long   lim_slot_floor    = 0;   //   a slot whose local scale |U_L|+|U_R| is below eps * (its reference scale) and dropped out of the inner product (O-32, docs/PLAN_high_order.pdf Stage 2d)
    long   lim_wave_noise    = 0;   //   a wave every slot of which dropped out: no correction, phi = 0 (O-32)
    long   recon_fallback    = 0;   // reconstructed faces (ps_wp_order=4): a cell-direction returned to its raw averages (presence front in the stencil, or an inadmissible edge state); docs/PLAN_high_order.pdf Stage 3
    long   recon_thinc       = 0;   //   a cell-direction whose alpha_1 edges took the THINC candidate (ps_wp_sharpen=1; follow-up 11)
    long   recon_mc_raw      = 0;   //   a cell away from a presence front whose point value was inadmissible and kept W(<U>) (ps_wp_mc; follow-up 12)
    long   recon_mc_face     = 0;   //   a face whose point side states were inadmissible and kept the stored edges (ps_wp_mc=2)
    long   recon_mc_bdry     = 0;   //   a cell whose conversion stencil leaves the domain across a non-periodic face and kept the average's set (ps_wp_mc_bdry=1; follow-up 16b)
    long   proj_pt_raw       = 0;   // the stage projection on point values (ps_rk_project_pt=1): a cell that took the projection of its average instead (follow-up 13)
    double proj_pt_resid     = 0.0; //   the worst |P1-P2|/P of the converted averages since the last report (ps_pres_diag)
    double proj_pt_dev       = 0.0; //   the worst |P1-P2|/P of the stage point values the kernel receives (ps_pres_diag; follow-up 14)
    double proj_pt_move      = 0.0; //   the worst |delta alpha_1| the kernel applies to them
    long   proj_fail_input   = 0;   // the stage projection's failures by the kernel's reason (follow-up 17): entry
    long   proj_fail_eos     = 0;   //   an iterate's phase state invalid
    long   proj_fail_slope   = 0;   //   no usable dP/dalpha
    long   proj_fail_maxiter = 0;   //   not converged within max_iter
    long   reflux_absent     = 0;   // reflux co-move: a present phase whose refluxed mass is <= 0 (handed to the fold)
    long   reflux_sliver     = 0;   //   mass given to an absent phase (folded back)
    long   reflux_badrho     = 0;   //   inadmissible input (non-positive pre-reflux density or total volume), skipped
    long   reflux_emove      = 0;   // reflux energy co-move (ps_bl_reflux_e): phases whose E_k was carried with the refluxed mass (follow-up 25)
    double reflux_emove_max  = 0.0; //   the largest relative correction |e_k_slot - e_k_pre| / |e_k_pre| it made since the last report
    long   nscbc_zg_c        = 0;   // NSCBC zero-gradient refusals by cause
    long   nscbc_zg_sup      = 0;
    long   nscbc_zg_slots    = 0;
    long   nscbc_zg_pack     = 0;
    long   nscbc_flash       = 0;
    long   nscbc_pmix_fail   = 0;
    long   nscbc_fan_cells   = 0;   // fan visits and their substeps (diagnostic)
    long   nscbc_fan_steps   = 0;
    long   nscbc_br_sup      = 0;   // the branch the fill took per ghost: supersonic copy, small-signal ghost (the fan is nscbc_fan_cells; follow-up 16)
    long   nscbc_br_small    = 0;
    long   nscbc_ghost_nostate = 0; //   ghosts the fill formed that are not a state on their phases' branches (ps_pres_diag)
    double nscbc_gP_min = 1.0e300, nscbc_gP_max = -1.0e300;  //   the first-layer ghosts' pressure and outward normal velocity, extremes since the last report (ps_pres_diag; follow-up 16)
    double nscbc_gu_min = 1.0e300, nscbc_gu_max = -1.0e300;
    long   nscbc_small_inflow = 0;  //   small-signal ghosts selected by the inflow bound, and by the strain cap (follow-up 16)
    long   nscbc_small_cap    = 0;
    double nscbc_cap_ratio_min = 1.0e300, nscbc_cap_ratio_max = -1.0e300;  //   |P_disp - P_N| / cap0 at the dispatch, extremes
    double nscbc_uout_min = 1.0e300;   //   the outward normal velocity at N, minimum
    long   nscbc_locked      = 0;   // outflow cells of an undeclared face held on the
                                    // linear branch as subcooled liquid above an ambient
                                    // below their saturation pressure (a locked face)
    // ---- clean_state mass repairs (PS_guards.H), [PS-FLOOR] ----------
    long   mass_neg          = 0;
    long   mass_nonfinite    = 0;
    double mass_neg_kg       = 0.0; // summed mass created by zeroing

    // ---- face audit (PS_hllc.H), [PS-FACE]/[PS-FLCAUSE]/[PS-RELAXFB] ---
    long   fl_seen[3]        = {0, 0, 0};   // per face class
    long   fl_fail[3]        = {0, 0, 0};
    double max_incmis[3]     = {0.0, 0.0, 0.0};
    int    mis_i[3]          = {-9999, -9999, -9999};   // location of max_incmis
    int    mis_j[3]          = {-9999, -9999, -9999};
    int    mis_k[3]          = {-9999, -9999, -9999};
    int    mis_dir[3]        = {-1, -1, -1};
    long   fl_cause[PS_COUNTERS_FL_NCAUSE] = {0};
    long   fl_side[2]        = {0, 0};
    long   relax_fb          = 0;   // equal-strain (B.14) fallbacks

    // ---- checked promotion / floor legs (PS_promote.H), [PS-PROMOTE] --
    long   promote_refuse    = 0;
    long   floor_skip        = 0;
    long   floor_pre_skip    = 0;   // phases the pre-test table skipped (PLAN_hygiene item 2)
    long   floor_eval        = 0;   // phases the floor evaluated
    long   promote_refuse_sc = 0;   // Independent liquids whose root lies above T_c, counted and kept Independent (follow-up 19; the demotion was refuted)
    long   promote_subtrip_liq = 0, promote_subtrip_vap = 0;   // phases whose root lies below the triple point, counted (follow-up 22; the liquid demotion of 23 reverted)

    // ---- presence (PS_presence.H, PS_relaxation.H), [PS-FOLD] --------
    long   pres_f1_demote    = 0;   // Independent refused for density degeneracy
    long   pres_f3_hyst      = 0;   // gate passes through the hysteresis band
    long   pres_f3_sc_refuse = 0;   // band liquids whose root lies above T_c, counted and admitted (follow-up 18 refused them; reverted, follow-up 20)
    long   pres_f3_subtrip   = 0;   // band liquids whose root lies below the triple point, counted (follow-up 22; the refusal of 23 reverted)

    // ---- relaxation sweep (PS_relaxation.H) --------------------------
    //  One counter for the presence-gate refusal at every relax entry
    //  site ([PS-GATE]): identical predicate on identical inputs, so a
    //  per-mode split would only re-encode ps_relax_mode.
    long        n_gate_presence = 0;
    //  Corridor saturation projection (PS_corridor.H), per sweep.
    //  corridor_probed counts the (Corridor, Independent) cells whose
    //  corridor phase was probed for a root on its own branch (one EOS
    //  inversion each, whatever the answer): the sweep's per-cell EOS cost
    //  that no compaction of the relaxation kernel removes.
    long        corridor_projected = 0, corridor_host_fail = 0, corridor_noroot = 0;
    long        corridor_probed = 0;
    long        corridor_kept = 0, corridor_kept_sc = 0;   // own-branch probes that kept their phase; of those, liquids above T_c (follow-up 17)
    PsPresDiag  pres;
    PsFoldAudit fold;

    // ---- hem kernels (hem_pelanti_shyue.H) ---------------------------
    long   pressure_relax_count = 0;   // cumulative, never reset
    long   mass_transfer_count  = 0;
    long   flash_source_count   = 0;
    long   triple_point_count   = 0;
    long   mt_cause[PS_COUNTERS_MT_NCAUSE]      = {0};   // [PS-MTCAUSE]
    long   flash_refuse[PS_COUNTERS_FLR_NCAUSE] = {0};   // [PS-FLASH-REF]
    hem::PsMtStats   mt_stats;
    hem::PsMtCoexist mt_coexist;
    hem::PsPrStats   pr_stats;
    hem::PsX3Stats   x3_stats;

    // ---- per-section resets (cadence in the header comment) ----------
    void reset_guards() {
        guard_seen = guard_reject_low = rho_clamp_lo = 0;
        ctop_seen = ctop_sub = ctop_host_floor = 0;
        flux_sanit = reflux_absent = reflux_sliver = reflux_badrho = 0;
        reflux_emove = 0; reflux_emove_max = 0.0;
        lim_slot_denorm = lim_wave_drop = lim_face_drop = lim_slot_floor = lim_wave_noise = 0;
        recon_fallback = recon_thinc = recon_mc_raw = recon_mc_face = recon_mc_bdry = 0;
        proj_pt_raw = 0; proj_pt_resid = 0.0; proj_pt_dev = 0.0; proj_pt_move = 0.0;
        proj_fail_input = proj_fail_eos = proj_fail_slope = proj_fail_maxiter = 0;
        pr_stats.barrier_lo = pr_stats.barrier_hi = 0; pr_stats.eos_fail_liq = pr_stats.eos_fail_vap = 0;
        pr_stats.entry_subtrip_liq = pr_stats.entry_subtrip_vap = 0;
        nscbc_zg_c = nscbc_zg_sup = nscbc_zg_slots = nscbc_zg_pack = 0;
        nscbc_flash = nscbc_pmix_fail = 0;
        nscbc_fan_cells = nscbc_fan_steps = nscbc_locked = 0;
        nscbc_br_sup = nscbc_br_small = nscbc_ghost_nostate = 0;
        nscbc_gP_min = nscbc_gu_min = 1.0e300; nscbc_gP_max = nscbc_gu_max = -1.0e300;
        nscbc_small_inflow = nscbc_small_cap = 0; nscbc_cap_ratio_min = nscbc_uout_min = 1.0e300; nscbc_cap_ratio_max = -1.0e300;
    }
    void reset_mass() { mass_neg = mass_nonfinite = 0; mass_neg_kg = 0.0; }
    void reset_face() {
        for (int c = 0; c < 3; ++c) {
            fl_seen[c] = 0; fl_fail[c] = 0; max_incmis[c] = 0.0;
            mis_i[c] = -9999; mis_j[c] = -9999; mis_k[c] = -9999; mis_dir[c] = -1;
        }
        for (int c = 0; c < PS_COUNTERS_FL_NCAUSE; ++c) fl_cause[c] = 0;
        fl_side[0] = 0; fl_side[1] = 0;
        relax_fb = 0;
    }
    void reset_promote() { promote_refuse = 0; floor_skip = 0; floor_pre_skip = 0; floor_eval = 0; promote_refuse_sc = 0; promote_subtrip_liq = promote_subtrip_vap = 0; }
    void reset_presence() { pres_f1_demote = 0; pres_f3_hyst = 0; pres_f3_sc_refuse = 0; pres_f3_subtrip = 0; }
    void reset_mt_cause() { for (int c = 0; c < PS_COUNTERS_MT_NCAUSE; ++c) mt_cause[c] = 0; }
    void reset_flash_refuse() { for (int c = 0; c < PS_COUNTERS_FLR_NCAUSE; ++c) flash_refuse[c] = 0; }
    //  Everything, the cumulative counters included.
    void reset() { *this = PsCounters{}; }
};

//  The one instance (function-local static, so the header stays
//  include-safe).  Host-only: no kernel takes its address.
inline PsCounters& ps_counters()
{
    static PsCounters c;
    return c;
}

#endif  // USE_PS_HYDRO
#endif  // PS_COUNTERS_H
