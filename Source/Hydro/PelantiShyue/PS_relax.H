#ifndef PS_RELAX_H
#define PS_RELAX_H
// =====================================================================
//  PS_relax.H — the relaxation dispatch of the six-equation Pelanti–Shyue
//  model on the PSAMR grid: the EOS callback (ps_make_psamr_eos_api), the
//  presence relax gate and entry gate, the per-mode cell wrappers over
//  the hem kernels (modes 0, 2 and the default coupled X3, mode 5),
//  the relaxation dials, and ps_apply_relaxation with its sweep reports
//  ([PS-PRES], [PS-GATE], [PS-DT], [PS-X3]).
//  Algorithm: docs/MODEL_AND_ALGORITHM.md §5.  Included through the
//  PS_relaxation.H umbrella; PS_folds.H and PS_diag.H build on it.
// =====================================================================

#ifdef USE_PS_HYDRO

#include "IndexDefines.H"
#include "EOS.H"
#include "hem_pelanti_shyue.H"
#include "PS_presence.H"   // PsPres, ps_regime, the fold thresholds
#include "PS_promote.H"    // ps_regime_reach (checked promotion)
#include "PS_constants.H"  // ps_const:: shared numeric floors
#include "PS_util.H"       // ps_kinetic_energy, ps_pack_v6 / ps_unpack_v6
#include "PS_counters.H"   // PsCounters (fold audit, floor and sweep counters)
#include "PS_dials.H"      // ps_dial_int / ps_dial_real (PSAMR.* reads)
#include "PS_classes.H"    // cell classes, compact per-box launches (ps_class_compact)
#include "PS_state.H"      // PsCellState, ps_cell_prims / ps_cell_state (the point-value projection)
#include <AMReX_Array4.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Print.H>
#include <AMReX_BLProfiler.H>
#include <cmath>
#include <algorithm>   // std::max / std::min
#include <string>

// ---------------------------------------------------------------------
//  Per-sweep counters (PsPresDiag, n_gate_presence: PS_counters.H),
//  stored in PsCounters; reset at the start of each ps_apply_relaxation
//  sweep and reported at its end.
// ---------------------------------------------------------------------
inline PsPresDiag&  ps_pres_diag_data() { return ps_counters().pres; }

// PSAMR.ps_pres_diag: see hem::ps_diag_pres (the single read site).
inline int ps_pres_diag() {
    return hem::ps_diag_pres();
}

// -----------------------------------------------------------------
//  ps_make_psamr_eos_api — the PsPhaseAPI value every hem kernel here
//  uses.  Fluid metadata (critical and triple points, density domain,
//  molar mass) comes from the EOS, so no kernel carries a fluid
//  constant; the branch-locked phase query is hem::ps_eos_phase_state
//  (docs/MODEL_AND_ALGORITHM.md §7.2); the kernel dials are resolved here
//  (hem::ps_kernel_dials).  Host-only; the result is captured by copy.
// -----------------------------------------------------------------
inline hem::PsPhaseAPI
ps_make_psamr_eos_api()
{
    hem::PsPhaseAPI api;
    //  The EOS queries themselves are PsPhaseAPI member functions
    //  (hem::ps_eos_phase_state, EOS::Psat): device-callable, so the API is
    //  a plain value the per-cell kernels capture by copy.
    api.has_eos  = true;
    api.has_psat = true;
    api.dials    = hem::ps_kernel_dials();
    api.T_crit_K   = double(EOS::T_crit());
    api.P_crit_Pa  = double(EOS::P_crit());
    api.T_triple_K = double(EOS::T_triple());
    api.P_triple_Pa = double(EOS::P_triple());
    // The backend density domain bounds the mechanical kernels' alpha walk
    // so neither phase can be stepped off it (docs §5.2).
    api.rho_dom_lo = double(EOS::rho_min());
    api.rho_dom_hi = double(EOS::rho_max());
    // Molar mass for the SRT kinetic prefactor.
    {
        amrex::Real Yw[NUM_SPECIES];
        ps_pure_species(Yw);
        amrex::Real wbar = amrex::Real(0.0);
        EOS::Y2WBAR(Yw, wbar);            // [kg/mol] in every PSAMR backend
        api.M_kg_mol = double(wbar);
        api.R_gas_J_molK = double(hem::R_gas);
    }
    return api;
}


// -----------------------------------------------------------------
//  ps_presence_relax_gate — true when the cell may be relaxed: both
//  phases Independent, or both relaxable with at least one in the
//  hysteresis band.  A cell with an Absent or Corridor
//  phase is single-phase and has nothing to relax; its metastability, if
//  any, is the flash nucleator's business.  This is what stands in for a
//  metastable guard: an un-nucleated cell is never dragged to the dome
//  because it is never relaxed, and a nucleated cell always equilibrates.
//  See docs/MODEL_AND_ALGORITHM.md §4.5 and §4.7.
// -----------------------------------------------------------------
//  Hysteresis band: a phase in the upper corridor (alpha >= alpha_cond/2)
//  with a non-degenerate density quotient still counts as relaxable, so
//  the source on/off boundary is moved away from the exact alpha the
//  relaxation itself moves.  Gate passes granted through the band are
//  counted per evaluation in PsCounters::pres_f3_hyst.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ps_presence_relax_gate(amrex::Real a1, amrex::Real m1, amrex::Real m2,
                            const PsPres& pr) noexcept
{
    const amrex::Real a2 = amrex::Real(1.0) - a1;
    const bool i1 = ps_regime(a1, m1, pr) == PsRegime::Independent;
    const bool i2 = ps_regime(a2, m2, pr) == PsRegime::Independent;
    if (i1 && i2) return true;
    {
        //  Hysteresis band: both phases must qualify (Independent or band).
        //  Part of the predicate on every build; only the count is host-only.
        auto band = [&pr](amrex::Real a, amrex::Real m) noexcept {
            return a >= amrex::Real(0.5) * pr.alpha_cond
                && m > pr.rho_deg * a;
        };
        const bool q1 = i1 || band(a1, m1);
        const bool q2 = i2 || band(a2, m2);
        if (q1 && q2 && (band(a1, m1) || band(a2, m2))) {
            PS_COUNT(++ps_counters().pres_f3_hyst);
            return true;
        }
    }
    return false;
}

// -----------------------------------------------------------------
//  ps_relax_entry_gate — the entry test the mode 2 and 5 relax
//  cells share: alpha_1 finite and inside [ALPHA_ROUNDOFF, 1 -
//  ALPHA_ROUNDOFF], both partial masses positive, and the presence gate
//  (docs §4.7), whose refusal is counted.  Returns true when the cell may
//  be relaxed, with alpha1_in, m1, m2 read out; false leaves the cell
//  untouched.  This is the RELAX class predicate of the class pass
//  (PS_classes.H), evaluated once per cell on every launch form; every
//  mode's gated wrapper below is this gate followed by the mode's
//  _admitted operator.  pr is ps_presence_params(), resolved by the host
//  sweep.
// -----------------------------------------------------------------
AMREX_GPU_HOST_DEVICE
inline bool
ps_relax_entry_gate(int i, int j, int k,
                    amrex::Array4<amrex::Real> const& U, const PsPres& pr,
                    amrex::Real& alpha1_in, amrex::Real& m1, amrex::Real& m2)
{
    using amrex::Real;
    constexpr Real alpha_floor = ps_const::ALPHA_ROUNDOFF;
    alpha1_in = U(i,j,k, UALPHA1);
    if (!std::isfinite(alpha1_in)) return false;
    if (alpha1_in < alpha_floor || alpha1_in > Real(1.0) - alpha_floor) return false;
    m1 = U(i,j,k, UM1RHO1);
    m2 = U(i,j,k, UM2RHO2);
    if (m1 <= Real(0.0) || m2 <= Real(0.0)) return false;
    // presence gate; refusal counted
    if (!ps_presence_relax_gate(alpha1_in, m1, m2, pr)) {
        PS_COUNT(++ps_counters().n_gate_presence);
        return false;
    }
    return true;
}

#include "PS_corridor.H"   // ps_corridor_project: needs ps_presence_relax_gate above

// -----------------------------------------------------------------
//  ps_mechanical_relax_cell (ps_relax_mode=0): instantaneous P1 = P2 by
//  Newton on alpha_1 (hem::ps_pressure_relax_cell) with the interfacial
//  work booked into UE1/UE2.  Writes UALPHA1, UE1, UE2 only; URHO,
//  momentum, UEDEN, UEINT and the phase masses are invariants of the
//  kernel.  Every early return leaves the cell unchanged, as does a
//  kernel failure.
//
//  Every mode's wrapper comes in two forms: the _admitted form is the
//  operator past the entry gate, with alpha1_in, m1, m2 already read and
//  gated by the caller (the class pass of PS_classes.H, or the gated
//  wrapper); the gated wrapper is ps_relax_entry_gate followed by it, the
//  per-cell body of the whole-box launch.  Both forms touch the cell
//  identically, so the launch forms are bitwise interchangeable.
// -----------------------------------------------------------------
AMREX_GPU_HOST_DEVICE
inline void
ps_mechanical_relax_admitted(int i, int j, int k,
                             amrex::Array4<amrex::Real> const& U,
                             hem::PsPhaseAPI const& eos,
                             amrex::Real alpha1_in, amrex::Real m1, amrex::Real m2,
                             double tol_P_rel = hem::PsPressureRelaxParams{}.tol_P_rel)
{
    // V6[3] = rho_mix |u|: the kernel forms ½(V6[3]/rho_mix)² = ½|u|²,
    // which is direction-free, so the momentum magnitude is exact.
    // Pack V6 in the layout ps_pressure_relax_cell expects.
    hem::V6 Uv = ps_pack_v6(U, i, j, k, alpha1_in, m1, m2);

    // Default PsPressureRelaxParams (max_iter 30, tol_P_rel 1e-4,
    // alpha_min ps_const::ALPHA_NEWTON_MIN = 1e-9, step_cap_frac 0.3;
    // single-phase skip at eos.dials.single_phase_threshold); the
    // stage projection (ps_apply_mechanical_projection) passes its own
    // tol_P_rel, the X3F residual tolerance, so the constraint is
    // converged rather than approximated between the stages.
    // Uv[0], Uv[4], Uv[5] update in place; Uv[1..3] are invariants.
    int reason = hem::PSR_OK;
    hem::PsPressureRelaxParams prp{};
    prp.tol_P_rel = tol_P_rel;
    const bool ok = hem::ps_pressure_relax_cell(Uv, eos, prp, &reason);
    if (!ok) {
        return;   // failure: leave the cell unchanged
    }

    // An out-of-range result is written as computed, never clamped.
    // Write back only the components the relaxation touches.
    U(i,j,k, UALPHA1) = Uv[0];
    U(i,j,k, UE1    ) = Uv[4];
    U(i,j,k, UE2    ) = Uv[5];
}

AMREX_GPU_HOST_DEVICE
inline void
ps_mechanical_relax_cell(int i, int j, int k,
                          amrex::Array4<amrex::Real> const& U,
                          hem::PsPhaseAPI const& eos, const PsPres& pr,
                          double tol_P_rel = hem::PsPressureRelaxParams{}.tol_P_rel)
{
    // Entry classification: non-finite, near-pure (either side), a
    // non-positive phase mass and the presence gate (refusal counted) all
    // return without touching the cell.
    amrex::Real alpha1_in, m1, m2;
    if (!ps_relax_entry_gate(i, j, k, U, pr, alpha1_in, m1, m2)) return;
    ps_mechanical_relax_admitted(i, j, k, U, eos, alpha1_in, m1, m2, tol_P_rel);
}


// -----------------------------------------------------------------
//  ps_ptg_relax_cell (ps_relax_mode=2): isochoric pressure equilibrium
//  plus finite-rate thermal relaxation, T1 -> T2 at rate 1/theta (exact
//  exponential), in a three-sweep Picard alternation.  With
//  theta > 0 the cell is not re-pinned on the dome, so the split
//  mass-transfer source keeps a Gibbs driving force while the partial
//  thermal coupling bounds the per-phase energies.  Writes UE1/UE2 only.
//  Retained as the finite-rate control closure (docs §5.7).
// -----------------------------------------------------------------
AMREX_GPU_HOST_DEVICE
inline void
ps_ptg_relax_admitted(int i, int j, int k,
                      amrex::Array4<amrex::Real> const& U,
                      hem::PsPhaseAPI const& eos,
                      amrex::Real alpha1_in, amrex::Real m1, amrex::Real m2,
                      amrex::Real dt, amrex::Real theta,
                      int outer_iters = 3)
{
    using amrex::Real;
    hem::V6 Uv = ps_pack_v6(U, i, j, k, alpha1_in, m1, m2);

    bool ok = true;
    for (int it = 0; it < outer_iters; ++it) {
        if (!hem::ps_iso_pressure_relax_cell(Uv, eos)) { ok = false; break; }
        if (!hem::ps_iso_thermal_relax_cell_finite(Uv, eos, double(dt),
                                                   double(theta))) { ok = false; break; }
    }
    if (!ok) return;

    U(i,j,k, UE1) = Uv[4];
    U(i,j,k, UE2) = Uv[5];
}

AMREX_GPU_HOST_DEVICE
inline void
ps_ptg_relax_cell(int i, int j, int k,
                  amrex::Array4<amrex::Real> const& U,
                  hem::PsPhaseAPI const& eos, const PsPres& pr,
                  amrex::Real dt, amrex::Real theta,
                  int outer_iters = 3)
{
    amrex::Real alpha1_in, m1, m2;
    if (!ps_relax_entry_gate(i, j, k, U, pr, alpha1_in, m1, m2)) return;
    ps_ptg_relax_admitted(i, j, k, U, eos, alpha1_in, m1, m2, dt, theta, outer_iters);
}


inline int ps_coexist_action();

// -----------------------------------------------------------------
//  ps_x3_grid_relax_cell (ps_relax_mode=5, the default): the coupled X3
//  operator hem::ps_x3_relax_cell.  P1 = P2 is a DAE constraint imposed
//  at every path evaluation; the thermal and SRT mass-transfer rates are
//  integrated by sub-stepped backward Euler on that manifold, with one
//  eligibility question (ps_coexist_action).  The flash stays upstream in
//  ps_apply_sources as the birth channel; the split transfer source is
//  compiled out at this mode.  Entry gates as mode 2; entry
//  and projection failures leave the cell untouched, a sub-step bail-out
//  keeps its partial, admissible, conservative progress.  Writes alpha,
//  both masses and UE1/UE2; mixture rho, momentum and E are invariant.
//  xp0 carries the host-resolved rate, carrier and policy dials
//  (ps_x3_params); pres_diag is ps_pres_diag().
//  Docs/MODEL_AND_ALGORITHM.md §5.1; 0-D harness PSAMR.ps_x3_test.
// -----------------------------------------------------------------
AMREX_GPU_HOST_DEVICE
inline void
ps_x3_grid_relax_admitted(int i, int j, int k,
                          amrex::Array4<amrex::Real> const& U,
                          hem::PsPhaseAPI const& eos,
                          amrex::Real alpha1_in, amrex::Real m1, amrex::Real m2,
                          const hem::PsX3Params& xp0, int pres_diag,
                          amrex::Real dt)
{
    hem::V6 Uv = ps_pack_v6(U, i, j, k, alpha1_in, m1, m2);

    hem::PsX3Params xp = xp0;
    //  [PS-DT] census from the kernel's own entry/exit states; -1 = the
    //  thermal leg did not run.
    double dt_pre = -1.0, dt_post = -1.0;
    if (pres_diag != 0) { xp.dT_pre = &dt_pre; xp.dT_post = &dt_post; }

    hem::ps_x3_relax_cell(Uv, eos, double(dt), xp);

#if !defined(AMREX_USE_GPU)
    if (pres_diag != 0 && dt_pre >= 0.0 && dt_post >= 0.0) {
        PsPresDiag& pdt = ps_pres_diag_data();
        ++pdt.n_dt;
        pdt.dtpre_sum  += dt_pre;
        pdt.dtpost_sum += dt_post;
        if (dt_pre  > pdt.dtpre_max)  pdt.dtpre_max  = dt_pre;
        if (dt_post > pdt.dtpost_max) pdt.dtpost_max = dt_post;
    }
#endif

    //  The kernel writes Uv only past its entry gates, and everything it
    //  writes is admissible and conservative, a bail-out's partial
    //  progress included.  Copy back unconditionally.
    ps_unpack_v6(U, i, j, k, Uv);
}





AMREX_GPU_HOST_DEVICE
inline void
ps_x3_grid_relax_cell(int i, int j, int k,
                      amrex::Array4<amrex::Real> const& U,
                      hem::PsPhaseAPI const& eos, const PsPres& pr,
                      const hem::PsX3Params& xp0, int pres_diag,
                      amrex::Real dt)
{
    // Entry gates (presence refusal counted).
    amrex::Real alpha1_in, m1, m2;
    if (!ps_relax_entry_gate(i, j, k, U, pr, alpha1_in, m1, m2)) return;
    ps_x3_grid_relax_admitted(i, j, k, U, eos, alpha1_in, m1, m2, xp0, pres_diag, dt);
}

// PSAMR.ps_relax_mode (6): 6 = the coupled operator flattened, X3F
// (ps_x3f_grid_relax_cell: one Newton in (alpha, T1, T2, q, dm) on the
// forward EOS with step-doubling control, docs/DESIGN_DECISIONS.md O-28
// and O-29; the default since 2026-10-27); 5 = the nested X3 operator
// (ps_x3_grid_relax_cell), retained as the control closure of the same
// fixed point; 0 = mechanical only, for frozen-limit tests;
// 2 = isochoric P + finite-rate thermal with the split Gibbs-driven
// transfer source (the finite-rate alternative, retained as the control
// closure while the X3 rate closure is under study, docs §5.7).  The
// default is 6 because the code defaults are the acceptance
// configuration.  This is the single read site, and an unrecognised value
// aborts here rather than falling through to mode-0 physics while the
// split transfer source (gated on != 5) silently re-opens.  The former
// modes 1 (P+T equilibrium) and 4 (sequential chain) and 3 (finite-rate
// mechanical leg) abort with their replacement named.
inline int ps_relax_mode()
{
    static const int m = []() -> int {
        const int v = ps_dial_int("ps_relax_mode", 6);
        if (v == 1 || v == 3 || v == 4) {
            amrex::Abort("PSAMR.ps_relax_mode = " + std::to_string(v) +
                         " is retired: use 6 (flattened X3F), 5 (coupled X3), 2 (finite-rate "
                         "thermal + split transfer) or 0 (mechanical only)");
        }
        if (v != 0 && v != 2 && v != 5 && v != 6) {
            amrex::Abort("PSAMR.ps_relax_mode = " + std::to_string(v) +
                         " is not a mode (valid: 0,2,5,6)");
        }
        return v;
    }();
    return m;
}

//  True when the relaxation operator owns mass transfer (modes 5 and 6:
//  the coupled operator integrates the SRT rate inside the relaxation
//  stage, so the split transfer source in PS_sources.H is gated out).
inline bool ps_relax_owns_mt() { const int m = ps_relax_mode(); return m == 5 || m == 6; }

// -----------------------------------------------------------------
//  PSAMR.ps_coexist_action (0): the thermal leg's response when a phase
//  leaves the coexistence band (T outside (T_triple, T_crit)), modes 4
//  and 5 (PsX3Params.coexist_action).  One predicate gates thermal
//  relaxation, mass transfer and the flash, so a cell whose vapour has
//  left the band has no operator left to bring it back — a self-lock,
//  silent unless named and counted.  0 = the leg stands (the runaway
//  clause, a hi-side-only exit beyond 2 T_crit, still runs it); 1 = abort
//  on the first exit with the cell context.  Any other value aborts: the
//  high-side veto is load-bearing at cross-critical contacts.  Docs §5.3;
//  DESIGN_DECISIONS X-5.
// -----------------------------------------------------------------
inline int ps_coexist_action()
{
    static const int a = []() -> int {
        const int v = ps_dial_int("ps_coexist_action", 0);
        if (v == 2 || v == 3 || v == 4) {
            amrex::Abort("PSAMR.ps_coexist_action=" + std::to_string(v)
                + " is retired: the thermal leg stands on a band exit "
                  "(the runaway clause is always active); use 0 or 1");
        }
        if (v != 0 && v != 1) {
            amrex::Abort("PSAMR.ps_coexist_action must be 0 or 1");
        }
#if defined(AMREX_USE_GPU)
        //  Action 1 is a diagnosis mode: its abort is raised inside the
        //  per-cell kernel (a no-op in device code) and read from the host
        //  counters (compiled out on GPU builds), so it would silently act
        //  as 0 here.
        if (v == 1) {
            amrex::Abort("PSAMR.ps_coexist_action=1 needs the host counters "
                         "and the per-cell abort; run it on a CPU build");
        }
#endif
        return v; }();
    return a;
}

// PSAMR.ps_theta_tau (1e-7 s): thermal relaxation time, modes 2 and 5.
//   theta > 0   finite backward-Euler rate (the linearised T1-T2 decay time).
//   theta <= 0  instantaneous thermal equilibrium: at mode 5, T1 = T2 joins
//               the P1 = P2 constraint.  It never means "thermal off" —
//               mass transfer without a thermal leg is not a configuration.
// The default is the acceptance-configuration value, so a change to it is
// visible in the acceptance table.  Docs/MODEL_AND_ALGORITHM.md §5.3.
inline amrex::Real ps_theta_tau()
{
    static const amrex::Real v = ps_dial_real("ps_theta_tau", amrex::Real(1.0e-7));
    return v;
}

// -----------------------------------------------------------------
//  ps_x3_params_from_dials — the mode-5 operator settings as the sweep
//  runs them.  Rate and carrier dials are the mass-transfer family, so a
//  configuration means the same thing at every mode; ps_mt_tau has no
//  meaning here (the SRT source carries no relaxation time).  Shared by
//  the sweep, the cell harvester (PS_harvest.H) and the replay bench
//  (PS_zerod_test.H), so a replayed cell sees the dials the run saw.
// -----------------------------------------------------------------
inline hem::PsX3Params
ps_x3_params_from_dials()
{
    hem::PsX3Params xp;
    xp.theta          = double(ps_theta_tau());
    xp.h_w            = hem::ps_dial_mt_h_weight();
    xp.srt_D          = hem::ps_dial_mt_srt_d();
    xp.srt_delta      = hem::ps_dial_mt_srt_delta();
    xp.coexist_action = ps_coexist_action();
    return xp;
}

// -----------------------------------------------------------------
//  ps_x3f_params_from_dials — the mode-6 operator settings: the X3 rate
//  and carrier dials, plus the flattened solve's own:
//    PSAMR.ps_x3f_jac (0): 0 analytic Jacobian, 1 forward differences
//    PSAMR.ps_x3f_fixed_newton (0): > 0 = fixed Newton count per solve
//    PSAMR.ps_x3f_max_newton (25), PSAMR.ps_x3f_tol (1e-10)
//    PSAMR.ps_x3f_step_tol_a (1e-3), PSAMR.ps_x3f_step_tol_e (1e-4), PSAMR.ps_x3f_step_rtol (0.1):
//      the step-doubling error control of the sub-step ladder (tol_a <= 0 disables)
//    PSAMR.ps_x3f_order (1; 2 under ps_wp_order=4): the sub-step's order;
//      1 = backward Euler, 2 = the two-stage SDIRK (second order in the
//      rate terms, so the full Strang step is second order;
//      docs/PLAN_high_order.pdf follow-up 4, step A).  The order-4 path
//      defaults to 2 (Marc, 18 Nov 2026, on the step A record); set 1
//      explicitly there for the backward-Euler sub-step.  Any other
//      value aborts.
//  Shared with the replay bench (PS_zerod_test.H).
// -----------------------------------------------------------------
inline hem::PsX3fParams
ps_x3f_params_from_dials()
{
    const hem::PsX3Params xp = ps_x3_params_from_dials();
    hem::PsX3fParams xf;
    xf.theta = xp.theta; xf.h_w = xp.h_w; xf.srt_D = xp.srt_D; xf.srt_delta = xp.srt_delta;
    xf.coexist_action = xp.coexist_action;
    static const int    jac   = ps_dial_int("ps_x3f_jac", 0);
    static const int    fixed = ps_dial_int("ps_x3f_fixed_newton", 0);
    static const int    maxn  = ps_dial_int("ps_x3f_max_newton", 25);
    static const double tol   = double(ps_dial_real("ps_x3f_tol", amrex::Real(1.0e-10)));
    static const double tola  = double(ps_dial_real("ps_x3f_step_tol_a", amrex::Real(1.0e-3)));   // <= 0: no step control
    static const double tole  = double(ps_dial_real("ps_x3f_step_tol_e", amrex::Real(1.0e-4)));
    static const double rtol  = double(ps_dial_real("ps_x3f_step_rtol",  amrex::Real(0.1)));
    static const int    order = []() -> int {
        //  The order-4 default, read as PSAMR_advance.cpp reads it for
        //  ps_strang and ps_rk_project: a query without adding back,
        //  PS_umeth.cpp owns and validates ps_wp_order.
        int wpo = 2; amrex::ParmParse("PSAMR").query("ps_wp_order", wpo);
        const int v = ps_dial_int("ps_x3f_order", (wpo == 4) ? 2 : 1);
        if (v != 1 && v != 2) { amrex::Abort("PSAMR.ps_x3f_order must be 1 (backward Euler) or 2 (two-stage SDIRK)"); }
        return v; }();
    xf.jac = jac; xf.fixed_newton = fixed; xf.max_newton = maxn; xf.tol = tol;
    xf.step_tol_a = tola; xf.step_tol_e = tole; xf.step_rtol = rtol;
    xf.order = order;
    return xf;
}

//  The X3F outcome census on the host (the [PS-X3F] line; a no-op on the
//  device) and the band-exit abort of ps_coexist_action=1.
AMREX_GPU_HOST_DEVICE
inline void
ps_x3f_count(int i, int j, int k, int status, int exit_code, int nsub, int nit, int coexist_action)
{
#if !defined(AMREX_USE_GPU)
    hem::PsX3Stats& xs = hem::ps_x3_stats();
    ++xs.n_calls;
    switch (status) {
        case hem::X3_OK:            ++xs.n_ok; break;
        case hem::X3_ENTRY_REFUSED: ++xs.n_entry_refused; break;
        case hem::X3_STOOD:         ++xs.n_gate_stood; break;
        case hem::X3_PROJ_ENTRY:    ++xs.n_proj_entry; break;
        case hem::X3_NEWTON:        ++xs.n_newton; break;
        case hem::X3_DEAD:          ++xs.n_dead; break;
        case hem::X3_SUBCAP:        ++xs.n_subcap; break;
        default:                    ++xs.n_newton; break;   // fixed-count non-convergence
    }
    xs.n_sub += nsub; xs.n_nit += nit;
    if (coexist_action == 1 && (exit_code & 3) != 0 && (exit_code & 4) == 0) {
        amrex::AllPrint()
            << "\n[PS-X3F] COEXISTENCE BAND EXIT (ps_coexist_action=1 -> abort)"
            << "  cell (" << i << "," << j << "," << k << ")  exit bits " << exit_code << "\n";
        amrex::Abort("PS-X3F: phase left the coexistence band (PSAMR.ps_coexist_action=1)");
    }
#else
    amrex::ignore_unused(i, j, k, status, exit_code, nsub, nit, coexist_action);
#endif
}

// -----------------------------------------------------------------
//  ps_x3f_grid_relax_cell (ps_relax_mode=6): the flattened operator on
//  one cell, the same entry gates, pack/unpack and [PS-DT] census as the
//  X3 wrapper.  On host builds the per-cell outcome and cost feed the
//  same PsX3Stats the [PS-X3] report prints for mode 5, so the two modes
//  are read with one report; the band-exit abort (ps_coexist_action=1)
//  is applied here from the operator's exit code.
// -----------------------------------------------------------------
AMREX_GPU_HOST_DEVICE
inline void
ps_x3f_grid_relax_admitted(int i, int j, int k,
                           amrex::Array4<amrex::Real> const& U,
                           hem::PsPhaseAPI const& eos,
                           amrex::Real alpha1_in, amrex::Real m1, amrex::Real m2,
                           const hem::PsX3fParams& xf0, int pres_diag,
                           amrex::Real dt)
{
    hem::V6 Uv = ps_pack_v6(U, i, j, k, alpha1_in, m1, m2);

    hem::PsX3fParams xf = xf0;
    double dt_pre = -1.0, dt_post = -1.0;
    int status = 0, exit_code = 0, nit = 0, nsub = 0;
    xf.status = &status; xf.exit_code = &exit_code; xf.it_out = &nit; xf.sub_out = &nsub;
    if (pres_diag != 0) { xf.dT_pre = &dt_pre; xf.dT_post = &dt_post; }

    hem::ps_x3f_relax_cell(Uv, eos, double(dt), xf);

    ps_x3f_count(i, j, k, status, exit_code, nsub, nit, xf0.coexist_action);
#if !defined(AMREX_USE_GPU)
    if (pres_diag != 0 && dt_pre >= 0.0 && dt_post >= 0.0) {
        PsPresDiag& pdt = ps_pres_diag_data();
        ++pdt.n_dt;
        pdt.dtpre_sum  += dt_pre;
        pdt.dtpost_sum += dt_post;
        if (dt_pre  > pdt.dtpre_max)  pdt.dtpre_max  = dt_pre;
        if (dt_post > pdt.dtpost_max) pdt.dtpost_max = dt_post;
    }
#endif
    //  As the X3 wrapper: the operator writes Uv only past its entry gates,
    //  and what it writes is admissible and conservative.  Copy back.
    ps_unpack_v6(U, i, j, k, Uv);
}

AMREX_GPU_HOST_DEVICE
inline void
ps_x3f_grid_relax_cell(int i, int j, int k,
                       amrex::Array4<amrex::Real> const& U,
                       hem::PsPhaseAPI const& eos, const PsPres& pr,
                       const hem::PsX3fParams& xf0, int pres_diag,
                       amrex::Real dt)
{
    amrex::Real alpha1_in, m1, m2;
    if (!ps_relax_entry_gate(i, j, k, U, pr, alpha1_in, m1, m2)) return;
    ps_x3f_grid_relax_admitted(i, j, k, U, eos, alpha1_in, m1, m2, xf0, pres_diag, dt);
}

//  The per-sweep counters of the relaxation sweep ([PS-GATE], [PS-CORRIDOR],
//  [PS-PRES], [PS-DT]); under ps_imex=3 they run over the four stage
//  sweeps and the final, reset at stage 1 and reported by the final.
inline void
ps_relax_reset_sweep_counters()
{
    ps_counters().n_gate_presence = 0;
    ps_counters().corridor_projected = ps_counters().corridor_host_fail = ps_counters().corridor_noroot = 0;
    ps_counters().corridor_probed = 0;
    ps_pres_diag_data().reset();
}

// -----------------------------------------------------------------
//  ps_relax_report — the gated reports of a relaxation sweep ([PS-CLASS],
//  [PS-PRES], [PS-CORRIDOR], [PS-GATE], [PS-DT], [PS-X3]/[PS-X3F]) and
//  the reset of the X3 outcome counters.  n_relax is the class pass's
//  admitted count (-1 when not formed), n_cells its cell total, form the
//  launch form.  Called by ps_apply_relaxation and by the IMEX final
//  sweep (ps_apply_imex_sweep at stage 5), whose stage sweeps feed the same
//  counters.
// -----------------------------------------------------------------
inline void
ps_relax_report(amrex::Long n_relax, amrex::Long n_cells, int form)
{
    // [PS-PRES] (ps_pres_diag): the post-thermal P1 != P2 residual by cause.
    if (ps_pres_diag() != 0) {
        //  [PS-CLASS]: the relaxation population from the class pass, on
        //  every build (the device has no other census).
        if (n_relax >= 0) {
            amrex::AllPrint()
                << "[PS-CLASS] relax: admitted=" << n_relax << " of " << n_cells
                << " cells (" << (n_cells > 0 ? 100.0 * double(n_relax) / double(n_cells) : 0.0)
                << " %), launch form " << form << "\n";
        }
        const PsPresDiag& pd = ps_pres_diag_data();
        auto mean = [](double s, long n) { return n > 0 ? s/double(n) : 0.0; };
        amrex::AllPrint()
            << "[PS-PRES] thermal RAN+moved(>1K): n=" << pd.n_run_moved
            << " max=" << pd.dp_moved_max
            << " mean=" << mean(pd.dp_moved_sum, pd.n_run_moved)
            << " | RAN+still: n=" << pd.n_run_still
            << " max=" << pd.dp_still_max
            << " mean=" << mean(pd.dp_still_sum, pd.n_run_still)
            << " | gate-SKIPPED: n=" << pd.n_skipped
            << " max=" << pd.dp_skip_max
            << " mean=" << mean(pd.dp_skip_sum, pd.n_skipped)
            << " | post-invalid=" << pd.n_postinval
            << "   (|P1-P2|/max(P1,P2) at chain exit)\n";
        //  [PS-GATE]: presence-gate refusals this sweep, all modes (cells
        //  [PS-X3] never sees).  Not an anomaly — the gate doing its job —
        //  but a stall routed through it must be visible.
        //  [PS-CORRIDOR]: the saturation projection's census, with the count
        //  of own-branch probes (one EOS inversion per (Corridor, Independent)
        //  cell per sweep, the whole-grid EOS cost of this stage).
        if (ps_counters().corridor_probed + ps_counters().corridor_projected
            + ps_counters().corridor_host_fail + ps_counters().corridor_noroot > 0) {
            amrex::AllPrint()
                << "[PS-CORRIDOR] probed=" << ps_counters().corridor_probed
                << " projected=" << ps_counters().corridor_projected
                << " host_fail=" << ps_counters().corridor_host_fail
                << " noroot=" << ps_counters().corridor_noroot << "\n";
        }
        if (ps_counters().n_gate_presence > 0) {
            amrex::AllPrint()
                << "[PS-GATE] S4 presence relax gate refused n="
                << ps_counters().n_gate_presence
                << " in-band cells this sweep (phase pair not "
                   "Independent+Independent; all modes)\n";
        }
        //  [PS-DT]: thermal completeness, |T1-T2| at entry vs exit over the
        //  cells whose thermal leg ran.
        if (pd.n_dt > 0) {
            amrex::AllPrint()
                << "[PS-DT] n=" << pd.n_dt
                << " dTpre: max=" << pd.dtpre_max
                << " mean=" << mean(pd.dtpre_sum, pd.n_dt)
                << " | dTpost: max=" << pd.dtpost_max
                << " mean=" << mean(pd.dtpost_sum, pd.n_dt)
                << "   (K; residual ratio max-based="
                << (pd.dtpre_max > 0.0 ? pd.dtpost_max / pd.dtpre_max : 0.0)
                << ")\n";
        }
    }
    //  [PS-X3]: the coupled operator's outcome and the P1 = P2 constraint's
    //  failures, by cause.  Modes 5 and 6 (mode 6 fills the outcome and
    //  cost fields from the flattened operator's per-cell codes; its
    //  constraint-cause and path fields stay zero); 'dead' counts
    //  structural refusals, and a sweep in which every cell is dead is
    //  banner-flagged.
    if (ps_relax_owns_mt()) {
        hem::PsX3Stats& xs = hem::ps_x3_stats();
        if (ps_pres_diag() != 0 && xs.n_calls > 0) {
            amrex::AllPrint()
                << (ps_relax_mode() == 6 ? "[PS-X3F] calls=" : "[PS-X3] calls=") << xs.n_calls
                << " ok=" << xs.n_ok
                << " gate_stood=" << xs.n_gate_stood
                << " entry_refused=" << xs.n_entry_refused
                << " | dead(rc2)=" << xs.n_dead
                << " newton_nonconv=" << xs.n_newton
                << " tiny_nonconv=" << xs.n_tiny_nonconv
                << " subcap=" << xs.n_subcap
                << " | constraint: entry_fail=" << xs.n_proj_entry
                << " (input=" << xs.n_pr_input
                << " eos=" << xs.n_pr_eos
                << " slope=" << xs.n_pr_slope
                << " maxiter=" << xs.n_pr_maxiter << ")"
                << " path_fail=" << xs.n_proj_path
                << " | cost: sub=" << xs.n_sub
                << " newton_it=" << xs.n_nit
                << " path=" << xs.n_path
                << " proj_it=" << hem::ps_mt_stats().pr_iters
                << " (cumulative)";
            if (xs.n_dead > 0 && xs.n_ok == 0) {
                amrex::AllPrint()
                    << "   *** EVERY cell structurally dead: the operator "
                       "is a NO-OP this sweep";
            }
            amrex::AllPrint() << "\n";
        }
#if !defined(AMREX_USE_GPU)
        //  The class pass admits exactly the cells the kernel counts as
        //  calls; a difference would mean the two forms gate differently.
        if (n_relax >= 0 && ps_pres_diag() != 0 && n_relax != amrex::Long(xs.n_calls)) {
            amrex::AllPrint() << "[PS-CLASS] *** MISMATCH: class pass admitted " << n_relax
                              << " cells, the kernel counted " << xs.n_calls << " calls\n";
        }
#endif
        xs.reset();
    }
}

// -----------------------------------------------------------------
//  The IMEX stage sweeps (PSAMR.ps_imex=3; docs/PLAN_high_order.pdf
//  follow-up 4, step B; the stepping in PSAMR_advance.cpp).  D holds the
//  stage transfers (q, dm) of the four implicit stages, two components
//  per stage, each written by that stage's own solve; st the per-cell
//  stage status, 0 while every stage was OK, else the code of the first
//  cause met (1 stood at the band, 2 a Newton failure, 4 a base refusal:
//  entry refused, entry projection, dead rates; 8 the final apply), which
//  marks the cell for the fallback of the final sweep.  c[j] is the
//  tableau factor a_sj / alpha of the earlier stage j (the kernel's
//  offsets are sum_j c[j] D_j); dt_s = alpha dt, and 0 for the final
//  apply (stage 5, with c the weights b_j / alpha).
//
//  Each operator comes in the two forms of the relaxation sweep: the
//  _admitted body past the entry gate, and the gated wrapper (the gate,
//  then the body), so the sweep's three launch forms (ps_class_compact)
//  are bitwise interchangeable (PS_classes.H).  The stage's own transfer
//  slot is zeroed, and the dropped-transfer census taken, for every cell
//  of the box, in the wrapper or in the class pass.
// -----------------------------------------------------------------
AMREX_GPU_HOST_DEVICE
inline void
ps_x3f_imex_stage_admitted(int i, int j, int k,
                           amrex::Array4<amrex::Real> const& U,
                           amrex::Array4<amrex::Real> const& D,
                           amrex::Array4<int> const& st,
                           hem::PsPhaseAPI const& eos,
                           amrex::Real alpha1_in, amrex::Real m1, amrex::Real m2,
                           const hem::PsX3fParams& xf0, int stage,
                           amrex::GpuArray<amrex::Real,4> const& c, amrex::Real dt_s)
{
    const int nprev = (stage <= 4) ? stage - 1 : 4;
    double q_off = 0.0, dm_off = 0.0;
    for (int s = 0; s < nprev; ++s) {
        q_off  += double(c[s]) * double(D(i,j,k,2*s));
        dm_off += double(c[s]) * double(D(i,j,k,2*s+1));
    }
    hem::V6 Uv = ps_pack_v6(U, i, j, k, alpha1_in, m1, m2);
    hem::PsX3fParams xf = xf0;
    int status = 0, exit_code = 0, nit = 0, nsub = 0;
    double qo = 0.0, dmo = 0.0;
    xf.imex = 1; xf.imex_q_off = q_off; xf.imex_dm_off = dm_off;
    xf.imex_q_out = &qo; xf.imex_dm_out = &dmo;
    xf.status = &status; xf.exit_code = &exit_code; xf.it_out = &nit; xf.sub_out = &nsub;
    hem::ps_x3f_relax_cell(Uv, eos, double(dt_s), xf);
    //  A stand at the final apply is the state the fallback would reach
    //  (the projected base), so it is not a fallback.
    if (status == hem::X3_OK || (stage == 5 && status == hem::X3_STOOD)) {
        if (stage <= 4) { D(i,j,k,2*(stage-1)) = amrex::Real(qo); D(i,j,k,2*(stage-1)+1) = amrex::Real(dmo); }
    } else if (st(i,j,k) == 0) {
        st(i,j,k) = (stage == 5) ? 8
                  : (status == hem::X3_STOOD) ? 1
                  : (status == hem::X3_NEWTON || status == hem::X3_FIXED_NONCONV) ? 2 : 4;
    }
    ps_x3f_count(i, j, k, status, exit_code, nsub, nit, xf0.coexist_action);
    ps_unpack_v6(U, i, j, k, Uv);
}

//  The per-cell work of a stage sweep that is not the operator: the
//  stage's own transfer slot zeroed, the corridor projection, and (the
//  final apply, host) the census of a cell admitted at the stages whose
//  transfers the final's presence gate refuses.  Returns the gate.
AMREX_GPU_HOST_DEVICE
inline bool
ps_x3f_imex_prepare(int i, int j, int k,
                    amrex::Array4<amrex::Real> const& U,
                    amrex::Array4<amrex::Real> const& D,
                    const PsPres& pr, int stage,
                    amrex::Real& alpha1_in, amrex::Real& m1, amrex::Real& m2)
{
    if (stage <= 4) { D(i,j,k,2*(stage-1)) = amrex::Real(0.0); D(i,j,k,2*(stage-1)+1) = amrex::Real(0.0); }
    ps_corridor_project(i, j, k, U, pr);
    const bool admitted = ps_relax_entry_gate(i, j, k, U, pr, alpha1_in, m1, m2);
#if !defined(AMREX_USE_GPU)
    if (!admitted && stage == 5) {
        bool any = false;
        for (int s = 0; s < 8; ++s) any = any || (D(i,j,k,s) != amrex::Real(0.0));
        if (any) ++hem::ps_x3_stats().n_imex_dropped;
    }
#endif
    return admitted;
}

//  The final sweep's operator on an admitted cell: the final apply of
//  the stage transfers where every stage solved, else the present
//  operator over the whole step from the explicit final state (the
//  fallback, counted by cause on the host).  That state already carries
//  2/3 of stage 4's transfers, so a fallback cell sees part of its
//  transfer twice: the fallback is first order, as the plan records.
AMREX_GPU_HOST_DEVICE
inline void
ps_x3f_imex_final_admitted(int i, int j, int k,
                           amrex::Array4<amrex::Real> const& U,
                           amrex::Array4<amrex::Real> const& D,
                           amrex::Array4<int> const& st,
                           hem::PsPhaseAPI const& eos,
                           amrex::Real alpha1_in, amrex::Real m1, amrex::Real m2,
                           const hem::PsX3fParams& xf0, int pres_diag,
                           amrex::GpuArray<amrex::Real,4> const& bw, amrex::Real dt)
{
    if (st(i,j,k) == 0) {
        ps_x3f_imex_stage_admitted(i, j, k, U, D, st, eos, alpha1_in, m1, m2, xf0, 5, bw, amrex::Real(0.0));
        if (st(i,j,k) == 0) return;
    }
#if !defined(AMREX_USE_GPU)
    {
        hem::PsX3Stats& xs = hem::ps_x3_stats();
        switch (st(i,j,k)) {
            case 2:  ++xs.n_imex_fb_newton; break;
            case 4:  ++xs.n_imex_fb_base;   break;
            case 1:  ++xs.n_imex_fb_stand;  break;
            default: ++xs.n_imex_fb_final;  break;
        }
    }
#endif
    //  The fallback reads the cell as the gate left it (the kernel wrote
    //  nothing on a failed final apply).
    ps_x3f_grid_relax_admitted(i, j, k, U, eos, U(i,j,k,UALPHA1), U(i,j,k,UM1RHO1), U(i,j,k,UM2RHO2), xf0, pres_diag, dt);
}

//  One sweep of the IMEX stepping: stage 1 to 4 the stage solves over
//  dt_s, stage 5 the final apply with the fallback.  The launch forms of
//  PS_classes.H: 0 the whole box; 1 the class pass (the preparation and
//  the gate, every cell) and the operator over the compact list; 2 the
//  class pass and a whole-box launch gated on the class byte.  Under
//  mode 0 the operator is the mechanical projection at mode0_tol (no
//  rate terms, D stays zero).  The final sweep reports and resets the
//  relaxation counters, which the stage-1 sweep reset.
inline void
ps_apply_imex_sweep(amrex::MultiFab& S, amrex::MultiFab& D, amrex::iMultiFab& st,
                    int stage, amrex::GpuArray<amrex::Real,4> const& c,
                    amrex::Real dt_s, amrex::Real dt, double mode0_tol)
{
    BL_PROFILE("PS::ps_apply_imex_sweep()");
    const hem::PsPhaseAPI eos = ps_make_psamr_eos_api();
    const PsPres l_pres = ps_presence_params();
    const hem::PsX3fParams xf = ps_x3f_params_from_dials();
    const int relax_mode = ps_relax_mode();
    const int pres_diag = ps_pres_diag();
    const int form = ps_class_compact();
    const bool final = (stage == 5);
    if (stage == 1) ps_relax_reset_sweep_counters();
    amrex::Long n_relax = -1, n_cells = 0;
    PsClassLists* wsp = nullptr;
    if (form != 0) { wsp = new PsClassLists(S, 0); n_cells = wsp->ncells_total(); }
    for (amrex::MFIter mfi(S); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& U  = S.array(mfi);
        amrex::Array4<amrex::Real> const& Da = D.array(mfi);
        amrex::Array4<int>         const& sa = st.array(mfi);
        auto admitted = [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::Real a1, amrex::Real m1, amrex::Real m2) noexcept
        {
            if (relax_mode != 6) {
                ps_mechanical_relax_admitted(i, j, k, U, eos, a1, m1, m2, mode0_tol);
            } else if (final) {
                ps_x3f_imex_final_admitted(i, j, k, U, Da, sa, eos, a1, m1, m2, xf, pres_diag, c, dt);
            } else {
                ps_x3f_imex_stage_admitted(i, j, k, U, Da, sa, eos, a1, m1, m2, xf, stage, c, dt_s);
            }
        };
        if (form == 0) {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::Real a1, m1, m2;
                if (ps_x3f_imex_prepare(i, j, k, U, Da, l_pres, stage, a1, m1, m2)) admitted(i, j, k, a1, m1, m2);
            });
            continue;
        }
        PsClassLists& ws = *wsp;
        unsigned char* AMREX_RESTRICT cls   = ws.cls_ptr(mfi);
        int*           AMREX_RESTRICT list  = ws.list_ptr(mfi);
        int*           AMREX_RESTRICT count = ws.count_ptr(mfi);
        const int ncell = ws.ncell(mfi);
        ps_classify_box(bx, cls, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> unsigned char
        {
            amrex::Real a1, m1, m2;
            return ps_x3f_imex_prepare(i, j, k, U, Da, l_pres, stage, a1, m1, m2) ? ps_class::RELAX : ps_class::NONE;
        });
        //  The operator on an admitted cell: the gate's reads repeated
        //  (nothing has written the cell since the class pass), no gate.
        auto op = [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            admitted(i, j, k, U(i,j,k, UALPHA1), U(i,j,k, UM1RHO1), U(i,j,k, UM2RHO2));
        };
        if (form == 2) {
            ps_class_launch_box(bx, cls, ps_class::RELAX, op);
            if (pres_diag != 0) ps_class_list_box(ncell, cls, ps_class::RELAX, list, count);
        } else {
            ps_class_list_box(ncell, cls, ps_class::RELAX, list, count);
            ps_class_launch_list(bx, list, count, op);
        }
    }
    if (wsp != nullptr) { if (pres_diag != 0) n_relax = wsp->total(); delete wsp; }
    if (final) {
#if !defined(AMREX_USE_GPU)
        if (pres_diag != 0 && relax_mode == 6) {
            const hem::PsX3Stats& xs = hem::ps_x3_stats();
            amrex::AllPrint() << "[PS-IMEX] fallback: newton=" << xs.n_imex_fb_newton
                              << " base=" << xs.n_imex_fb_base
                              << " stand=" << xs.n_imex_fb_stand
                              << " final=" << xs.n_imex_fb_final
                              << " dropped=" << xs.n_imex_dropped
                              << "   (cells of this step; the [PS-X3F] line counts every stage solve)\n";
        }
#endif
        ps_relax_report(n_relax, n_cells, form);
    }
}

// -----------------------------------------------------------------
//  ps_apply_mechanical_projection — the constraint part of the
//  relaxation operator alone (instantaneous P1 = P2 by the mode-0 kernel
//  ps_mechanical_relax_cell, past the same entry gate), on every cell of
//  the box grown by nghost.  Used inside the Runge-Kutta stages under
//  PSAMR.ps_rk_project=1 (docs/PLAN_high_order.pdf section 4.3, Stage 4):
//  the stage state is projected back onto the mechanical-equilibrium
//  manifold so that every stage is a convex combination of admissible
//  forward Euler steps; the finite-rate legs (thermal, mass transfer,
//  flash) and the corridor projection stay in the reaction step outside
//  the stages.  No reports, no counter resets: the stage sweep leaves the
//  reaction step's census alone.  tol_P_rel is the Newton's stopping
//  tolerance on |P1-P2|/max(P1,P2) (PSAMR.ps_rk_project_tol, default the
//  X3F residual tolerance 1e-10, so the stage state is on the constraint
//  to the precision the production operator solves it; the mode-0
//  kernel's own 1e-4 floored the Stage 4 self-convergence at 3e-5 of P).
// -----------------------------------------------------------------
inline void
ps_apply_mechanical_projection(amrex::MultiFab& S, int nghost = 0,
                               double tol_P_rel = 1.0e-10)
{
    BL_PROFILE("PS::ps_apply_mechanical_projection()");
    const hem::PsPhaseAPI eos = ps_make_psamr_eos_api();
    const PsPres l_pres = ps_presence_params();
    for (amrex::MFIter mfi(S); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.growntilebox(nghost);
        amrex::Array4<amrex::Real> const& U = S.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            ps_mechanical_relax_cell(i, j, k, U, eos, l_pres, tol_P_rel);
        });
    }
}

// -----------------------------------------------------------------
//  ps_apply_mechanical_projection_pt — the stage projection on point
//  values (PSAMR.ps_rk_project_pt=1; docs/PLAN_high_order.pdf follow-up
//  13).  The average of a field in mechanical equilibrium is not in
//  equilibrium at second order, so the projection of cell averages is a
//  second-order term of the order-4 path; here the cell's point value
//  U_c = <U> - (1/24) sum_d d2_d <U> is projected and the kernel's
//  change on the three slots it owns (UALPHA1, UE1, UE2; the invariants
//  are untouched), D_c = Pi(U_c) - U_c, is converted back and added to
//  the average, <U> += D_c + (1/24) sum_d d2_d D_c.  The pair acts on
//  the change, not on the state: its own fourth-order truncation is
//  never written into the averages, where it would be corrected again
//  at every stage and add up as a drift (follow-up 14).  Sg is the stage state with at least two
//  ghost layers filled; S_new receives the valid region.  A cell with a
//  presence change among its face neighbours, or whose point value is
//  not a state on its phases' branches, or whose converted average
//  leaves the admissible set or is not a state, takes the projection of
//  its average (counted proj_pt_raw).  With ps_pres_diag the host keeps
//  the worst |P1-P2|/P of the converted averages (the constraint
//  residual, reported on the [PS-GUARD] line).
// -----------------------------------------------------------------
namespace ps_pt_detail {
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool is_state(const amrex::Real U[NVAR], const PsPres& pr) noexcept
{
    PsCellState s; ps_cell_prims(U, pr, s);
    amrex::Real Yp[NUM_SPECIES]; ps_pure_species(Yp);
    amrex::Real Pq, Tq, Sq;
    if (s.rg1 != PsRegime::Absent && !EOS::REY2PTS_phase_try(s.rho1, s.e1, Yp, hem::Phase3::Liquid, Pq, Tq, Sq)) return false;
    if (s.rg2 != PsRegime::Absent && !EOS::REY2PTS_phase_try(s.rho2, s.e2, Yp, hem::Phase3::Vapor,  Pq, Tq, Sq)) return false;
    return true;
}
//  The mode-0 kernel on a local conserved array.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void project_local(amrex::Real U[NVAR], hem::PsPhaseAPI const& eos, const PsPres& pr, double tol) noexcept
{
    amrex::Array4<amrex::Real> A(U, amrex::Dim3{0,0,0}, amrex::Dim3{1,1,1}, NVAR);
    ps_mechanical_relax_cell(0, 0, 0, A, eos, pr, tol);
}
}

inline void
ps_apply_mechanical_projection_pt(amrex::MultiFab& S_new, const amrex::MultiFab& Sg,
                                  double tol_P_rel = 1.0e-10)
{
    BL_PROFILE("PS::ps_apply_mechanical_projection_pt()");
    AMREX_ALWAYS_ASSERT(Sg.nGrow() >= 2);
    const hem::PsPhaseAPI eos = ps_make_psamr_eos_api();
    const PsPres l_pres = ps_presence_params();
    const int slots[3] = { UALPHA1, UE1, UE2 };
    for (amrex::MFIter mfi(S_new); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.validbox();
        const amrex::Box gbx = amrex::grow(bx, 1);
        amrex::FArrayBox pc_fab(gbx, NVAR + 1, amrex::The_Async_Arena());
        auto const& pc = pc_fab.array();
        amrex::Array4<const amrex::Real> const& Ug = Sg.const_array(mfi);
        amrex::Array4<amrex::Real> const& Un = S_new.array(mfi);
        // Pass 1: the projected point value (or the projected average) of
        // every cell of the box grown by one.
        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            using amrex::Real;
            Real U[NVAR], Uc[NVAR], U0[NVAR], Uc0[NVAR];
            for (int n = 0; n < NVAR; ++n) { U[n] = Ug(i, j, k, n); Uc[n] = U[n]; }
            PsCellState sa; ps_cell_prims(U, l_pres, sa);
            //  Only a two-phase cell is a candidate: the kernel's entry
            //  gate refuses the others, so the point form has nothing to
            //  do there and the refusal is not counted.
            const bool two = (sa.rg1 != PsRegime::Absent) && (sa.rg2 != PsRegime::Absent);
            bool pt = two;
            for (int d = 0; d < AMREX_SPACEDIM && pt; ++d) {
                const int oi = (d == 0), oj = (d == 1), ok = (d == 2);
                Real Um[NVAR], Up[NVAR];
                for (int n = 0; n < NVAR; ++n) {
                    Um[n] = Ug(i - oi, j - oj, k - ok, n);
                    Up[n] = Ug(i + oi, j + oj, k + ok, n);
                    Uc[n] -= (Up[n] - Real(2.0) * U[n] + Um[n]) / Real(24.0);
                }
                PsCellState sm, sp; ps_cell_prims(Um, l_pres, sm); ps_cell_prims(Up, l_pres, sp);
                if ((sm.rg1 == PsRegime::Absent) != (sa.rg1 == PsRegime::Absent) || (sm.rg2 == PsRegime::Absent) != (sa.rg2 == PsRegime::Absent) ||
                    (sp.rg1 == PsRegime::Absent) != (sa.rg1 == PsRegime::Absent) || (sp.rg2 == PsRegime::Absent) != (sa.rg2 == PsRegime::Absent)) pt = false;
            }
            if (pt) {
                PsCellState sc; ps_cell_prims(Uc, l_pres, sc);
                if ((sc.rg1 == PsRegime::Absent) != (sa.rg1 == PsRegime::Absent) || (sc.rg2 == PsRegime::Absent) != (sa.rg2 == PsRegime::Absent)) pt = false;
                else if (!ps_pt_detail::is_state(Uc, l_pres)) pt = false;
            }
            for (int n = 0; n < NVAR; ++n) { U0[n] = U[n]; Uc0[n] = Uc[n]; }
            if (two && !pt) { PS_COUNT(++ps_counters().proj_pt_raw); }
            Real* W = pt ? Uc : U;
#if !defined(AMREX_USE_GPU)
            //  The deviation from equilibrium of the point value the kernel
            //  receives, and the movement it applies (follow-up 14).
            Real a_before = W[UALPHA1], dev = Real(0.0);
            if (ps_pres_diag() != 0 && pt) {
                const PsCellState sd = ps_cell_state(W, l_pres);
                if (sd.rg1 != PsRegime::Absent && sd.rg2 != PsRegime::Absent && sd.P_mix > Real(0.0))
                    dev = std::abs(sd.P1 - sd.P2) / sd.P_mix;
            }
#endif
            ps_pt_detail::project_local(W, eos, l_pres, tol_P_rel);
#if !defined(AMREX_USE_GPU)
            if (ps_pres_diag() != 0 && pt) {
                if (dev > ps_counters().proj_pt_dev) ps_counters().proj_pt_dev = dev;
                const Real mv = std::abs(W[UALPHA1] - a_before);
                if (mv > ps_counters().proj_pt_move) ps_counters().proj_pt_move = mv;
            }
#endif
            //  The store holds the kernel's change on the three slots (the
            //  projected value less the value it received), so the pair's
            //  own truncation is never written into the state: an
            //  equilibrium field passes through unchanged.
            for (int n = 0; n < NVAR; ++n) pc(i, j, k, n) = W[n] - (pt ? Uc0[n] : U0[n]);
            pc(i, j, k, NVAR) = pt ? Real(1.0) : Real(0.0);
        });
        // Pass 2: the average's three slots plus the converted change of
        // the projected point values, into the valid region.
        amrex::Array4<const amrex::Real> const pcc = pc_fab.const_array();
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            using amrex::Real;
            Real Uavg[NVAR];
            for (int n = 0; n < NVAR; ++n) Uavg[n] = Ug(i, j, k, n);
            if (pcc(i, j, k, NVAR) == Real(0.0)) {
                for (int q = 0; q < 3; ++q) Un(i, j, k, slots[q]) = Uavg[slots[q]] + pcc(i, j, k, slots[q]);
                return;
            }
            Real v[3];
            for (int q = 0; q < 3; ++q) {
                const int n = slots[q];
                Real w = pcc(i, j, k, n);
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    const int oi = (d == 0), oj = (d == 1), ok = (d == 2);
                    w += (pcc(i + oi, j + oj, k + ok, n) - Real(2.0) * pcc(i, j, k, n) + pcc(i - oi, j - oj, k - ok, n)) / Real(24.0);
                }
                v[q] = Uavg[n] + w;
            }
            Real Ut[NVAR];
            for (int n = 0; n < NVAR; ++n) Ut[n] = Uavg[n];
            for (int q = 0; q < 3; ++q) Ut[slots[q]] = v[q];
            bool ok = std::isfinite(v[0]) && v[0] > Real(ps_const::ALPHA_ROUNDOFF) && v[0] < Real(1.0) - Real(ps_const::ALPHA_ROUNDOFF)
                   && std::isfinite(v[1]) && std::isfinite(v[2]) && ps_pt_detail::is_state(Ut, l_pres);
            if (!ok) {
                ps_pt_detail::project_local(Uavg, eos, l_pres, tol_P_rel);
                for (int q = 0; q < 3; ++q) Ut[slots[q]] = Uavg[slots[q]];
                PS_COUNT(++ps_counters().proj_pt_raw);
            }
            for (int q = 0; q < 3; ++q) Un(i, j, k, slots[q]) = Ut[slots[q]];
#if !defined(AMREX_USE_GPU)
            if (ps_pres_diag() != 0) {
                const PsCellState st = ps_cell_state(Ut, l_pres);
                if (st.rg1 != PsRegime::Absent && st.rg2 != PsRegime::Absent && st.P_mix > Real(0.0)) {
                    const Real r = std::abs(st.P1 - st.P2) / st.P_mix;
                    if (r > ps_counters().proj_pt_resid) ps_counters().proj_pt_resid = r;
                }
            }
#endif
        });
    }
}

// -----------------------------------------------------------------
//  ps_apply_relaxation — the relaxation sweep.  Builds the EOS callback
//  once, resets the per-sweep counters, dispatches every cell to the
//  kernel for ps_relax_mode, then prints the gated reports ([PS-PRES],
//  [PS-GATE], [PS-DT], [PS-X3], and the class census [PS-CLASS]).
//  Called from apply_ps_reaction under PSAMR.ps_do_relax; do_print is
//  accepted for the Strang pre-half step, which prints nothing else.
//  Every dial it needs is resolved here and captured by value.
//
//  Launch forms (PSAMR.ps_class_compact, PS_classes.H):
//    0  one ParallelFor over every cell of every box: corridor projection,
//       then the mode's gated wrapper (the whole-box form);
//    1  per box, the class pass (projection, then ps_relax_entry_gate into
//       the RELAX bit), the compact list of the admitted cells, and the
//       mode's _admitted operator launched over the list only;
//    2  the class pass, then the whole-box launch gated on the class byte
//       (the measurement arm).
//  All three evaluate the gate once per cell on the projected state and
//  run the operator on the same cells in the same order, so they are
//  bitwise identical and the host counters agree (docs/FUTURE_WORK.md §12).
// -----------------------------------------------------------------
inline void
ps_apply_relaxation(amrex::MultiFab& S_new, amrex::Real dt = amrex::Real(0),
                    int nghost = 0, bool do_print = true)
{
    // EOS-heavy (branch-locked solves per phase per cell): the dominant
    // physics cost of PSAMR_advance.
    BL_PROFILE("PS::ps_apply_relaxation()");

    const hem::PsPhaseAPI eos = ps_make_psamr_eos_api();

    ps_relax_reset_sweep_counters();
    const PsPres l_pres = ps_presence_params();

    // Dispatch on ps_relax_mode (documented at its accessor).
    const int relax_mode = ps_relax_mode();
    const amrex::Real theta = ps_theta_tau();
    const int pres_diag = ps_pres_diag();

    //  Mode-5 and mode-6 operator settings, resolved once from the dials.
    const hem::PsX3Params  xp = ps_x3_params_from_dials();
    const hem::PsX3fParams xf = ps_x3f_params_from_dials();

    const int form = ps_class_compact();
    //  The admitted-cell total of the class pass (forms 1 and 2), -1 when
    //  not formed: it is the one census that exists on every build, and
    //  it costs one device sync per sweep, so it is taken under
    //  ps_pres_diag only.
    amrex::Long n_relax = -1;
    amrex::Long n_cells = 0;

    if (form == 0) {
        for (amrex::MFIter mfi(S_new); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.growntilebox(nghost);
            amrex::Array4<amrex::Real> const& U = S_new.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                ps_corridor_project(i, j, k, U, l_pres);
                if (relax_mode == 5) {
                    ps_x3_grid_relax_cell(i, j, k, U, eos, l_pres, xp, pres_diag, dt);
                } else if (relax_mode == 6) {
                    ps_x3f_grid_relax_cell(i, j, k, U, eos, l_pres, xf, pres_diag, dt);
                } else if (relax_mode == 2) {
                    ps_ptg_relax_cell(i, j, k, U, eos, l_pres, dt, theta);
                } else {
                    ps_mechanical_relax_cell(i, j, k, U, eos, l_pres);
                }
            });
        }
    } else {
        const bool want_census = (pres_diag != 0);
        PsClassLists ws(S_new, nghost);
        n_cells = ws.ncells_total();
        for (amrex::MFIter mfi(S_new); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.growntilebox(nghost);
            amrex::Array4<amrex::Real> const& U = S_new.array(mfi);
            unsigned char* AMREX_RESTRICT cls   = ws.cls_ptr(mfi);
            int*           AMREX_RESTRICT list  = ws.list_ptr(mfi);
            int*           AMREX_RESTRICT count = ws.count_ptr(mfi);
            const int ncell = ws.ncell(mfi);

            //  The class pass: the corridor projection first (it can move a
            //  (Corridor, Independent) cell into the band), then the entry
            //  gate, counted exactly as the whole-box form counts it.  Its
            //  own profile region (host-exact; on the device read it with
            //  tiny_profiler.device_synchronize_around_region=1): the
            //  projection's EOS probe on every (Corridor, Independent) cell
            //  is the part of this sweep that compaction cannot remove.
            {
                BL_PROFILE("PS::ps_relax_classify()");
                ps_classify_box(bx, cls, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> unsigned char
                {
                    ps_corridor_project(i, j, k, U, l_pres);
                    amrex::Real a1, m1, m2;
                    return ps_relax_entry_gate(i, j, k, U, l_pres, a1, m1, m2)
                           ? ps_class::RELAX : ps_class::NONE;
                });
            }

            //  The operator on an admitted cell: the gate's reads repeated
            //  (nothing has written the cell since the class pass), no gate.
            auto relax_admitted = [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const amrex::Real alpha1_in = U(i,j,k, UALPHA1);
                const amrex::Real m1 = U(i,j,k, UM1RHO1);
                const amrex::Real m2 = U(i,j,k, UM2RHO2);
                if (relax_mode == 5) {
                    ps_x3_grid_relax_admitted(i, j, k, U, eos, alpha1_in, m1, m2, xp, pres_diag, dt);
                } else if (relax_mode == 6) {
                    ps_x3f_grid_relax_admitted(i, j, k, U, eos, alpha1_in, m1, m2, xf, pres_diag, dt);
                } else if (relax_mode == 2) {
                    ps_ptg_relax_admitted(i, j, k, U, eos, alpha1_in, m1, m2, dt, theta);
                } else {
                    ps_mechanical_relax_admitted(i, j, k, U, eos, alpha1_in, m1, m2);
                }
            };

            {
                BL_PROFILE("PS::ps_relax_launch()");   // the scan and the operator over the admitted cells
                if (form == 2) {
                    ps_class_launch_box(bx, cls, ps_class::RELAX, relax_admitted);
                    if (want_census) ps_class_list_box(ncell, cls, ps_class::RELAX, list, count);
                } else {
                    ps_class_list_box(ncell, cls, ps_class::RELAX, list, count);
                    ps_class_launch_list(bx, list, count, relax_admitted);
                }
            }
        }
        if (want_census) n_relax = ws.total();
    }

    ps_relax_report(n_relax, n_cells, form);
    amrex::ignore_unused(do_print, n_cells);
}

#endif // USE_PS_HYDRO
#endif // PS_RELAX_H
