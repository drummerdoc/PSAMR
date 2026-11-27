#include "PSAMR.H"
#include <limits>
#include "IndexDefines.H"
#include <iomanip>
#include <string>
#ifdef USE_PS_HYDRO
#include "PS_relaxation.H"
#include "PS_sources.H"
#include "PS_validate.H"   // gated invariant tripwire (PSAMR.ps_validate)
#include "PS_cell_trace.H"  // ps_cell_trace (PSAMR.ps_trace_cell; follow-up 24)
#endif

#include <cmath>
#include <AMReX_ParallelDescriptor.H>        // reductions in the PS-MASS probe
#include <AMReX_iMultiFab.H>                 // the IMEX stage status (ps_imex=3)
#ifdef USE_PS_HYDRO
#include "Hydro/PelantiShyue/PS_counters.H"  // PsCounters (guard / face / fold audits)
#include "Hydro/PelantiShyue/PS_dials.H"     // ps_dial_int (PSAMR.* reads)
#include "Hydro/PelantiShyue/PS_hllc.H"      // PS_FL_NCAUSE / PS_FL_OK (face-audit labels)
#include "Hydro/PelantiShyue/PS_umeth.H"    // dial accessors
#endif

using std::string;

using namespace amrex;

amrex::Real
PSAMR::advance(
  amrex::Real time, amrex::Real dt, int amr_iteration, int amr_ncycle)
{
  // The main driver for a single level implementing the time advance.
  //        @param time the current simulation time
  //        @param dt the timestep to advance (e.g., go from time to time + dt)
  //        @param amr_iteration where we are in the current AMR subcycle.  Each
  //                        level will take a number of steps to reach the
  //                        final time of the coarser level below it.  This
  //                        counter starts at 1
  //        @param amr_ncycle  the number of subcycles at this level

    BL_PROFILE("PSAMR::advance()");

    if (do_mol) {
        amrex::Print() << "Doing MOL Advance" << std::endl;
#ifdef USE_PS_HYDRO
    } else if (ps_hydro != 0) {
        // Pelanti-Shyue six-equation branch: wp is the only flux, second
        // order via the limited correction fluxes from raw cell averages.
        // The banner names the resolved flux through its single-read
        // accessor, so it cannot assert a flux that did not run.
        amrex::Print() << "Doing PS Advance (flux=" << ps_flux_name()
                       << ")" << std::endl;
#endif  // USE_PS_HYDRO
    } else {
        amrex::Print() << "Doing Godunov Advance" << std::endl;
    }

    amrex::Real dt_new;
    dt_new = PSAMR_advance(time, dt, amr_iteration, amr_ncycle);

    return dt_new;
}

amrex::Real
PSAMR::PSAMR_advance (Real time,
                    Real dt,
                    int  amr_iteration,
                    int  amr_ncycle)

  // Advance the solution at one level
  //
  // arguments:
  //    time          : the current simulation time
  //    dt            : the timestep to advance (e.g., go from time to
  //                    time + dt)
  //    amr_iteration : where we are in the current AMR subcycle.  Each
  //                    level will take a number of steps to reach the
  //                    final time of the coarser level below it.  This
  //                    counter starts at 1
  //    amr_ncycle    : the number of subcycles at this level

{
    BL_PROFILE("PSAMR::PSAMR_advance()");

    Real dt_new = dt;

    int finest_level = parent->finestLevel();

    if (level < finest_level && do_reflux) {

        getFluxReg(level + 1).reset();

    }

    // Swap time levels before calling advance
    for (int i = 0; i < num_state_type; ++i) {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    // Ensure data is valid before beginning advance. This addresses
    // the fact that we may have new data on this level that was interpolated
    // from a coarser level, and the interpolation in general cannot be
    // trusted to respect the consistency between certain state variables
    // (e.g. UEINT and UEDEN) that we demand in every zone.

    clean_state(get_old_data(State_Type), 0, false);   // UTEMP refreshed at the end of the step

    MultiFab& S_old = get_old_data(State_Type);
    amrex::ignore_unused(S_old);

    MultiFab& S_new = get_new_data(State_Type);

    // Do the advance.

    const Real prev_time = state[State_Type].prevTime();
    const Real  cur_time = state[State_Type].curTime();

#ifdef USE_PS_HYDRO
    // Pelanti-Shyue reaction operator (resyncs, folds, floor, relaxation,
    // finite-rate sources; docs/MODEL_AND_ALGORITHM.md §8.5) and its
    // splitting controls.
    //   PSAMR.ps_do_relax (1): run the relaxation operator
    //     (ps_apply_relaxation, the whole mechanical/thermal/mass-transfer
    //     kernel selected by ps_relax_mode); 0 = frozen, for no-relaxation
    //     tests.
    //   PSAMR.ps_strang (0; 1 under the order-4 configuration, 0 under
    //     ps_imex=3): 0 = Lie (the whole reaction step after the hydro,
    //     first order in time); 1 = Strang (half steps before and after,
    //     second order with do_mol=1).  DESIGN H-8.
    //   PSAMR.ps_class_compact (1): the launch form of the reaction step's
    //     per-cell operators (PS_classes.H): 1 = one class pass and one
    //     compact launch per operator over the admitted cells only; 0 = the
    //     whole-box launches (the A/B arm); 2 = the class pass with the
    //     whole-box launch (the measurement arm).  Bitwise identical by
    //     construction; resolved here so job_info records it.
    static const int ps_do_relax_cached = []() -> int {
        const int v = ps_dial_int("ps_do_relax", 1);   // resolved value -> job_info
        (void) ps_class_compact();                      // resolved value -> job_info
#ifdef USE_PS_HYDRO
        //  At mode 5 the X3 kernel inside ps_apply_relaxation owns mass
        //  transfer and the split MT source in ps_apply_sources is gated
        //  out, so ps_do_relax=0 with ps_mt_tau>0 would leave no operator
        //  running it.  Refuse the contradictory configuration.
        if (v == 0 && ps_relax_owns_mt() &&
            ps_source_dials().mt_tau > amrex::Real(0.0)) {
            amrex::Abort("PSAMR.ps_do_relax=0 with ps_relax_mode=5/6 and "
                         "ps_mt_tau>0: mode 5's X3 kernel owns mass transfer "
                         "and is disabled, so no operator would run it. "
                         "Set ps_mt_tau=0 (A/C-style no-phase-change run) or "
                         "re-enable ps_do_relax.");
        }
#endif
        return v;
    }();
    //   The default of the two splitting dials under the order-4
    //   configuration (ps_wp_order=4 with rk_order=3; the dial read without
    //   adding back, PS_umeth.cpp owns and validates ps_wp_order).
    static const int ps_order4_default = []() -> int {
        int wpo = 2; amrex::ParmParse("PSAMR").query("ps_wp_order", wpo);
        return (wpo == 4 && rk_order == 3) ? 1 : 0; }();
    //   PSAMR.ps_rk_project (0; 1 under the order-4 configuration): under
    //     rk_order=3, 1 = each stage state is folded, mass-resynced and
    //     projected onto mechanical equilibrium before it feeds the next
    //     stage (every stage a convex combination of admissible forward
    //     Euler steps; DESIGN H-8); 0 = the stages carry the fold only.
    static const int ps_rk_project_cached = []() -> int {
        const int v = ps_dial_int("ps_rk_project", ps_order4_default);
        if (v != 0 && v != 1) { amrex::Abort("PSAMR.ps_rk_project must be 0 or 1"); }
        return v; }();
    //   PSAMR.ps_rk_project_pt (0): under ps_rk_project=1, 1 = the stage
    //     projection acts on the cell's point value and its result is
    //     converted back to an average (the conversion pair around the
    //     mode-0 kernel, ps_apply_mechanical_projection_pt; one more
    //     stage fill per stage); 0 = on the average, as before.  Not
    //     combined with ps_imex=3.  docs/PLAN_high_order.pdf follow-up 13.
    static const int ps_rk_project_pt_cached = []() -> int {
        const int v = ps_dial_int("ps_rk_project_pt", 0);
        if (v != 0 && v != 1) { amrex::Abort("PSAMR.ps_rk_project_pt must be 0 or 1"); }
        if (v == 1 && ps_rk_project_cached != 1) { amrex::Abort("PSAMR.ps_rk_project_pt=1 needs ps_rk_project=1"); }
        return v; }();
    //   PSAMR.ps_rk_project_tol (1e-10): the stage projection's Newton
    //     stopping tolerance on |P1-P2|/max(P1,P2), the X3F residual
    //     tolerance by default (the mode-0 kernel's own 1e-4 floors the
    //     self-convergence; docs/PLAN_high_order.pdf follow-up 1).
    static const amrex::Real ps_rk_project_tol_cached = []() -> amrex::Real {
        const amrex::Real v = ps_dial_real("ps_rk_project_tol", amrex::Real(1.0e-10));
        if (!(v > amrex::Real(0.0)) || v >= amrex::Real(1.0)) {
            amrex::Abort("PSAMR.ps_rk_project_tol must lie in (0, 1)");
        }
        return v; }();
    //   PSAMR.ps_imex (0): 3 = the IMEX pairing IMEX-SSP3(4,3,3): the X3F
    //     kernel's backward-Euler solves inside the stages, the stage
    //     transfers combined in the reaction step with a per-cell fallback,
    //     the flash once after (MODEL 5.1, DESIGN R-18).  Needs do_mol=1,
    //     rk_order=3, rk_stages=3, ps_rk_project=1, ps_hydro=1 and
    //     ps_relax_mode 6 (a finite ps_theta_tau) or 0; sets ps_strang=0.
    //     Under ps_do_relax=0 the stages carry the projection only.
    static const int ps_imex_cached = [&]() -> int {
        const int v = ps_dial_int("ps_imex", 0);
        if (v != 0 && v != 3) { amrex::Abort("PSAMR.ps_imex must be 0 or 3 (IMEX-SSP3(4,3,3))"); }
        if (v == 3) {
            amrex::ParmParse pp("PSAMR");
            int sv = 0;  pp.query("ps_strang", sv);
            const int mode = ps_relax_mode();
            if (do_mol == 0 || rk_order != 3 || rk_stages != 3 || ps_hydro == 0) {
                amrex::Abort("PSAMR.ps_imex=3 needs do_mol=1, rk_order=3, rk_stages=3 and ps_hydro=1 (the explicit part is SSPRK(3,3))");
            }
            if (ps_rk_project_cached != 1) { amrex::Abort("PSAMR.ps_imex=3 needs ps_rk_project=1"); }
            if (ps_rk_project_pt_cached != 0) { amrex::Abort("PSAMR.ps_imex=3 is not combined with ps_rk_project_pt=1"); }
            if (mode != 6 && mode != 0) { amrex::Abort("PSAMR.ps_imex=3 needs ps_relax_mode 6 or 0 (modes 2 and 5 are not backward-Euler stage solves)"); }
            if (mode == 6 && !(ps_theta_tau() > amrex::Real(0.0))) { amrex::Abort("PSAMR.ps_imex=3 needs a finite ps_theta_tau (the stage transfers are rate integrals)"); }
            if (sv == 1) { amrex::Abort("PSAMR.ps_imex=3 sets ps_strang=0; do not set ps_strang=1 with it"); }
        }
        return v; }();
    static const int ps_strang_cached = ps_dial_int("ps_strang", (ps_imex_cached != 0) ? 0 : ps_order4_default);
    // PSAMR.ps_temp_refresh (0): 0 = the UTEMP diagnostic slot refreshed at the
    // end of every step, as before; 1 = refreshed where it is read, before a
    // plotfile, a small plotfile or a checkpoint is written (PSAMR_io.cpp), so
    // the end-of-step clean_state is intermediate (docs/PLAN_hygiene.md item 1b).
    static const int ps_temp_refresh_cached = ps_dial_int("ps_temp_refresh", 0);
    // PSAMR.ps_corridor_cache (0): 1 = the corridor projection's own-branch
    // probe is skipped where the cell's (rho_k, e_k, k) is bitwise the
    // triple of its last kept probe (docs/PLAN_hygiene.md item 3).
    static const int ps_corridor_cache_cached = ps_dial_int("ps_corridor_cache", 0);
    // PSAMR.ps_diag_alpha (accessor in PS_diag.H): report the alpha_1
    // range at labelled points so a drive toward a pure phase can be
    // attributed to mass transfer (jump across the relax bracket) or to
    // advection/C-F (high on entry).
    auto diag_a1 = [&](const amrex::MultiFab& S, const char* label) {
        if (ps_diag_alpha() == 0) return;
        amrex::Print() << "[PS-DIAG] L" << level << " step "
                       << parent->levelSteps(level) << " " << label
                       << ": alpha1 in [" << S.min(UALPHA1, 0) << ", "
                       << S.max(UALPHA1, 0) << "]\n";
    };
    // ---- Mass-consistency probe (PSAMR.ps_diag_mass=1) --------------------
    // The state stores mixture mass twice, URHO and UM1RHO1 + UM2RHO2, and
    // the hydro lets them drift.  The probe reports the drift, and the guard
    // and fold counters, at each stage of the reaction sequence so the
    // responsible operator can be named.  Reads the state, changes nothing.
    // See docs/MODEL_AND_ALGORITHM.md §8.6 for every line printed here.
#ifdef PSAMR_PS_DIAG
#define PS_CELL_PROBE(S,L) cell_probe((S),(L))
    // PSAMR.ps_cell_diag (-1 = off): dump one cell's PS state at every stage
    // of the reaction sequence.  Reads only; not nested in the ps_diag_mass
    // gate.
    auto cell_probe = [&](const amrex::MultiFab& S, const char* label) {
#ifdef USE_PS_HYDRO
        static const int icd = ps_dial_int("ps_cell_diag", -1);
        if (icd < 0) return;
        for (amrex::MFIter mfi(S); mfi.isValid(); ++mfi) {
            const amrex::Box& pbx = mfi.validbox();
            if (pbx.smallEnd(0) > icd || pbx.bigEnd(0) < icd) continue;
            auto const& sa = S.const_array(mfi);
            const amrex::Real a1 = sa(icd,0,0,UALPHA1);
            const amrex::Real m1 = sa(icd,0,0,UM1RHO1);
            const amrex::Real m2 = sa(icd,0,0,UM2RHO2);
            const amrex::Real rho = sa(icd,0,0,URHO);
            const amrex::Real rhoe = sa(icd,0,0,UEINT);
            const amrex::Real mx = sa(icd,0,0,UMX);
            amrex::Print() << "[CELL " << icd << "] step "
                           << parent->levelSteps(level) << " | " << label
                           << " a1=" << a1 << " m1=" << m1 << " m2=" << m2
                           << " rho1=" << (a1 > 0.0 ? m1/a1 : -1.0)
                           << " rho2=" << ((1.0-a1) > 0.0 ? m2/(1.0-a1) : -1.0)
                           << " rho=" << rho << " m1+m2=" << (m1+m2)
                           << " rhoe=" << rhoe
                           << " u=" << (rho != 0.0 ? mx/rho : 0.0) << "\n";
        }
#endif
    };
#else
#define PS_CELL_PROBE(S,L) ((void)0)
#endif  // PSAMR_PS_DIAG
    // PSAMR.ps_trace_cell (i j k; off when absent), ps_trace_level (0): one
    // cell's PS state printed at every stage, every post-stage operator and
    // every label of the step, with each stage's rate scaled by its step
    // (follow-up 24; PS_cell_trace.H, shared with the coarse-fine sequence
    // in PSAMR.cpp).  Host builds only; reads only.
    const bool tracing_cell = ps_trace_cell_on(level);
    const amrex::IntVect trace_iv = ps_trace_cell_iv();
    auto cell_trace = [&](const amrex::MultiFab& S, const char* label) {
        ps_cell_trace(S, level, parent->levelSteps(level), label);
    };
    auto rate_trace = [&](const amrex::MultiFab& K, const amrex::MultiFab& S, amrex::Real fac, const char* label) {
#if defined(USE_PS_HYDRO) && !defined(AMREX_USE_GPU)
        if (!tracing_cell) return;
        for (amrex::MFIter mfi(K); mfi.isValid(); ++mfi) {
            if (!mfi.validbox().contains(trace_iv)) continue;
            auto const& ka = K.const_array(mfi); auto const& sa = S.const_array(mfi);
            const amrex::Real a1 = sa(trace_iv, UALPHA1), m1 = sa(trace_iv, UM1RHO1), E1 = sa(trace_iv, UE1), m2 = sa(trace_iv, UM2RHO2);
            const amrex::Real da = fac * ka(trace_iv, UALPHA1), dm1 = fac * ka(trace_iv, UM1RHO1), dE1 = fac * ka(trace_iv, UE1), dm2 = fac * ka(trace_iv, UM2RHO2);
            amrex::AllPrint() << std::setprecision(6) << "[CELL-TRACE] step " << parent->levelSteps(level) << " " << label
                              << ": d_alpha1=" << da << " (" << (a1 != 0.0 ? da / a1 : 0.0) << " of a1)"
                              << " d_m1=" << dm1 << " (" << (m1 != 0.0 ? dm1 / m1 : 0.0) << " of m1)"
                              << " d_E1=" << dE1 << " (" << (E1 != 0.0 ? dE1 / E1 : 0.0) << " of E1)"
                              << " d_m2=" << dm2 << " (" << (m2 != 0.0 ? dm2 / m2 : 0.0) << " of m2)\n";
        }
#else
        amrex::ignore_unused(K, S, fac, label);
#endif
    };
    auto diag_mass = [&](amrex::MultiFab& S, const char* label, bool pre_repair = false) {
        cell_trace(S, label);
        // Invariant tripwire (PSAMR.ps_validate, 0 = off), independent of
        // the ps_diag_mass gate.  pre_repair marks the stages before the
        // floor/fold/clean repair (H, A, A2), where a round-off alpha_1
        // outside [0, 1] or a negative partial mass is counted, not aborted on.
        ps_validate_state(S, label, 0, pre_repair);
        // PSAMR.ps_diag_mass (0): the probe's gate.
        static const int dgm = ps_dial_int("ps_diag_mass", 0);
        if (dgm == 0) return;
        //  Per-cell contributions, each >= 0 (the extrema start at 0 and
        //  take finite values only), reduced max / max / max / max / sum.
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax,
                         amrex::ReduceOpMax, amrex::ReduceOpSum> mass_op;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                          amrex::Long> mass_data(mass_op);
        using MassTuple = typename decltype(mass_data)::Type;
        for (amrex::MFIter mfi(S); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.validbox();
            amrex::Array4<const amrex::Real> const& U = S.const_array(mfi);
            mass_op.eval(bx, mass_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> MassTuple
            {
                const amrex::Real r  = U(i,j,k, URHO);
                const amrex::Real m1 = U(i,j,k, UM1RHO1);
                const amrex::Real m2 = U(i,j,k, UM2RHO2);
                const amrex::Real s  = m1 + m2;
                // mass / specific-energy extrema: catch creation, not just drift
                const amrex::Real mc = (std::isfinite(s) && s > 0.0) ? s : 0.0;
                amrex::Real e1c = 0.0, e2c = 0.0;
                if (m1 > 0.0) { const amrex::Real e = std::abs(U(i,j,k, UE1)) / m1;
                                if (std::isfinite(e)) e1c = e; }
                if (m2 > 0.0) { const amrex::Real e = std::abs(U(i,j,k, UE2)) / m2;
                                if (std::isfinite(e)) e2c = e; }
                if (!std::isfinite(r) || !std::isfinite(s) || std::abs(r) <= 0.0)
                    return {0.0, mc, e1c, e2c, 0};
                const amrex::Real rel = std::abs(r - s) / std::abs(r);
                return {rel, mc, e1c, e2c, (rel > 1.0e-3) ? 1 : 0};
            });
        }
        const MassTuple mass_v = mass_data.value(mass_op);
        //  max(0, .) restores the zero start value on a rank with no cells.
        amrex::Real worst = amrex::max(amrex::Real(0.0), amrex::get<0>(mass_v));
        amrex::Real mmax  = amrex::max(amrex::Real(0.0), amrex::get<1>(mass_v));
        amrex::Real e1max = amrex::max(amrex::Real(0.0), amrex::get<2>(mass_v));
        amrex::Real e2max = amrex::max(amrex::Real(0.0), amrex::get<3>(mass_v));
        amrex::Long nbad  = amrex::get<4>(mass_v);
        amrex::ParallelDescriptor::ReduceRealMax(worst);
        amrex::ParallelDescriptor::ReduceRealMax(mmax);
        amrex::ParallelDescriptor::ReduceRealMax(e1max);
        amrex::ParallelDescriptor::ReduceRealMax(e2max);
        amrex::ParallelDescriptor::ReduceLongSum(nbad);
        amrex::Print() << "[PS-MASS] L" << level << " step "
                       << parent->levelSteps(level) << " " << label
                       << ": drift = " << worst << " (n>1e-3 " << nbad
                       << ")  max(m1+m2) = " << mmax
                       << "  max|E1| = " << e1max
                       << "  max|E2| = " << e2max << "\n";
#if !defined(AMREX_USE_GPU)
        // Guard audit: `seen` beside `reject` distinguishes "never reached"
        // from "reached, never trips".  Reported then reset, so the numbers
        // are per interval.
        {
            PsCounters& pc = ps_counters();
            amrex::Long gs = pc.guard_seen;
            amrex::Long gl = pc.guard_reject_low;
            amrex::ParallelDescriptor::ReduceLongSum(gs);
            amrex::ParallelDescriptor::ReduceLongSum(gl);
            amrex::Long rl = pc.rho_clamp_lo;
            amrex::ParallelDescriptor::ReduceLongSum(rl);
            amrex::Print() << "[PS-GUARD] L" << level << " step "
                           << parent->levelSteps(level) << " " << label
                           << ": phase-P seen = " << gs
                           << "  rej_low = " << gl
                           << " | rho_clamp_lo = " << rl;
            //  Ctoprim reference audit (PS_guards.H); ctop_sub = 0 over a
            //  run means no cell reached the host-phase reference.
            amrex::Long ck_ = pc.ctop_seen;
            amrex::Long cs_ = pc.ctop_sub;
            amrex::Long cf_ = pc.ctop_host_floor;
            amrex::ParallelDescriptor::ReduceLongSum(ck_);
            amrex::ParallelDescriptor::ReduceLongSum(cs_);
            amrex::ParallelDescriptor::ReduceLongSum(cf_);
            amrex::Print() << " | ctop_seen = " << ck_
                           << "  ctop_sub = " << cs_
                           << "  ctop_host_floor = " << cf_;
            //  The counted limiter (flux non-finite sanitisation) and the
            //  reflux co-move's three counts report and reset at the same
            //  cadence (PS_guards.H).
            amrex::Long fs_ = pc.flux_sanit;
            amrex::Long ra_ = pc.reflux_absent;
            amrex::Long rs_ = pc.reflux_sliver;
            amrex::Long rb_ = pc.reflux_badrho;
            amrex::ParallelDescriptor::ReduceLongSum(fs_);
            amrex::ParallelDescriptor::ReduceLongSum(ra_);
            amrex::ParallelDescriptor::ReduceLongSum(rs_);
            amrex::ParallelDescriptor::ReduceLongSum(rb_);
            amrex::Print() << " | flux_sanit = " << fs_
                           << "  reflux_absent = " << ra_
                           << "  reflux_sliver = " << rs_
                           << "  reflux_badrho = " << rb_;
            //  The reflux energy co-move's count and largest relative
            //  correction (ps_bl_reflux_e, follow-up 25).
            amrex::Long rem_ = pc.reflux_emove;
            amrex::Real remx_ = pc.reflux_emove_max;
            amrex::ParallelDescriptor::ReduceLongSum(rem_);
            amrex::ParallelDescriptor::ReduceRealMax(remx_);
            amrex::Print() << "  reflux_emove = " << rem_
                           << "  reflux_emove_max = " << remx_;
            //  The second-order limiter's three counts (PS_guards.H, O-31).
            amrex::Long ld_ = pc.lim_slot_denorm;
            amrex::Long lw_ = pc.lim_wave_drop;
            amrex::Long lf_ = pc.lim_face_drop;
            amrex::Long ln_ = pc.lim_slot_floor;
            amrex::Long lz_ = pc.lim_wave_noise;
            amrex::ParallelDescriptor::ReduceLongSum(ln_);
            amrex::ParallelDescriptor::ReduceLongSum(lz_);
            amrex::ParallelDescriptor::ReduceLongSum(ld_);
            amrex::ParallelDescriptor::ReduceLongSum(lw_);
            amrex::ParallelDescriptor::ReduceLongSum(lf_);
            amrex::Long rf_ = pc.recon_fallback;
            amrex::Long rt_ = pc.recon_thinc;
            amrex::Long rm_ = pc.recon_mc_raw;
            amrex::Long rmf_ = pc.recon_mc_face;
            amrex::ParallelDescriptor::ReduceLongSum(rmf_);
            amrex::Long rmb_ = pc.recon_mc_bdry;
            amrex::ParallelDescriptor::ReduceLongSum(rmb_);
            amrex::Long pp_ = pc.proj_pt_raw;
            amrex::Real ppr_ = pc.proj_pt_resid;
            amrex::Real ppd_ = pc.proj_pt_dev;
            amrex::Real ppm_ = pc.proj_pt_move;
            amrex::Long pfi_ = pc.proj_fail_input, pfe_ = pc.proj_fail_eos, pfs_ = pc.proj_fail_slope, pfm_ = pc.proj_fail_maxiter;
            amrex::Long pbl_ = pc.pr_stats.barrier_lo, pbh_ = pc.pr_stats.barrier_hi;
            amrex::Long pel_ = pc.pr_stats.eos_fail_liq, pev_ = pc.pr_stats.eos_fail_vap;
            amrex::Long psl_ = pc.pr_stats.entry_subtrip_liq, psv_ = pc.pr_stats.entry_subtrip_vap;
            amrex::ParallelDescriptor::ReduceLongSum(pp_);
            amrex::ParallelDescriptor::ReduceRealMax(ppr_);
            amrex::ParallelDescriptor::ReduceRealMax(ppd_);
            amrex::ParallelDescriptor::ReduceRealMax(ppm_);
            amrex::ParallelDescriptor::ReduceLongSum(rf_);
            amrex::ParallelDescriptor::ReduceLongSum(rt_);
            amrex::ParallelDescriptor::ReduceLongSum(rm_);
            amrex::Print() << " | lim_slot_denorm = " << ld_
                           << "  lim_wave_drop = " << lw_
                           << "  lim_face_drop = " << lf_
                           << "  lim_slot_floor = " << ln_
                           << "  lim_wave_noise = " << lz_
                           << "  recon_fallback = " << rf_
                           << "  recon_thinc = " << rt_
                           << "  recon_mc_raw = " << rm_
                           << "  recon_mc_face = " << rmf_
                           << "  recon_mc_bdry = " << rmb_
                           << "  proj_pt_raw = " << pp_
                           << "  proj_pt_resid = " << ppr_
                           << "  proj_pt_dev = " << ppd_
                           << "  proj_pt_move = " << ppm_
                           << "  proj_fail(input,eos,slope,iter) = " << pfi_ << "," << pfe_ << "," << pfs_ << "," << pfm_
                           << "  barrier(lo,hi) = " << pbl_ << "," << pbh_
                           << "  eos_fail(liq,vap) = " << pel_ << "," << pev_
                           << "  entry_subtrip(liq,vap) = " << psl_ << "," << psv_;
            //  NSCBC ghost-fill zero-gradient fallbacks by cause
            //  (PS_guards.H); all zero unless NSCBC is enabled on a face.
            amrex::Long nc_ = pc.nscbc_zg_c;
            amrex::Long ns_ = pc.nscbc_zg_sup;
            amrex::Long nt_ = pc.nscbc_zg_slots;
            amrex::Long np_ = pc.nscbc_zg_pack;
            amrex::Long nf_ = pc.nscbc_flash;
            amrex::Long nm_ = pc.nscbc_pmix_fail;
            amrex::Long nfc_ = pc.nscbc_fan_cells;
            amrex::Long nfs_ = pc.nscbc_fan_steps;
            amrex::Long nlk_ = pc.nscbc_locked;
            amrex::Long nbs_ = pc.nscbc_br_sup, nbm_ = pc.nscbc_br_small, ngn_ = pc.nscbc_ghost_nostate;
            amrex::ParallelDescriptor::ReduceLongSum(nlk_);
            amrex::ParallelDescriptor::ReduceLongSum(nbs_);
            amrex::ParallelDescriptor::ReduceLongSum(nbm_);
            amrex::ParallelDescriptor::ReduceLongSum(ngn_);
            amrex::ParallelDescriptor::ReduceLongSum(nc_);
            amrex::ParallelDescriptor::ReduceLongSum(ns_);
            amrex::ParallelDescriptor::ReduceLongSum(nt_);
            amrex::ParallelDescriptor::ReduceLongSum(np_);
            amrex::ParallelDescriptor::ReduceLongSum(nf_);
            amrex::ParallelDescriptor::ReduceLongSum(nm_);
            amrex::ParallelDescriptor::ReduceLongSum(nfc_);
            amrex::ParallelDescriptor::ReduceLongSum(nfs_);
            amrex::Print() << " | nscbc_zg(c=" << nc_
                           << ",sup=" << ns_
                           << ",slots=" << nt_
                           << ",pack=" << np_ << ")"
                           << " nscbc_flash=" << nf_
                           << " pmix_fail=" << nm_
                           << " br_sup=" << nbs_ << " br_small=" << nbm_ << " ghost_nostate=" << ngn_
                           << " ghostP=[" << pc.nscbc_gP_min << "," << pc.nscbc_gP_max << "]"
                           << " ghostU=[" << pc.nscbc_gu_min << "," << pc.nscbc_gu_max << "]"
                           << " small(inflow=" << pc.nscbc_small_inflow << ",cap=" << pc.nscbc_small_cap << ")"
                           << " capratio=[" << pc.nscbc_cap_ratio_min << "," << pc.nscbc_cap_ratio_max << "]"
                           << " uout_min=" << pc.nscbc_uout_min
                           << " fan_cells=" << nfc_
                           << " fan_steps=" << nfs_
                           << " locked=" << nlk_ << "\n";
            pc.reset_guards();
            // Face audit, gated on PSAMR.ps_face_diag (0); the counters
            // themselves cost a few compares per face.  The health lines
            // ([PS-RELAXFB], [PS-PROMOTE], [PS-FLCAUSE]) print under
            // ps_diag_mass alone; their counters reset at the same cadence
            // either way.
            {
                static const int fd = ps_dial_int("ps_face_diag", 0);
                if (fd != 0) {
                    static const char* cn[3] = {"II", "C ", "A "};
                    amrex::Print() << "[PS-FACE] L" << level << " step "
                                   << parent->levelSteps(level) << " " << label;
                    for (int c = 0; c < 3; ++c) {
                        amrex::Long fs = pc.fl_seen[c];
                        amrex::Long ff = pc.fl_fail[c];
                        amrex::Real fm = pc.max_incmis[c];
                        const amrex::Real fm_loc = fm;   // pre-reduce
                        amrex::ParallelDescriptor::ReduceLongSum(fs);
                        amrex::ParallelDescriptor::ReduceLongSum(ff);
                        amrex::ParallelDescriptor::ReduceRealMax(fm);
                        //  Name the face holding the class max (a magnitude
                        //  with no location cannot be attributed).  MAXLOC by
                        //  hand: the lowest rank whose local max equals the
                        //  global max broadcasts its (i,j,k,idir).
                        int mloc[4] = { pc.mis_i[c], pc.mis_j[c],
                                        pc.mis_k[c], pc.mis_dir[c] };
                        if (fm > 0.0) {
                            const int np_ = amrex::ParallelDescriptor::NProcs();
                            int owner = (fm_loc == fm)
                                ? amrex::ParallelDescriptor::MyProc() : np_;
                            amrex::ParallelDescriptor::ReduceIntMin(owner);
                            if (owner < np_ && np_ > 1) {
                                amrex::ParallelDescriptor::Bcast(mloc, 4, owner);
                            }
                        }
                        amrex::Print() << " | " << cn[c]
                                       << " fl:" << fs << "/" << ff
                                       << " incmis=" << fm;
                        if (fm > 0.0) {
                            amrex::Print() << "@(" << mloc[0] << "," << mloc[1]
                                           << "," << mloc[2] << ",d" << mloc[3]
                                           << ")";
                        }
                    }
                    amrex::Print() << "\n";
                }
                //  Faces where the relaxed star construction fell back
                //  to the B.14 form: zero on the 1-D suite; persistent
                //  non-zero in production is a finding.
                {
                    amrex::Long nrf = pc.relax_fb;
                    amrex::ParallelDescriptor::ReduceLongSum(nrf);
                    if (nrf > 0) {
                        amrex::Print() << "[PS-RELAXFB] L" << level
                                       << " step "
                                       << parent->levelSteps(level) << " "
                                       << label << ": B.14 fallbacks = "
                                       << nrf << "\n";
                    }
                }
                //  Checked-promotion refusals and floor legs skipped on
                //  non-Independent phases.  Printed only when non-zero.
                {
                    amrex::Long npr = pc.promote_refuse;
                    amrex::Long nfs = pc.floor_skip;
                    amrex::Long nfp = pc.floor_pre_skip, nfe = pc.floor_eval;
                    amrex::Long nsc = pc.promote_refuse_sc;
                    amrex::Long nstl = pc.promote_subtrip_liq, nstv = pc.promote_subtrip_vap;
                    amrex::ParallelDescriptor::ReduceLongSum(nstl);
                    amrex::ParallelDescriptor::ReduceLongSum(nstv);
                    amrex::ParallelDescriptor::ReduceLongSum(npr);
                    amrex::ParallelDescriptor::ReduceLongSum(nfs);
                    amrex::ParallelDescriptor::ReduceLongSum(nfp);
                    amrex::ParallelDescriptor::ReduceLongSum(nfe);
                    amrex::ParallelDescriptor::ReduceLongSum(nsc);
                    if (npr > 0 || nfs > 0 || nsc > 0 || nstl > 0 || nstv > 0 || nfp > 0 || nfe > 0) {
                        amrex::Print() << "[PS-PROMOTE] L" << level
                                       << " step "
                                       << parent->levelSteps(level) << " "
                                       << label
                                       << ": promote_refuse = " << npr
                                       << "  promote_refuse_sc = " << nsc
                                       << "  promote_subtrip(liq,vap) = " << nstl << "," << nstv
                                       << "  floor_skip = " << nfs
                                       << "  floor_eval = " << nfe << "  floor_pre_skip = " << nfp << "\n";
                    }
                    pc.reset_promote();
                }
                //  Refusal-cause breakdown for the wp fluctuation path;
                //  printed only when something refused.
                {
                    static const char* cz[PS_HLLC::PS_FL_NCAUSE] = {
                        "ok", "face", "ws_denom", "ws_SM", "ws_Pstar",
                        "st_denom", "st_massneg", "st_rho", "st_q",
                        "st_Estar" };
                    amrex::Long tot = 0;
                    amrex::Long cv[PS_HLLC::PS_FL_NCAUSE];
                    for (int c = 0; c < PS_HLLC::PS_FL_NCAUSE; ++c) {
                        cv[c] = pc.fl_cause[c];
                        amrex::ParallelDescriptor::ReduceLongSum(cv[c]);
                        if (c != PS_HLLC::PS_FL_OK) tot += cv[c];
                    }
                    amrex::Long sL = pc.fl_side[0];
                    amrex::Long sR = pc.fl_side[1];
                    amrex::ParallelDescriptor::ReduceLongSum(sL);
                    amrex::ParallelDescriptor::ReduceLongSum(sR);
                    if (tot > 0) {
                        amrex::Print() << "[PS-FLCAUSE] L" << level << " step "
                                       << parent->levelSteps(level) << " "
                                       << label << ": total=" << tot;
                        for (int c = 1; c < PS_HLLC::PS_FL_NCAUSE; ++c) {
                            if (cv[c] > 0) {
                                amrex::Print() << "  " << cz[c] << "=" << cv[c];
                            }
                        }
                        amrex::Print() << "  | star side L=" << sL
                                       << " R=" << sR << "\n";
                    }
                }
                pc.reset_face();
            }
            // Fold-mass audit (same cadence/reset as the guard audit).
            {
                PsFoldAudit& fa = pc.fold;
                long nv = fa.n_vanish, nt = fa.n_tfloor, nx = fa.n_vacuum;
                amrex::Real mv = fa.m_vanish, mt = fa.m_tfloor, mx = fa.m_vacuum;
                long nc = fa.n_cdeg, ne = fa.n_edeg;
                amrex::Real mc = fa.m_cdeg, me = fa.m_edeg;
                long nf1 = pc.pres_f1_demote, nf3 = pc.pres_f3_hyst;
                long nf3s = pc.pres_f3_sc_refuse;
                amrex::ParallelDescriptor::ReduceLongSum(nf3s);
                long nf3t = pc.pres_f3_subtrip;
                amrex::ParallelDescriptor::ReduceLongSum(nf3t);
                amrex::ParallelDescriptor::ReduceLongSum(nv);
                amrex::ParallelDescriptor::ReduceLongSum(nt);
                amrex::ParallelDescriptor::ReduceLongSum(nx);
                amrex::ParallelDescriptor::ReduceLongSum(nc);
                amrex::ParallelDescriptor::ReduceLongSum(ne);
                amrex::ParallelDescriptor::ReduceLongSum(nf1);
                amrex::ParallelDescriptor::ReduceLongSum(nf3);
                amrex::ParallelDescriptor::ReduceRealSum(mv);
                amrex::ParallelDescriptor::ReduceRealSum(mt);
                amrex::ParallelDescriptor::ReduceRealSum(mx);
                amrex::ParallelDescriptor::ReduceRealSum(mc);
                amrex::ParallelDescriptor::ReduceRealSum(me);
                if (nv + nt + nx + nc + ne + nf1 + nf3 + nf3s + nf3t > 0) {
                    amrex::Print() << "[PS-FOLD] L" << level << " step "
                                   << parent->levelSteps(level) << " " << label
                                   << ": vanish n=" << nv << " m=" << mv
                                   << " | tfloor n=" << nt << " m=" << mt
                                   << " | vacuum n=" << nx << " m=" << mx
                                   << " | cdeg n=" << nc << " m=" << mc
                                   << " | edeg n=" << ne << " m=" << me
                                   << " | f1_demote_ev=" << nf1
                                   << " | f3_hyst_ev=" << nf3 << " f3_sc_refuse=" << nf3s << " f3_subtrip=" << nf3t << "\n";
                }
                pc.reset_presence();
                fa.reset();
            }
        }
#endif
    };
    //  The IMEX final weights b_j / alpha (ps_imex=3), set by the stage loop
    //  and consumed by the reaction step with the stage transfers imex_D.
    amrex::GpuArray<amrex::Real,4> imex_bw = {0.0, 0.0, 0.0, 0.0};
    auto apply_ps_reaction = [&](amrex::MultiFab& S, amrex::Real dt_r,
                                 int ng, bool do_print) {
#if defined(USE_PRTAB_EOS) && !defined(AMREX_USE_GPU)
        EOS::prtab_diag::stage() = 1;   // PRTab guard census: calls below belong to the reaction operator
#endif
        diag_mass(S, "A enter (post-hydro/C-F)", true);
        PS_CELL_PROBE(S, "A  enter (post-hydro)   ");
        ps_resync_mixture_mass(S, ng);  // URHO == UM1RHO1 + UM2RHO2 after hydro/C-F
        diag_mass(S, "A2 post mass resync", true);
        PS_CELL_PROBE(S, "A2 post mixture resync ");
        ps_resync_phase_energy(S, ng);
        PS_CELL_PROBE(S, "A3 post phase-E resync ");  // UE1 + UE2 == UEDEN after C-F interp/regrid
        // Folds run before the floor, as in the standalone's clamp_cons6:
        // ps_apply_floor is what raises e_k to meet the temperature floor,
        // and it must see the cells the folds have just rewritten.
        // See docs/MODEL_AND_ALGORITHM.md §8.5.
        ps_apply_vanish_fold(S, ng);
        PS_CELL_PROBE(S, "A5 post VANISH fold    ");
        ps_apply_tfloor_fold(S, ng);          // must follow the vanish fold
        PS_CELL_PROBE(S, "A6 post TFLOOR fold    ");
        ps_apply_floor(S, ng);
        PS_CELL_PROBE(S, "A7 post FLOOR          ");          // positivity floor
        clean_state(S, ng, false);   // intermediate: skip the UTEMP diagnostic sweep
        diag_mass(S, "B post floor/fold/clean");
        PS_CELL_PROBE(S, "B  post clean_state    ");
        diag_a1(S, "reaction pre-relax");   // entering: reflects hydro/advection/C-F
        ps_harvest_cells(S, ng, dt_r);       // PSAMR.ps_harvest_cells: the operator's cell inputs (read-only, host builds)
        if (ps_do_relax_cached != 0) {
            if (ps_imex_cached == 3 && imex_D.ok()) {
                //  IMEX (follow-up 4, step B): the final combination of the
                //  stage transfers with the per-cell fallback, in place of
                //  the relaxation sweep; the flash follows as under Lie.
                ps_apply_imex_sweep(S, imex_D, imex_st, 5, imex_bw, amrex::Real(0.0), dt_r, ps_rk_project_tol_cached);
            } else {
                amrex::MultiFab* ccache = nullptr;
                if (ps_corridor_cache_cached != 0) {
                    if (!corridor_cache.ok() || corridor_cache.boxArray() != S.boxArray()
                        || corridor_cache.DistributionMap() != S.DistributionMap() || corridor_cache.nGrow() < ng) {
                        corridor_cache.define(S.boxArray(), S.DistributionMap(), 3, ng);
                        corridor_cache.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());   // no verdict yet
                    }
                    ccache = &corridor_cache;
                }
                ps_apply_relaxation(S, dt_r, ng, do_print, ccache);   // dt_r for the finite-rate legs
            }
            clean_state(S, ng, false); // intermediate: skip the UTEMP diagnostic sweep
        }
        diag_mass(S, "C post relaxation (P/T/MT)");
        PS_CELL_PROBE(S, "C  post relaxation     ");
        diag_a1(S, "reaction post-relax");  // jump here => MT/relaxation is the driver
        ps_harvest_states(S, ng);            // PSAMR.ps_harvest: EOS-input harvest (read-only, host builds)
        ps_apply_sources(S, dt_r, ng);
        diag_mass(S, "D post sources (flash)");
#ifdef PSAMR_PS_DIAG
        {   // PSAMR.ps_floor_diag (0): per-step census of the EOS repairs.
            // Not nested in ps_diag_mass.
            static const int fc = ps_dial_int("ps_floor_diag", 0);
            if (fc != 0) {
                amrex::Print() << "[PS-FLOOR] step " << parent->levelSteps(level)
                    << "  eos_calls=" << hem::floor_census::n_calls()
                    << "  T_clamp_1K=" << hem::floor_census::n_T_clamp()
                    << "  nonconv_LOCKED=" << hem::floor_census::n_T_nonconv()
                    << "  nonconv_detect=" << hem::floor_census::n_T_nonconv_det()
                    << "  P_floor_inuse=" << hem::floor_census::n_P_floor()
                    << "  P_floor_probe=" << hem::floor_census::n_P_floor_probe()
                    << "  mass_neg=" << ps_counters().mass_neg
                    << "  mass_neg_kg=" << ps_counters().mass_neg_kg
                    << "  mass_nonfinite=" << ps_counters().mass_nonfinite << "\n";
                ps_counters().reset_mass();
                hem::floor_census::reset();
            }
        }
#endif
        PS_CELL_PROBE(S, "D  post sources        ");
        diag_a1(S, "reaction post-sources");// jump here => flash/source is the driver
#if defined(USE_PRTAB_EOS) && !defined(AMREX_USE_GPU)
        EOS::prtab_diag::stage() = 0;
#endif
    };
    // Strang pre-hydro half step, applied to the old-time data in place so
    // that the expand_state FillPatch carries it into the Sborder ghost
    // cells and the do_mol=1 Heun update, which references S_old, sees the
    // relaxed old state.  For the instantaneous projection it is nearly
    // idempotent; for finite-rate sources the two adjacent half steps
    // combine to a full dt between hydro solves.
    if (ps_strang_cached != 0 && ps_hydro != 0) {
        apply_ps_reaction(get_old_data(State_Type), amrex::Real(0.5) * dt,
                          0, false);
    }
#endif

    // For the hydrodynamics update we need to have numGrow() ghost zones available,
    // but the state data does not carry ghost zones. So we use a FillPatch
    // using the state data to give us Sborder, which does have ghost zones.

    expand_state(Sborder, prev_time, numGrow());

    if (do_react) {
      react(Sborder);
    }

    // Initialize all forces to zero
    for (int n = 0; n < src_list.size(); ++n) {
        old_sources[src_list[n]]->setVal(0.0);
        new_sources[src_list[n]]->setVal(0.0);
    }

    // Initialize the new-time data.
    MultiFab::Copy(S_new, Sborder, 0, 0, NVAR, S_new.nGrow());

    // Build sources at t_old, then add them to S_new (under rk_order=3 the
    // stages build them on every stage state instead).
    //
    // The Godunov hydro reads these sources (via sources_for_hydro --> srcQ) in
    // the ghost cells as well, so we must build them over the full numGrow()
    // footprint, using Sborder since the state itself carries no ghost cells.
    // The MOL hydro never builds srcQ, so there we only need the valid region.
    if (!(do_mol && rk_order == 3)) {
        const int ng_src = (do_mol) ? 0 : numGrow();
        for (int n = 0; n < src_list.size(); ++n) {
            construct_old_source(src_list[n], Sborder, time, dt, ng_src);
            MultiFab::Saxpy(S_new, dt, *old_sources[src_list[n]], 0, 0, NVAR, 0);
        }
    }

    //
    // Now build and add the hydro source term(s) to S_new
    //
    if (do_mol && rk_order == 3) {
        //  SSPRK(3,3) (Shu-Osher), SSPRK(4,3) (Spiteri-Ruuth, rk_stages=4) and
        //  the IMEX pairing (ps_imex=3, its explicit part SSPRK(3,3)) as one
        //  loop over a stage table: stage s evaluates k_s = L(S_new) on the
        //  stage state with its ghost cells filled at t_s (rk_fill), then
        //  S_new <- a_s S_old + b_s (S_new + c_s dt k_s), then the post-stage
        //  map, the fold with the mass resync and the mechanical projection
        //  under ps_rk_project (every stage a convex combination of
        //  admissible forward Euler steps, DESIGN H-8) or the next IMEX stage
        //  solve; the last stage is left to the reaction step.  The flux
        //  registers take the stage weights; the old-time sources are
        //  rebuilt on every stage state.  docs/PLAN_high_order.pdf sections
        //  4.3 and 4.4, follow-ups 4 and 6.
        struct Stage { amrex::Real a, b, c, t; };
        static const Stage tab3[3] = { {0.0, 1.0, 1.0, 0.0}, {0.75, 0.25, 1.0, 1.0}, {1.0/3.0, 2.0/3.0, 1.0, 0.5} };
        static const Stage tab4[4] = { {0.0, 1.0, 0.5, 0.0}, {0.0, 1.0, 0.5, 0.5}, {2.0/3.0, 1.0/3.0, 0.5, 1.0}, {0.0, 1.0, 0.5, 0.5} };
        static const amrex::Real wgt3[3] = { 1.0/6.0, 1.0/6.0, 2.0/3.0 };
        static const amrex::Real wgt4[4] = { 1.0/6.0, 1.0/6.0, 1.0/6.0, 0.5 };
        const int ns = rk_stages;
        const Stage* tab = (ns == 4) ? tab4 : tab3;
        const amrex::Real* wgt = (ns == 4) ? wgt4 : wgt3;
        auto fresh = [&](auto const& mf) { return !mf.ok() || mf.boxArray() != grids || mf.DistributionMap() != dmap; };
        amrex::MultiFab* rkk[4] = { nullptr, nullptr, nullptr, nullptr };
        for (int s = 0; s < ns; ++s) {
            amrex::MultiFab& k = (ns == 4) ? rk_k4[s] : rk_k3[s];
            if (fresh(k)) { k.define(grids, dmap, NVAR, 0, amrex::MFInfo(), Factory()); }
            rkk[s] = &k;
        }
        auto rhs = [&](int s, amrex::MultiFab& Sb, amrex::Real t) {
            construct_hydro_source(Sb, *rkk[s], t, dt, wgt[s]);
            for (int n = 0; n < src_list.size(); ++n) {
                construct_old_source(src_list[n], Sb, t, dt, 0);
                MultiFab::Add(*rkk[s], *old_sources[src_list[n]], 0, 0, NVAR, 0);
            }
        };
        bool imex = false;
#ifdef USE_PS_HYDRO
        imex = (ps_imex_cached == 3 && ps_hydro != 0);
        //  IMEX-SSP3(4,3,3) (Pareschi-Russo 2005, Table 6) in Shu-Osher form:
        //  four backward-Euler solves of the kernel over alpha dt, the first
        //  two on the old state, the others on the SSPRK(3,3) stage bases
        //  formed from the relaxed stage states; the kernel's offsets on the
        //  earlier stages' transfers D_j are the tableau's a_ij/alpha
        //  rewritten for those bases (a stage state carries its own implicit
        //  content, which the rewrite subtracts); the final weights likewise.
        //  The transfers never form a grid state.  DESIGN R-18.
        constexpr amrex::Real al = amrex::Real(0.24169426078821);
        constexpr amrex::Real be = amrex::Real(0.06042356519705);
        constexpr amrex::Real et = amrex::Real(0.12915286960590);
        constexpr amrex::Real a43 = amrex::Real(0.5) - be - et - al;
        const amrex::Real w6 = amrex::Real(1.0) / (amrex::Real(6.0) * al);
        const amrex::Real t23 = amrex::Real(2.0/3.0);
        const amrex::GpuArray<amrex::Real,4> coff[5] = {
            { amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0) },
            { amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0) },                       // stage 1: no earlier stage
            { amrex::Real(-1.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0) },                      // stage 2 on U^n: -D_1
            { amrex::Real(1.0), (amrex::Real(1.0) - amrex::Real(2.0) * al) / al, amrex::Real(0.0), amrex::Real(0.0) },   // stage 3 on U_2 + dt k_2
            { be / al, et / al - (amrex::Real(1.0) - al) / (amrex::Real(4.0) * al), a43 / al - amrex::Real(0.25), amrex::Real(0.0) } };   // stage 4 on 3/4 U^n + 1/4 (U_3 + dt k_3)
        if (imex) {
            imex_bw = { -t23 * be / al, w6 - t23 * et / al, w6 - t23 * a43 / al, amrex::Real(4.0) * w6 - t23 };   // the final on 1/3 U^n + 2/3 (U_4 + dt k_4)
            if (fresh(imex_D))  { imex_D.define(grids, dmap, 8, 0, amrex::MFInfo(), Factory()); }
            if (fresh(imex_st)) { imex_st.define(grids, dmap, 1, 0); }
            imex_st.setVal(0);
        }
        auto imex_stage = [&](int s) {
            ps_apply_vanish_fold(S_new, 0);
            ps_resync_mixture_mass(S_new, 0);
            if (ps_do_relax_cached != 0) {
                ps_apply_imex_sweep(S_new, imex_D, imex_st, s, coff[s], al * dt, dt, ps_rk_project_tol_cached);
            } else {
                ps_apply_mechanical_projection(S_new, 0, ps_rk_project_tol_cached);
            }
        };
#endif
        auto post = [&](int s) {
#ifdef USE_PS_HYDRO
            if (ps_hydro == 0) return;
            if (imex) { imex_stage(s + 3); return; }
            ps_apply_vanish_fold(S_new, 0);
            cell_trace(S_new, "post fold");
            if (ps_rk_project_cached != 0) {
                ps_resync_mixture_mass(S_new, 0);
                cell_trace(S_new, "post resync");
                if (ps_rk_project_pt_cached != 0) {
                    //  The point form reads the stage state with its ghosts:
                    //  the fill the next stage would make, made here as well.
                    MultiFab::Copy(Sborder, S_new, 0, 0, NVAR, 0);
                    rk_fill(Sborder, prev_time + tab[s + 1].t * dt, s + 2, amr_iteration, amr_ncycle);
                    ps_apply_mechanical_projection_pt(S_new, Sborder, ps_rk_project_tol_cached);
                } else {
                    ps_apply_mechanical_projection(S_new, 0, ps_rk_project_tol_cached);
                }
                cell_trace(S_new, "post projection");
            }
#else
            amrex::ignore_unused(s);
#endif
        };
        MultiFab::Copy(S_new, S_old, 0, 0, NVAR, 0);
#ifdef USE_PS_HYDRO
        if (imex) {
            //  Stages 1 and 2 on the old state (no hydro: b~_1 = 0, a~_i1 = 0);
            //  stage 1 only where it can produce a transfer.
            if (ps_do_relax_cached != 0 && ps_relax_mode() == 6) {
                imex_stage(1);
                MultiFab::Copy(S_new, S_old, 0, 0, NVAR, 0);
            }
            imex_stage(2);
        }
#endif
        for (int s = 0; s < ns; ++s) {
            const amrex::Real ts = prev_time + tab[s].t * dt;
            //  Sborder holds the old state with its FillPatch ghosts; a later
            //  stage (or the relaxed old state under IMEX) replaces its valid
            //  region, and the coarse-fine ghosts come from rk_fill.
            if (s > 0 || imex) { MultiFab::Copy(Sborder, S_new, 0, 0, NVAR, 0); }
            if (s > 0 || imex || level > 0) { rk_fill(Sborder, ts, s + 1, amr_iteration, amr_ncycle); }
            rhs(s, Sborder, ts);
            { std::string lb = "stage " + std::to_string(s + 1) + " in"; cell_trace(Sborder, lb.c_str());
              lb = "stage " + std::to_string(s + 1) + " rate x c dt"; rate_trace(*rkk[s], Sborder, tab[s].c * dt, lb.c_str()); }
            MultiFab::Saxpy(S_new, tab[s].c * dt, *rkk[s], 0, 0, NVAR, 0);
            if (tab[s].b != amrex::Real(1.0)) { S_new.mult(tab[s].b, 0, NVAR, 0); }
            if (tab[s].a != amrex::Real(0.0)) { MultiFab::Saxpy(S_new, tab[s].a, S_old, 0, 0, NVAR, 0); }
            { std::string lb = "stage " + std::to_string(s + 1) + " combined"; cell_trace(S_new, lb.c_str()); }
            if (s < ns - 1) { post(s); }
        }
        // The next finer level's stage ghosts come from these.
        if (ns == 4) { rk_store_coarse(prev_time, dt, S_old, rk_k4); }
        else         { rk_store_coarse(prev_time, dt, S_old, rk_k3); }

    } else if (do_mol) {
        construct_hydro_source(Sborder, hydro_source, time, dt);

        // S^{n+1,*} = S^n + dt * dSdt^{n}
        MultiFab::Saxpy(S_new, dt, hydro_source, 0, 0, NVAR, 0);
#ifdef USE_PS_HYDRO
        // The stage-1 intermediate reaches the stage-2 flux, so vanished
        // phases are folded on it too (the standalone clamps after both RK
        // stages).  The fold is gated inside the wrapper by
        // PSAMR.ps_presence_vanish.
        if (ps_hydro != 0) { ps_apply_vanish_fold(S_new, 0); }
#endif

        expand_state(Sborder, cur_time, numGrow());
        construct_hydro_source(Sborder, new_hydro_source, cur_time, dt);

        // S^{n+1} = 0.5 * (S^{n} + S^{n+1,*}) + 0.5 * dt * dSdt^{n+1,*}
        MultiFab::LinComb(S_new, 0.5, Sborder, 0, 0.5, S_old, 0, 0, NVAR, 0);
        MultiFab::Saxpy  (S_new, 0.5*dt, new_hydro_source, 0, 0, NVAR, 0);

        // The LinComb above averaged S^{n+1,*} -- which already held
        // dt * old_sources -- against S^n, which does not, so only half of the
        // old-time sources survived.  Put the other half back, so that once the
        // new-source correction (which is only 0.5*(Src_new - Src_old)) is
        // added below the state carries the time-centered
        // 0.5 * dt * (Src_old + Src_new), just as the Godunov branch does.
        for (int n = 0; n < src_list.size(); ++n) {
            MultiFab::Saxpy(S_new, 0.5*dt, *old_sources[src_list[n]], 0, 0, NVAR, 0);
        }

    } else {
        construct_hydro_source(Sborder, hydro_source, time, dt);

        // S^{n+1} = S^n + dt * dSdt^{n+1/2}
        MultiFab::Saxpy(S_new, dt, hydro_source, 0, 0, NVAR, 0);

    }

    // Sync up state after old sources and hydro source.
    //  Validator sample point on the raw post-hydro state, before the floor
    //  and resyncs inside clean_state: every other diag point is downstream
    //  of the floor, so this is the only window where a state computeTemp
    //  later aborts on still exists.  Reads the state, changes nothing.
#ifdef USE_PS_HYDRO
    diag_mass(S_new, "H post-hydro (pre-floor)", true);
#endif
    clean_state(S_new, 0, false);   // intermediate: skip the UTEMP diagnostic sweep


    // "new source" is actually the correction to the old source we've already
    // added; under rk_order=3 the sources entered every stage, so none.
    if (!(do_mol && rk_order == 3)) {
        for (int n = 0; n < src_list.size(); ++n) {
            construct_new_source(src_list[n], time, dt);
            MultiFab::Saxpy(S_new, dt, *new_sources[src_list[n]], 0, 0, NVAR, 0);
            clean_state(S_new, 0, false);   // intermediate: the end-of-step call refreshes UTEMP (PLAN_hygiene item 1a)
        }
    }

#ifdef USE_PS_HYDRO
    // Reaction operator on the post-hydro state (apply_ps_reaction above;
    // PS_relaxation.H, PS_sources.H).  α_1 transport is done in
    // wave-propagation form inside PS_umeth, so no transport correction is
    // applied here.  In Lie mode (ps_strang=0) this is the full-step
    // reaction after hydro; in Strang mode it is the second half step
    // completing R(dt/2)·H(dt)·R(dt/2).  The finite-rate sources default
    // on (PSAMR.ps_flash_tau = PSAMR.ps_mt_tau = 1e-7 s; see PS_sources.H).
    if (ps_hydro != 0) {
        diag_a1(S_new, "post-hydro");    // high here (w/o relax) => advection/C-F, not MT
        const amrex::Real dt_react =
            (ps_strang_cached != 0) ? amrex::Real(0.5) * dt : dt;

        // flash_rate derive: the phase-1 mass actually transferred by the
        // reaction substep, (m1_pre - m1_post)/dt_react, + = evaporation.
        // Measured rather than modelled so near-instant transfer that empties
        // a cell within one step is captured.
        if (!flash_src.ok() ||
            flash_src.boxArray() != grids ||
            flash_src.DistributionMap() != dmap) {
            flash_src.define(grids, dmap, 1, 0);
        }
        amrex::MultiFab m1_pre(grids, dmap, 1, 0);
        amrex::MultiFab::Copy(m1_pre, S_new, UM1RHO1, 0, 1, 0);
        apply_ps_reaction(S_new, dt_react, 0, true);
        amrex::MultiFab::Copy    (flash_src, m1_pre, 0, 0, 1, 0);
        amrex::MultiFab::Subtract(flash_src, S_new, UM1RHO1, 0, 1, 0); // m1_pre - m1_post
        flash_src.mult(amrex::Real(1.0) / dt_react, 0, 1, 0);
    }
#endif

    Sborder.clear();
    Sborder.define(grids, dmap, NVAR, numGrow(), amrex::MFInfo(), Factory());

    if (do_react) {
        react(S_new);
    }

    // Final UTEMP refresh for the completed new-time state.  The intermediate
    // clean_state calls pass refresh_temp=false (the UTEMP sweep is a
    // per-cell EOS inversion the conserved evolution never reads), so one
    // sweep per advance here keeps the plotfile/derive/tagging path current;
    // under PSAMR.ps_temp_refresh=1 that sweep moves to the writers
    // (docs/PLAN_hygiene.md item 1b) and this call is intermediate too.
    clean_state(S_new, 0, ps_temp_refresh_cached == 0);

    return dt_new;
}
