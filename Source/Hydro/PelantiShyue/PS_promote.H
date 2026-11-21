#ifndef PS_PROMOTE_H
#define PS_PROMOTE_H
// =====================================================================
//  PS_promote.H — checked promotion (docs/MODEL_AND_ALGORITHM.md §4.5).
//
//  A phase the classifier ps_regime would call Independent is demoted to
//  Corridor when its (rho_k, e_k) has no root on its own branch.  rho_deg
//  bounds the density quotient already; the missing check is the energy:
//  e_k = UE_k/m_k can sit below the coldest or above the hottest
//  reachable e on the branch, and such a phase, once promoted, drifts to
//  the branch edge and aborts in the first aborting branch-locked query.
//  The check is prevention at the promotion event: the phase stays
//  host-slaved (Corridor); nothing is repaired downstream.
//
//  Adds no constant: the branch's own energy domain is the bound, tested
//  with the non-aborting EOS::REY2PTS_phase_try, which brackets e(T) over
//  [1, 5000] K and reports a sign change (a root) or none.  Applied as an
//  extra condition on each site's existing "Independent -> branch query,
//  else host-slave" dispatch, so a demotion falls into the corridor path
//  with no new control flow.  Refusals are counted; on the 1-D battery
//  they are zero and the check is bit-identical on and off, which is the
//  test that no legitimate promotion is unreachable.
//
//  Dials: none.
//  This header isolates the EOS dependency: PS_presence.H has no EOS
//  include and EOS.H has no presence include.
// =====================================================================
#ifdef USE_PS_HYDRO

#include "PS_presence.H"
#include "PS_counters.H"
#include "PS_util.H"
#include "EOS.H"
#include <AMReX_GpuQualifiers.H>

//  Host-only diagnostic counters (PsCounters, PS_counters.H), reported
//  and reset in PSAMR_advance.cpp:
//    promote_refuse : would-be-Independent phases demoted to Corridor
//                     for energy-unreachability ([PS-PROMOTE]).
//    floor_skip     : per-phase floor legs skipped on a non-Independent
//                     phase (PS_floors.H).

//  Demote an Independent classification to Corridor iff (rho, e) has no
//  root on the requested branch; pass-through when the phase is not
//  Independent.  rho, e are the values the site is
//  about to hand the branch-locked EOS, so the test is exactly the query
//  that would fire.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PsRegime ps_regime_reach(PsRegime rg, amrex::Real rho, amrex::Real e,
                         bool want_liquid, const PsPres& pr) noexcept
{
    if (rg != PsRegime::Independent) return rg;
    amrex::Real Yq[NUM_SPECIES]; ps_pure_species(Yq);
    amrex::Real Pq, Tq, Sq;
    const hem::Phase3 br = want_liquid ? hem::Phase3::Liquid
                                       : hem::Phase3::Vapor;
    if (EOS::REY2PTS_phase_try(rho, e, Yq, br, Pq, Tq, Sq)) {
        //  Counted, not acted on (follow-up 19): an Independent liquid whose
        //  root lies above T_c.  Demoting it was built and refuted the same
        //  day: the production tube is supercritical CO2 at 320 K in the
        //  dense slot by design, and the demotion took every one of its
        //  cells; the slot is the dense phase, liquid or supercritical, and
        //  T_c is not the question.  The count stays as the instrument.
#if !defined(AMREX_USE_GPU)
        if (want_liquid && !(Tq < EOS::T_crit())) ++ps_counters().promote_refuse_sc;
        if (Tq < EOS::T_triple()) { if (want_liquid) ++ps_counters().promote_subtrip_liq; else ++ps_counters().promote_subtrip_vap; }   // follow-up 22
#endif
        return PsRegime::Independent;
    }
#if !defined(AMREX_USE_GPU)
    ++ps_counters().promote_refuse;
#endif
    return PsRegime::Corridor;
}

#endif  // USE_PS_HYDRO
#endif  // PS_PROMOTE_H
