// =====================================================================
//  PS_umeth.cpp  —  Pelanti-Shyue six-equation hydro interior.
//
//  Algorithm: Berger-LeVeque wave propagation (PSAMR.ps_flux=wp, the only
//  interior flux).  ps_wp_face forms the A±ΔQ fluctuations per face from
//  raw cell averages via the HLLC fan of PS_hllc.H; an optional limited
//  correction flux (LeVeque, one scalar limiter per wave) gives second
//  order; optional transverse and viscous terms are added in 2-D/3-D.  See docs/MODEL_AND_ALGORITHM.md §1-§3.
//
//  Contract: conserved slots receive the recovered interface flux
//  F* = ½[(F_L + A⁻) + (F_R − A⁺)] in flx and telescope through
//  hydro_consup; the non-conserved slots {UALPHA1, UE1, UE2} get flx = 0
//  and a per-cell deposit -(A⁺_lo + A⁻_hi)/dx [- ΔF̃/dx] written to dsdt.
//  pdivu is zeroed (pressure work already sits in F[UEDEN]).  Every raw
//  wave satisfies W[URHO] = W[UM1RHO1]+W[UM2RHO2] and W[UEDEN] =
//  W[UE1]+W[UE2]; the correction only ever scales whole waves so these
//  identities survive.  A face whose star state is refused falls back to
//  an identity-consistent local Lax-Friedrichs flux; no state is repaired.
//
//  Dials (PSAMR.*, each read once at its accessor below): ps_flux (wp),
//  ps_wp_order (2), ps_wp_limiter (vanleer), ps_wp_transverse (0),
//  ps_mu (0), ps_lim_ref_eps (1e-8); under ps_wp_order=4 (reconstructed
//  faces and the in-cell term, PS_recon.H, docs/PLAN_high_order.pdf
//  Stage 3): ps_wp_recon (weno5z), ps_wp_contact_taper (0), ps_wp_sharpen (0),
//  ps_wp_mc (0).
// =====================================================================

#include "PS_umeth.H"
#include "IndexDefines.H"
#include <iomanip>

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Utility.H>
#include <AMReX_Print.H>

#ifdef USE_PS_HYDRO

#include "EOS.H"
#include "PS_hllc.H"
#include "PS_presence.H"
#include "PS_promote.H"
#include "PS_guards.H"
#include "PS_ctoprim.H"
#include "PS_state.H"
#include "PS_wavespeed.H"
#include "PS_constants.H"
#include "PS_util.H"
#include "PS_dials.H"
#include "PS_recon.H"

#include <AMReX_ParmParse.H>
#include <limits>

using namespace amrex;

// ---------------------------------------------------------------------
//  ps_physical_flux_from_state: F(U) in direction idir from one cell's
//  conserved array U.  Euler form on the mixture slots, Pelanti-Shyue
//  (2014) eqs. (1)-(4) on the six-equation slots, with the phase-energy
//  work term at the mixture pressure P_mix = α₁P₁ + α₂P₂ (a per-phase
//  pressure there is bit-identical at mechanical equilibrium;
//  docs/DESIGN_DECISIONS.md H-3).  Per-phase (ρ_k, e_k, P_k) and P_mix
//  are the checked cell state ps_cell_state (PS_state.H); the PR
//  (ρ, e) → state cache absorbs the repeated per-face EOS solves.
//  Non-finite flux entries are zeroed and counted.
// ---------------------------------------------------------------------
//  The momentum slot of direction idir.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ps_umom_comp(int idir) noexcept
{
    return (idir == 0) ? UMX
#if (AMREX_SPACEDIM >= 2)
         : (idir == 1) ? UMY
#endif
#if (AMREX_SPACEDIM == 3)
         : UMZ
#else
         : UMX
#endif
         ;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
ps_physical_flux(int idir, const Real U[NVAR], const PsCellState& s, Real F[NVAR]) noexcept
{
    const Real rho     = s.rho;
    Real un = s.ux;
    if      (idir == 1) un = s.uy;
    else if (idir == 2) un = s.uz;

    const Real UEden   = ps_finite_or(U[UEDEN], Real(0.0));
    const Real UEint   = s.rhoe;
    const Real alpha_1 = s.alpha1;
    const Real alpha_2 = s.alpha2;
    const Real P_mix   = s.P_mix;

    for (int n = 0; n < NVAR; ++n) F[n] = Real(0.0);

    F[URHO ] = rho * un;
    F[UMX  ] = rho * un * s.ux + (idir == 0 ? P_mix : Real(0.0));
#if (AMREX_SPACEDIM >= 2)
    F[UMY  ] = rho * un * s.uy + (idir == 1 ? P_mix : Real(0.0));
#endif
#if (AMREX_SPACEDIM == 3)
    F[UMZ  ] = rho * un * s.uz + (idir == 2 ? P_mix : Real(0.0));
#endif
    F[UEDEN] = (UEden + P_mix) * un;
    F[UEINT] = UEint * un;
    F[UTEMP] = Real(0.0);

    for (int n = 0; n < NUM_SPECIES; ++n) {
        F[UFS + n] = ps_finite_or(U[UFS + n], Real(0.0)) * un;
    }
#if (NUM_ADV > 0)
    for (int n = 0; n < NUM_ADV; ++n)
        F[UFA + n] = ps_finite_or(U[UFA + n], Real(0.0)) * un;
#endif
#if (NUM_AUX > 0)
    for (int n = 0; n < NUM_AUX; ++n)
        F[UFX + n] = ps_finite_or(U[UFX + n], Real(0.0)) * un;
#endif

    F[UALPHA1] = alpha_1 * un;
    F[UM1RHO1] = ps_finite_or(U[UM1RHO1], Real(0.0)) * un;
    F[UM2RHO2] = ps_finite_or(U[UM2RHO2], Real(0.0)) * un;
    // Phase-energy work term at the mixture pressure (H-3).
    const Real Pe1 = P_mix;
    const Real Pe2 = P_mix;
    F[UE1    ] = (ps_finite_or(U[UE1], Real(0.0)) + alpha_1 * Pe1) * un;
    F[UE2    ] = (ps_finite_or(U[UE2], Real(0.0)) + alpha_2 * Pe2) * un;

    for (int n = 0; n < NVAR; ++n) {
        if (!std::isfinite(F[n])) { F[n] = Real(0.0); ps_guard::count_flux_sanit(); }
    }
}




// ---------------------------------------------------------------------
//  ps_wp_face: the A±ΔQ fluctuations at one face (nodal index (i,j,k) in
//  direction idir; left cell (iL,jL,kL), right cell (i,j,k)) from raw
//  cell averages.  Writes, on the valid face box only: conserved slots
//  -> recovered flux F* into flx (A⁺+A⁻ = ΔF, so consup's -div(F*)
//  telescopes to the fluctuation update); {UALPHA1,UE1,UE2} -> flx = 0
//  and A⁻/A⁺ into wpf (comps 0/1 alpha, 2/3 UE1, 4/5 UE2).  With
//  store_waves the raw waves and speeds go to wv (may extend into ghost
//  faces).  No reconstruction: face states are cell averages; second
//  order comes from the correction pass (docs/MODEL_AND_ALGORITHM.md §3.1).
//  A refused face takes an identity-consistent local Lax-Friedrichs flux.
// ---------------------------------------------------------------------
//  Under ps_wp_order=4 (recon_faces) the two side states are the
//  reconstructed edge states of PS_recon.H and the face also stores, for
//  the in-cell term of Pass 3, each side's phase-energy physical fluxes,
//  mixture pressure, alpha_1 and normal velocity (wpf comps 6..15).
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
ps_wp_face_core(int idir, int i, int j, int k,
                const amrex::Real UL[NVAR], const amrex::Real UR[NVAR],
                amrex::Array4<amrex::Real> const& flx,
                amrex::Array4<amrex::Real> const& wpf,
                amrex::Array4<amrex::Real> const& wv,
                bool store_waves, bool recon_faces,
                amrex::Box vbox,
                const PsPres& l_pres) noexcept
{
    using amrex::Real;
    const bool in_valid = vbox.contains(amrex::IntVect(AMREX_D_DECL(i,j,k)));
    Real FL[NVAR], FR[NVAR];
    const PsCellState sL = ps_cell_state(UL, l_pres);
    const PsCellState sR = ps_cell_state(UR, l_pres);
    ps_physical_flux(idir, UL, sL, FL);
    ps_physical_flux(idir, UR, sR, FR);
    const Real PL = sL.P_mix, PR = sR.P_mix;

    PS_HLLC::Fluctuations flu;
    const bool ok = PS_HLLC::fluctuations(idir, UL, UR, flu, l_pres,
                                          i, j, k);

    Real Am[NVAR], Ap[NVAR], flx_loc[NVAR];
    if (ok) {
        for (int n = 0; n < NVAR; ++n) {
            Am[n]      = flu.Am[n];
            Ap[n]      = flu.Ap[n];
            // Symmetrised recovered flux F* = ½[(F_L + A⁻) + (F_R − A⁺)]:
            // equals F_L + A⁻ where A⁻ + A⁺ = ΔF holds; where it does not
            // (non-conservative slots) the one-sided form is left-biased
            // and breaks reflection symmetry at mirror faces.
            flx_loc[n] = Real(0.5) * ((FL[n] + Am[n]) + (FR[n] - Ap[n]));
        }
    } else {
        const Real lamL    = ps_max_wave_speed_from_state(idir, UL, l_pres);
        const Real lamR    = ps_max_wave_speed_from_state(idir, UR, l_pres);
        const Real lam_raw = amrex::max(lamL, lamR);
        const Real lam     = std::isfinite(lam_raw) ? lam_raw : Real(0.0);
        for (int n = 0; n < NVAR; ++n) {
            const Real half = Real(0.5) * lam * (UR[n] - UL[n]);
            Am[n]      = -half;
            Ap[n]      =  half;
            flx_loc[n] = Real(0.5) * (FL[n] + FR[n]) - half;   // LLF flux
        }
        //  Identity-consistent split for the non-conserved slots: the
        //  pure-diffusion split above has Am+Ap = 0 on UE1/UE2 while
        //  UEDEN receives the full LLF flux difference, so a refused face
        //  would move total energy but no phase energy.  UE1/UE2 take
        //  Am = ½(ΔF − λΔU), Ap = ½(ΔF + λΔU) (Am+Ap = ΔF); alpha takes the
        //  advective form ū·Δα, ū = ½(u_nL + u_nR), not Δ(α·u_n), which would
        //  re-introduce the spurious α∇·u term; ū is symmetric so mirror
        //  faces keep exact reflection symmetry.  docs §2.6.
        {
            const int nslots[2] = { UE1, UE2 };
            for (int q = 0; q < 2; ++q) {
                const int n = nslots[q];
                const Real dF   = FR[n] - FL[n];
                const Real half = Real(0.5) * lam * (UR[n] - UL[n]);
                Am[n] = Real(0.5) * dF - half;
                Ap[n] = Real(0.5) * dF + half;
            }
            const int mcomp = ps_umom_comp(idir);
            const Real rL  = UL[URHO], rR = UR[URHO];
            const Real unL = (rL > Real(0.0)) ? UL[mcomp] / rL : Real(0.0);
            const Real unR = (rR > Real(0.0)) ? UR[mcomp] / rR : Real(0.0);
            const Real ubar = Real(0.5) * (unL + unR);
            const Real da   = UR[UALPHA1] - UL[UALPHA1];
            const Real halfa = Real(0.5) * lam * da;
            Am[UALPHA1] = Real(0.5) * ubar * da - halfa;
            Ap[UALPHA1] = Real(0.5) * ubar * da + halfa;
        }
#if !defined(AMREX_USE_GPU) && defined(PSAMR_PS_DIAG)
        //  PSAMR.ps_llf_diag (0): print each face where the LLF fallback
        //  fired, with lam decomposed on the left state that sets it.
        {
            static const int lld = ps_dial_int("ps_llf_diag", 0);
            if (lld != 0 && in_valid) {
                const Real a1d = UL[UALPHA1];
                const Real m1d = UL[UM1RHO1], m2d = UL[UM2RHO2];
                const Real rhod = UL[URHO];
                const Real uxd = (rhod != Real(0.0)) ? UL[UMX]/rhod : Real(0.0);
                const Real ked = Real(0.5)*uxd*uxd;
                const Real r1_raw = m1d / amrex::max(a1d, Real(ps_const::DENOM_TINY));
                const Real r1_cl = ps_guard::clamp_phase_density(r1_raw, EOS::rho_min(), EOS::rho_max());
                const Real r2_cl = ps_guard::clamp_phase_density(
                        m2d / amrex::max(Real(1.0)-a1d, Real(ps_const::DENOM_TINY)),
                        EOS::rho_min(), EOS::rho_max());
                const Real e1d = (m1d > ps_const::M_TINY) ? UL[UE1]/m1d - ked : UL[UEINT]/rhod;
                const Real e2d = (m2d > ps_const::M_TINY) ? UL[UE2]/m2d - ked : UL[UEINT]/rhod;
                Real Yd[NUM_SPECIES];
                ps_pure_species(Yd);
                Real P1d, c1d, P2d, c2d;
                EOS::REY2PCs_liquid(r1_cl, e1d, Yd, P1d, c1d);
                EOS::REY2PCs_vapor (r2_cl, e2d, Yd, P2d, c2d);
                const Real Y1d = m1d/rhod;
                amrex::Print() << "[LLF-FALLBACK] i=" << i << " lam=" << lam
                   << " lamL=" << lamL << " lamR=" << lamR
                   << " rhoL=" << UL[URHO] << " rhoR=" << UR[URHO]
                   << " drho=" << (UR[URHO]-UL[URHO]) << "\n"
                   << "[LAM-DECOMP]  a1=" << a1d << " m1=" << m1d << " m2=" << m2d
                   << " rho1_raw=" << r1_raw << " rho1_clamped=" << r1_cl
                   << " rho_max=" << EOS::rho_max()
                   << " e1=" << e1d << " P1=" << P1d << " c1=" << c1d
                   << " rho2=" << r2_cl << " e2=" << e2d << " P2=" << P2d
                   << " c2=" << c2d << " Y1=" << Y1d << "\n";
            }
        }
#endif
    }

    // flx/wpf live on the valid face box only; ghost faces of the grown
    // wave box contribute waves to the correction stencil but must not
    // write flx.
    if (in_valid) {
        for (int n = 0; n < NVAR; ++n) flx(i,j,k,n) = ps_finite_or(flx_loc[n], Real(0.0));
        flx(i,j,k, UTEMP)   = Real(0.0);   // T recomputed at ctoprim
        flx(i,j,k, UALPHA1) = Real(0.0);
        flx(i,j,k, UE1)     = Real(0.0);
        flx(i,j,k, UE2)     = Real(0.0);

        wpf(i,j,k, 0) = ps_finite_or(Am[UALPHA1], Real(0.0));
        wpf(i,j,k, 1) = ps_finite_or(Ap[UALPHA1], Real(0.0));
        wpf(i,j,k, 2) = ps_finite_or(Am[UE1],     Real(0.0));
        wpf(i,j,k, 3) = ps_finite_or(Ap[UE1],     Real(0.0));
        wpf(i,j,k, 4) = ps_finite_or(Am[UE2],     Real(0.0));
        wpf(i,j,k, 5) = ps_finite_or(Ap[UE2],     Real(0.0));
        if (recon_faces) {
            //  The in-cell term's ingredients (Pass 3) of each side state:
            //  F[UE_k] (6..9), P_mix (10, 11), the clamped alpha_1 (12, 13)
            //  and the normal velocity (14, 15).
            Real aL = UL[UALPHA1], aR = UR[UALPHA1];
            if (aL < Real(0.0)) aL = Real(0.0); if (aL > Real(1.0)) aL = Real(1.0);
            if (aR < Real(0.0)) aR = Real(0.0); if (aR > Real(1.0)) aR = Real(1.0);
            wpf(i,j,k, 6)  = ps_finite_or(FL[UE1], Real(0.0));
            wpf(i,j,k, 7)  = ps_finite_or(FL[UE2], Real(0.0));
            wpf(i,j,k, 8)  = ps_finite_or(FR[UE1], Real(0.0));
            wpf(i,j,k, 9)  = ps_finite_or(FR[UE2], Real(0.0));
            wpf(i,j,k, 10) = PL;
            wpf(i,j,k, 11) = PR;
            wpf(i,j,k, 12) = aL;
            wpf(i,j,k, 13) = aR;
            const int mcomp = ps_umom_comp(idir);
            const Real rL = amrex::max(ps_finite_or(UL[URHO], Real(0.0)), Real(ps_const::RHO_MIX_MIN));
            const Real rR = amrex::max(ps_finite_or(UR[URHO], Real(0.0)), Real(ps_const::RHO_MIX_MIN));
            wpf(i,j,k, 14) = ps_finite_or(UL[mcomp], Real(0.0)) / rL;
            wpf(i,j,k, 15) = ps_finite_or(UR[mcomp], Real(0.0)) / rR;
        }
    }

    // Raw waves W[l][n] and speeds s[l] for the correction pass; a refused
    // face stores zero waves so the correction skips it (first order there).
    if (store_waves) {
        for (int l = 0; l < 3; ++l) {
            const Real sl = ok ? flu.s[l] : Real(0.0);
            wv(i,j,k, 3*NVAR + l) = sl;
            for (int n = 0; n < NVAR; ++n)
                wv(i,j,k, l*NVAR + n) = ok ? ps_finite_or(flu.W[l][n], Real(0.0)) : Real(0.0);
        }
    }
}

//  ps_wp_face: the face between the cells (iL,jL,kL) and (i,j,k) from the
//  side states read at src(.., cL0 + n) on the left and src(.., cR0 + n)
//  on the right: the raw cell averages (ps_wp_order 1, 2: src = uin,
//  cL0 = cR0 = 0) or the reconstructed edge states of PS_RECON::cell_edges
//  (ps_wp_order=4: src = the per-cell edge store, cL0 = NVAR the high edge
//  of the left cell, cR0 = 0 the low edge of the right cell).
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
ps_wp_face(int idir, int i, int j, int k, int iL, int jL, int kL,
           amrex::Array4<const amrex::Real> const& src, int cL0, int cR0,
           amrex::Array4<amrex::Real> const& flx,
           amrex::Array4<amrex::Real> const& wpf,
           amrex::Array4<amrex::Real> const& wv,
           bool store_waves, bool recon_faces,
           amrex::Box vbox,
           const PsPres& l_pres) noexcept
{
    using amrex::Real;
    Real UL[NVAR], UR[NVAR];
    for (int n = 0; n < NVAR; ++n) {
        UL[n] = ps_finite_or(src(iL, jL, kL, cL0 + n), Real(0.0));
        UR[n] = ps_finite_or(src(i,  j,  k,  cR0 + n), Real(0.0));
    }
    ps_wp_face_core(idir, i, j, k, UL, UR, flx, wpf, wv, store_waves, recon_faces, vbox, l_pres);
}

// ---------------------------------------------------------------------
//  ps_wp_face_pt (ps_wp_mc=2): the face between the cells (iL,jL,kL) and
//  (i,j,k) from the face point values of the two edge states: each
//  side's primitive edge less a 24th of its transverse second difference
//  (McCorquodale and Colella 2011), mapped to the conserved array after
//  the conversion (the map is nonlinear, so converting after it would
//  keep its second-order term), unless an edge of the three transverse
//  neighbours belongs to a fallback cell or a point state leaves the
//  admissible set, when both sides keep the stored edges (counted
//  recon_mc_face).  The fluctuations it writes are face point values;
//  the caller converts them back to face averages over the valid faces.
// ---------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
ps_wp_face_pt(int idir, int i, int j, int k, int iL, int jL, int kL,
              amrex::Array4<const amrex::Real> const& uin,
              amrex::Array4<const amrex::Real> const& rq,
              amrex::Array4<amrex::Real> const& flx,
              amrex::Array4<amrex::Real> const& wpf,
              amrex::Array4<amrex::Real> const& wv,
              bool store_waves, amrex::Box vbox,
              const PsPres& l_pres) noexcept
{
    using amrex::Real;
    using namespace PS_RECON;
    Real UL[NVAR], UR[NVAR];
    for (int n = 0; n < NVAR; ++n) {
        UL[n] = ps_finite_or(rq(iL, jL, kL, NVAR + n), Real(0.0));
        UR[n] = ps_finite_or(rq(i,  j,  k,  n), Real(0.0));
    }
    // The primitive edges, converted to face point values by their
    // transverse second differences, then mapped to the conserved array
    // with each cell's presence flags and species.
    Real vL[NPRIM], vR[NPRIM];
    for (int m = 0; m < NPRIM; ++m) { vL[m] = rq(iL, jL, kL, RQ_PH + m); vR[m] = rq(i, j, k, RQ_PL + m); }
    //  The face converts only where the six cells it reads are free of
    //  the presence machinery's switches and share one regime pair, and
    //  none took its raw averages (the rule of follow-up 15).
    const Real swL = rq(iL, jL, kL, RQ_SW), swR = rq(i, j, k, RQ_SW);
    bool conv = (rq(iL, jL, kL, RQ_FB) == Real(0.0)) && (rq(i, j, k, RQ_FB) == Real(0.0))
             && swL >= Real(0.0) && swR >= Real(0.0) && swL == swR;
    for (int tdir = 0; tdir < AMREX_SPACEDIM && conv; ++tdir) {
        if (tdir == idir) continue;
        const int oi = (tdir == 0), oj = (tdir == 1), ok = (tdir == 2);
        if (rq(iL + oi, jL + oj, kL + ok, RQ_FB) != Real(0.0) || rq(iL - oi, jL - oj, kL - ok, RQ_FB) != Real(0.0) ||
            rq(i + oi, j + oj, k + ok, RQ_FB) != Real(0.0) || rq(i - oi, j - oj, k - ok, RQ_FB) != Real(0.0)) { conv = false; break; }
        if (rq(iL + oi, jL + oj, kL + ok, RQ_SW) != swL || rq(iL - oi, jL - oj, kL - ok, RQ_SW) != swL ||
            rq(i + oi, j + oj, k + ok, RQ_SW) != swR || rq(i - oi, j - oj, k - ok, RQ_SW) != swR) { conv = false; break; }
        for (int m = 0; m < NPRIM; ++m) {
            vL[m] -= (rq(iL + oi, jL + oj, kL + ok, RQ_PH + m) - Real(2.0) * rq(iL, jL, kL, RQ_PH + m) + rq(iL - oi, jL - oj, kL - ok, RQ_PH + m)) / Real(24.0);
            vR[m] -= (rq(i + oi, j + oj, k + ok, RQ_PL + m) - Real(2.0) * rq(i, j, k, RQ_PL + m) + rq(i - oi, j - oj, k - ok, RQ_PL + m)) / Real(24.0);
        }
    }
    if (conv) {
        Real UcL[NVAR], UcR[NVAR];
        for (int n = 0; n < NVAR; ++n) { UcL[n] = uin(iL, jL, kL, n); UcR[n] = uin(i, j, k, n); }
        const PsPrim PL = prim_from_cons(UcL, l_pres), PR = prim_from_cons(UcR, l_pres);
        auto admissible = [&] (const Real* v, const PsPrim& C) noexcept -> bool {
            if (!(v[IA1] >= Real(0.0) && v[IA1] <= Real(1.0))) return false;
            if (!C.ab1 && (!(v[IA1] > l_pres.alpha_vanish) || !(v[IR1] > Real(0.0)) || !std::isfinite(v[IR1]) || !std::isfinite(v[IE1]))) return false;
            if (!C.ab2 && (!(Real(1.0) - v[IA1] > l_pres.alpha_vanish) || !(v[IR2] > Real(0.0)) || !std::isfinite(v[IR2]) || !std::isfinite(v[IE2]))) return false;
            for (int m = 0; m < AMREX_SPACEDIM; ++m) if (!std::isfinite(v[IU + m])) return false;
            return true;
        };
        if (admissible(vL, PL) && admissible(vR, PR)) {
            auto species = [&] (const Real* U, Real* Y) noexcept {
                const Real rho = amrex::max(ps_finite_or(U[URHO], Real(ps_const::RHO_MIX_MIN)), Real(ps_const::RHO_MIX_MIN));
                for (int n = 0; n < NUM_SPECIES; ++n) Y[n] = ps_finite_or(U[UFS + n], Real(0.0)) / rho;
            };
            Real YL[NUM_SPECIES], YR[NUM_SPECIES];
            species(UcL, YL); species(UcR, YR);
            cons_from_prim(vL, PL.ab1, PL.ab2, YL, UL);
            cons_from_prim(vR, PR.ab1, PR.ab2, YR, UR);
        } else {
            ps_guard::count_recon_mc_face();
        }
    }
    ps_wp_face_core(idir, i, j, k, UL, UR, flx, wpf, wv, store_waves, true, vbox, l_pres);
}

// ---------------------------------------------------------------------
//  ps_wp_tvterm: LeVeque transverse correction (rpt2 form) to the
//  direction-d flux at d-face (i,j,k), accumulated into g[NVAR].  The
//  contact wave (l=1) of the four neighbouring t-face fluctuations is
//  transported in d at the d-velocity of that t-face, upwinded by sign.
//  Its transverse speed is the material velocity, so the term is an exact
//  no-op for flow with no d-velocity (1-D-aligned cases stay bit-identical).
//  Direction-generic: one path for 2-D and all six 3-D (d,t) pairs.
//  Contributions whose perpendicular column is an out-of-domain ghost are
//  dropped.  docs §3.5.
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
ps_wp_tvterm(int d, int t, int i, int j, int k,
             amrex::Array4<const amrex::Real> const& uin,
             amrex::Array4<amrex::Real> const& wv_t,
             const int* domlo, const int* domhi,
             amrex::Real dt, amrex::Real dxt, amrex::Real g[NVAR]) noexcept
{
    using amrex::Real;
    const int di=(d==0), dj=(d==1), dk=(d==2);
    const int ti=(t==0), tj=(t==1), tk=(t==2);
#if (AMREX_SPACEDIM == 3)
    const int UM_d = (d==0)?UMX : (d==1)?UMY : UMZ;
#elif (AMREX_SPACEDIM == 2)
    const int UM_d = (d==0)?UMX : UMY;
#else
    const int UM_d = UMX;   // 1D: transverse term is never invoked
#endif
    // d-velocity at the t-face whose high-side cell is (a,b,c):
    auto ud = [&](int a,int b,int c) noexcept -> Real {
        Real r0=uin(a-ti,b-tj,c-tk,URHO); r0=(std::isfinite(r0)&&r0>ps_const::RHO_MIX_MIN)?r0:Real(ps_const::RHO_MIX_MIN);
        Real r1=uin(a,   b,   c,   URHO); r1=(std::isfinite(r1)&&r1>ps_const::RHO_MIX_MIN)?r1:Real(ps_const::RHO_MIX_MIN);
        return Real(0.5)*(ps_finite_or(uin(a-ti,b-tj,c-tk,UM_d),Real(0.0))/r0
                        + ps_finite_or(uin(a,   b,   c,   UM_d),Real(0.0))/r1); };
    // contact-wave (l=1) A± of the t-fluctuation at t-face (a,b,c):
    auto Apm = [&](int a,int b,int c,int n,int sgn) noexcept -> Real {
        const int l=1; const Real sl=wv_t(a,b,c,3*NVAR+l);
        if((sgn>0&&sl>Real(0.0))||(sgn<0&&sl<Real(0.0))) return sl*wv_t(a,b,c,l*NVAR+n);
        return Real(0.0); };
    const int Id = (d==0)?i : (d==1)?j : k;                 // d-index of the d-face
    const Real mCp = (Id   <= domhi[d] && Id   >= domlo[d]) ? Real(1.0):Real(0.0);
    const Real mCm = (Id-1 <= domhi[d] && Id-1 >= domlo[d]) ? Real(1.0):Real(0.0);
    const Real cP0 = mCp*amrex::min(ud(i,      j,      k     ),Real(0.0));   // Cp low-t
    const Real cP1 = mCp*amrex::min(ud(i+ti,   j+tj,   k+tk  ),Real(0.0));   // Cp high-t
    const Real cM0 = mCm*amrex::max(ud(i-di,   j-dj,   k-dk  ),Real(0.0));   // Cm low-t
    const Real cM1 = mCm*amrex::max(ud(i-di+ti,j-dj+tj,k-dk+tk),Real(0.0));  // Cm high-t
    const Real h = Real(0.5)*dt/dxt;
    for(int n=0;n<NVAR;++n){
        g[n] += -h*( cP0*Apm(i,      j,      k,      n,+1) + cP1*Apm(i+ti,   j+tj,   k+tk,   n,-1)
                   + cM0*Apm(i-di,   j-dj,   k-dk,   n,+1) + cM1*Apm(i-di+ti,j-dj+tj,k-dk+tk,n,-1) );
    }
}

// ---------------------------------------------------------------------
//  ps_viscous_face: deviatoric Newtonian stress added to the d-direction
//  flux at face (i,j,k): momentum flux += -tau_{d,c}, energy flux +=
//  -u_c tau_{d,c}, tau_{dd} = mu(2 du_d/dd - 2/3 div u), tau_{dt} =
//  mu(du_t/dd + du_d/dt).  Gives the shear layer a finite thickness
//  ~mu/(rho U) so Kelvin-Helmholtz roll-up is a resolved mode rather than
//  grid-scale odd-even.  Flux form (refluxes), symmetric; viscous heating
//  partitioned to UE1/UE2 by mass fraction.  Transverse gradients from the
//  two straddling cells; skipped within one cell of a transverse domain
//  edge.  mu = PSAMR.ps_mu [Pa s] (see PS_umeth).  docs §3.6.
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
ps_viscous_face(int d, int i, int j, int k,
                amrex::Array4<const amrex::Real> const& uin,
                amrex::Array4<amrex::Real> const& flx,
                const int* domlo, const int* domhi,
                amrex::Real mu, const amrex::Real* dx) noexcept
{
    using amrex::Real;
    if (mu <= Real(0.0)) return;
    const int off[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };
#if (AMREX_SPACEDIM == 3)
    const int UM[3] = { UMX, UMY, UMZ };
#elif (AMREX_SPACEDIM == 2)
    const int UM[3] = { UMX, UMY, UMX };
#else
    const int UM[3] = { UMX, UMX, UMX };
#endif
    const int di=off[d][0], dj=off[d][1], dk=off[d][2];
    auto vel=[&](int a,int b,int c,int comp) noexcept -> Real {
        Real r=uin(a,b,c,URHO); r=(std::isfinite(r)&&r>ps_const::RHO_MIX_MIN)?r:Real(ps_const::RHO_MIX_MIN);
        return ps_finite_or(uin(a,b,c,comp),Real(0.0))/r; };
    // normal-direction gradient of each velocity component at the face
    Real dvdn[3] = {Real(0.0),Real(0.0),Real(0.0)};
    for (int c=0;c<AMREX_SPACEDIM;++c)
        dvdn[c] = (vel(i,j,k,UM[c]) - vel(i-di,j-dj,k-dk,UM[c]))/dx[d];
    // transverse gradients (average of the two straddling cells)
    Real dvdt[3][3]; for(int a=0;a<3;++a)for(int b=0;b<3;++b) dvdt[a][b]=Real(0.0);
    Real div = dvdn[d];
    for (int t=0;t<AMREX_SPACEDIM;++t){
        if (t==d) continue;
        const int ti=off[t][0], tj=off[t][1], tk=off[t][2];
        const int It=(t==0)?i:(t==1)?j:k;
        if (It+1 > domhi[t] || It-1 < domlo[t]) return;   // 1st-order-safe at edge
        for (int c=0;c<AMREX_SPACEDIM;++c){
            const Real gR=(vel(i+ti,j+tj,k+tk,UM[c])-vel(i-ti,j-tj,k-tk,UM[c]))/(Real(2.0)*dx[t]);
            const Real gL=(vel(i-di+ti,j-dj+tj,k-dk+tk,UM[c])-vel(i-di-ti,j-dj-tj,k-dk-tk,UM[c]))/(Real(2.0)*dx[t]);
            dvdt[t][c]=Real(0.5)*(gR+gL);
        }
        div += dvdt[t][t];
    }
    Real tau[3]={Real(0.0),Real(0.0),Real(0.0)};
    tau[d] = mu*(Real(2.0)*dvdn[d] - (Real(2.0)/Real(3.0))*div);
    for (int t=0;t<AMREX_SPACEDIM;++t){ if(t==d) continue;
        tau[t] = mu*(dvdn[t] + dvdt[t][d]); }
    Real Ef=Real(0.0);
    for (int c=0;c<AMREX_SPACEDIM;++c){
        const Real uf=Real(0.5)*(vel(i,j,k,UM[c])+vel(i-di,j-dj,k-dk,UM[c]));
        flx(i,j,k,UM[c]) = ps_finite_or(flx(i,j,k,UM[c]) - tau[c], Real(0.0));
        Ef += -uf*tau[c];
    }
    Real m1=Real(0.5)*(amrex::max(uin(i,j,k,UM1RHO1),Real(0.0))+amrex::max(uin(i-di,j-dj,k-dk,UM1RHO1),Real(0.0)));
    Real m2=Real(0.5)*(amrex::max(uin(i,j,k,UM2RHO2),Real(0.0))+amrex::max(uin(i-di,j-dj,k-dk,UM2RHO2),Real(0.0)));
    Real ms=m1+m2; if(!(ms>Real(1e-30))) ms=Real(1e-30);
    const Real Y1=m1/ms, Y2=Real(1.0)-Y1;
    flx(i,j,k,UEDEN)=ps_finite_or(flx(i,j,k,UEDEN)+Ef,   Real(0.0));
    flx(i,j,k,UE1)  =ps_finite_or(flx(i,j,k,UE1)  +Y1*Ef,Real(0.0));
    flx(i,j,k,UE2)  =ps_finite_or(flx(i,j,k,UE2)  +Y2*Ef,Real(0.0));
}

// ---------------------------------------------------------------------
//  Solver-selection dial accessors (declared in PS_umeth.H).  One
//  ParmParse read per key, process-wide; the resolved value is force-added
//  so job_info records what actually ran.
// ---------------------------------------------------------------------
//  PSAMR.ps_flux (wp): the interior flux.  wp is the only interior flux
//  (docs/DESIGN_DECISIONS.md H-2); any other name aborts so a deck
//  that names a flux that does not exist fails loudly.  The key survives
//  so decks can state the flux explicitly and job_info records it.
int ps_flux_selector()
{
    static const int v = []() -> int {
        std::string s = "wp";
        amrex::ParmParse pp("PSAMR");
        pp.query("ps_flux", s);
        if (!(s == "wp" || s.empty())) {
            amrex::Abort("PSAMR.ps_flux must be 'wp'");
        }
        pp.add("ps_flux", std::string("wp"));
        return 2;   // wp's selector value
    }();
    return v;
}

const char* ps_flux_name()
{
    return (ps_flux_selector() == 2) ? "wp" : "invalid";
}


// ---------------------------------------------------------------------
//  PS_umeth: the hydro interior for one box (signature shared with
//  MOL_umeth for dispatch in Hydro_umdrv.cpp; q, qa, q1-3, a1-3, vol and
//  the small_* arguments are unused — wp works from uin_arr).  Reads the
//  dials once (host side, captured by value into the kernels), runs Pass 1
//  (fluctuations), Pass 2 (limited correction), the transverse and
//  viscous terms and Pass 3 (non-conserved deposit), and returns.
// ---------------------------------------------------------------------
void
PS_umeth(const Box& bx,
         const int* bclo, const int* bchi,              // physical boundary codes (IndexDefines.H)
         const int* domlo, const int* domhi,   // transverse-term domain guard
         Array4<const Real> const& uin_arr,
         Array4<const Real> const& /*q*/,
         Array4<const Real> const& /*qa*/,
         Array4<Real> const& dsdt_arr,
         AMREX_D_DECL(Array4<Real> const& flx1,
                      Array4<Real> const& flx2,
                      Array4<Real> const& flx3),
         AMREX_D_DECL(Array4<Real> const& /*q1*/,
                      Array4<Real> const& /*q2*/,
                      Array4<Real> const& /*q3*/),
         AMREX_D_DECL(Array4<const Real> const& /*a1*/,
                      Array4<const Real> const& /*a2*/,
                      Array4<const Real> const& /*a3*/),
         Array4<Real> const& pdivu,
         Array4<const Real> const& /*vol*/,
         const GpuArray<Real, AMREX_SPACEDIM> dx,
         const Real dt,
         const Real /*small*/,
         const Real /*small_dens*/,
         const Real /*small_pres*/,
         const Real /*smallu*/,
         const int /*slope_order*/,
         const PassMap* /*lpmap*/)
{
    BL_PROFILE("PS_umeth()");

    // Flux selector: always wp or an abort; read here so the value gate
    // runs and job_info records the flux.  (Retired keys abort at startup:
    // PS_retired_keys.H.)
    ps_flux_selector();

    // Presence parameters: one host-side read, captured by value into every
    // kernel (see ps_presence_params, PS_presence.H).
    const PsPres l_pres = ps_presence_params();

    // PSAMR.ps_wp_order (2): 2 = first-order fluctuations plus the limited
    // correction fluxes on all three waves (second order in smooth flow,
    // first order at discontinuities) — the acceptance order; 1 = the
    // first-order fluctuations alone, for comparisons; 4 = the face states
    // reconstructed per cell edge (PS_recon.H) with the in-cell term of
    // the non-conserved slots and no correction flux (docs/
    // PLAN_high_order.pdf section 4, Stage 3).  Any other value aborts.
    // 4 needs the method-of-lines path (do_mol=1): the reconstructed faces
    // have no characteristic-tracing half step.
    auto ps_wp_order_cached = []() -> int
    {
        static const int cached = []() -> int {
            const int v = ps_dial_int("ps_wp_order", 2);
            if (v != 1 && v != 2 && v != 4) {
                amrex::Abort("PSAMR.ps_wp_order must be 1, 2 or 4");
            }
            if (v == 4 && ps_dial_int("do_mol", 0) == 0) {
                amrex::Abort("PSAMR.ps_wp_order=4 needs PSAMR.do_mol=1 (a method-of-lines operator)");
            }
            return v; }();
        return cached;
    };
    const int wp_order = ps_wp_order_cached();

    // PSAMR.ps_wp_recon (weno5z): under ps_wp_order=4, the edge
    // reconstruction: weno5z (fifth order on smooth data, Borges et al.
    // 2008) or ppm (the Colella-Woodward limited interpolant, the plan's
    // fallback that tells a WENO-weight fault from a projection or in-cell
    // fault).  Any other string aborts.
    auto ps_wp_recon_cached = []() -> int
    {
        static const int cached = []() -> int {
            const std::string s = ps_dial_string("ps_wp_recon", "weno5z");
            if (s == "weno5z") return 0;
            if (s == "ppm")    return 1;
            amrex::Abort("PSAMR.ps_wp_recon must be weno5z or ppm");
            return 0; }();
        return cached;
    };
    const int wp_recon = ps_wp_recon_cached();
    // PSAMR.ps_wp_weno_power (1): under ps_wp_recon=weno5z, the power on
    // the ratio tau/beta_k in the Borges weights; 2 pushes a marginal
    // stencil harder toward ENO (docs/PLAN_high_order.pdf follow-up 2,
    // the first-named option against the weak-shock width).  1 or 2.
    auto ps_wp_weno_power_cached = []() -> int
    {
        static const int cached = []() -> int {
            const int v = ps_dial_int("ps_wp_weno_power", 1);
            if (v != 1 && v != 2) { amrex::Abort("PSAMR.ps_wp_weno_power must be 1 or 2"); }
            return v; }();
        return cached;
    };
    const int wp_wpow = ps_wp_weno_power_cached();

    // PSAMR.ps_wp_contact_taper (0): under ps_wp_order=4, 1 = the H-5
    // keep-weight of the face (MODEL section 3.4, on the raw alpha jump)
    // scales the whole high-order increment of the edge states on that
    // face, so a material interface sees the raw averages as it does on
    // the correction path; 0 = the reconstruction's own smoothness measure
    // is the only interface treatment.  Default 0 (Marc, 16 Nov 2026, on
    // the Stage 3 record): the weight caps every smooth alpha variation
    // at third order, 0 gives the design order and the cleanest jet edge;
    // its cost, vacuum folds on the AMR tube, is the thing to watch.
    auto ps_wp_contact_taper_cached = []() -> int
    {
        static const int cached = []() -> int {
            const int v = ps_dial_int("ps_wp_contact_taper", 0);
            if (v != 0 && v != 1) {
                amrex::Abort("PSAMR.ps_wp_contact_taper must be 0 or 1");
            }
            return v; }();
        return cached;
    };
    const int wp_contact_taper = ps_wp_contact_taper_cached();

    // PSAMR.ps_wp_sharpen (0): under ps_wp_order=4, 1 = interface
    // sharpening of alpha_1 at two-phase contacts, the THINC candidate
    // edges chosen over the reconstructed ones by the smaller boundary
    // variation (PS_recon.H; docs/PLAN_high_order.pdf follow-up 11); 0 =
    // the reconstruction alone.
    auto ps_wp_sharpen_cached = []() -> int
    {
        static const int cached = []() -> int {
            const int v = ps_dial_int("ps_wp_sharpen", 0);
            if (v != 0 && v != 1) { amrex::Abort("PSAMR.ps_wp_sharpen must be 0 or 1"); }
            return v; }();
        return cached;
    };
    const int wp_sharpen = ps_wp_sharpen_cached();

    // PSAMR.ps_wp_mc (0): under ps_wp_order=4, the McCorquodale-Colella
    // conversions (PS_recon.H; docs/PLAN_high_order.pdf follow-up 12): 1 =
    // the reconstruction's primitive set from the cell's point value,
    // converted back to an average (one more ghost cell, PSAMR::numGrow);
    // 2 = also the face states to face point values and the fluctuations
    // back to face averages (two more ghost cells); 0 = the set formed
    // from the averages.  The same key sets numGrow.
    auto ps_wp_mc_cached = []() -> int
    {
        static const int cached = []() -> int {
            const int v = ps_dial_int("ps_wp_mc", 0);
            if (v < 0 || v > 2) { amrex::Abort("PSAMR.ps_wp_mc must be 0, 1 or 2"); }
            return v; }();
        return cached;
    };
    const int wp_mc = ps_wp_mc_cached();

    // PSAMR.ps_wp_mc_bdry (1): under ps_wp_mc, 1 = the conversions refuse
    // a cell whose stencil leaves the domain across a non-periodic face
    // (the point conversion where a face neighbour is outside, the average
    // conversion where a face neighbour was so refused), counted as
    // recon_mc_bdry; the average's set stands there as at a presence front
    // (follow-up 16b; the default by Marc's word, follow-up 16c).  0 = the
    // conversions run into the boundary ghosts, which doubled the first
    // NSCBC ghost's liquid mass on the blowdown tube.
    auto ps_wp_mc_bdry_cached = []() -> int
    {
        static const int cached = []() -> int {
            const int v = ps_dial_int("ps_wp_mc_bdry", 1);
            if (v < 0 || v > 1) { amrex::Abort("PSAMR.ps_wp_mc_bdry must be 0 or 1"); }
            return v; }();
        return cached;
    };
    const int wp_mc_bdry = ps_wp_mc_bdry_cached();
    // PSAMR.ps_wp_mc_trace (-1): under ps_pres_diag on a host build, the
    // face (2*dir + hi) whose boundary cell and first ghost have their
    // average and converted sets printed per conversion pass (follow-up 16b).
    auto ps_wp_mc_trace_cached = []() -> int
    {
        static const int cached = ps_dial_int("ps_wp_mc_trace", -1);
        return cached;
    };
    const int wp_mc_trace = (ps_dial_int("ps_pres_diag", 0) != 0) ? ps_wp_mc_trace_cached() : -1;

    // PSAMR.ps_wp_limiter (vanleer): "none"/"unlimited" sets φ=1 (pure
    // Lax-Wendroff correction).  Unlimited is not monotone and is meant only
    // for smooth order-of-accuracy checks, where van Leer clips extrema.
    // Any other string aborts.
    auto ps_wp_unlimited_cached = []() -> int
    {
        static const int cached = []() -> int {
            const std::string s = ps_dial_string("ps_wp_limiter", "vanleer");
            if (s == "vanleer") return 0;
            if (s == "none" || s == "unlimited") return 1;
            amrex::Abort("PSAMR.ps_wp_limiter must be vanleer, none or unlimited");
            return 0; }();
        return cached;
    };
    const int wp_unlimited = ps_wp_unlimited_cached();

    // Weight of the correction on the contact wave, which carries the
    // non-conservative alpha jump: correcting it at a material interface
    // smears alpha into the phase densities/energies, while skipping it
    // everywhere costs single-phase contact sharpening.  The weight is a
    // smoothstep taper in |Δα|/α_cond (1 in uniform regions, 0 across a
    // jump ≥ α_cond, no new threshold), applied to the whole wave only
    // (docs/DESIGN_DECISIONS.md H-5).

    // PSAMR.ps_wp_transverse (0): 0 = directionally split; 1 = contact-only
    // transverse correction (ps_wp_tvterm; the validated 2-D setting).
    // Any other value aborts.  2-D/3-D only.
    auto ps_wp_transverse_cached = []() -> int
    {
        static const int cached = []() -> int {
            const int v = ps_dial_int("ps_wp_transverse", 0);
            if (v == 2) {
                amrex::Abort("PSAMR.ps_wp_transverse=2 is retired: the acoustic "
                    "transverse coupling is deleted; use 1 (contact-only)");
            }
            if (v != 0 && v != 1) {
                amrex::Abort("PSAMR.ps_wp_transverse must be 0 or 1");
            }
            return v; }();
        return cached;
    };
#if (AMREX_SPACEDIM >= 2)
    const int wp_transverse = ps_wp_transverse_cached();
#else
    const int wp_transverse = 0;
    amrex::ignore_unused(ps_wp_transverse_cached);
#endif

    // PSAMR.rk_order (2): under 3 (SSPRK(3,3), PSAMR_advance.cpp) the
    // correction takes its method-of-lines form: no (1 - |s| dt/dx)
    // factor, because a dt-dependent spatial operator is first order in dt
    // inside any Runge-Kutta stage whatever the integrator's order
    // (measured: docs/PLAN_high_order.pdf section 11, Stage 0 item 3), and
    // no transverse term, because the stages supply the cross term.  Under
    // rk_order=2 (Heun) and the single-step scheme the Lax-Wendroff form is
    // unchanged.
    auto ps_wp_mol_form_cached = []() -> int
    {
        static const int cached = []() -> int {
            const int v = ps_dial_int("rk_order", 2);
            return (v == 3) ? 1 : 0; }();
        return cached;
    };
    const int wp_mol_form = ps_wp_mol_form_cached();

    // PSAMR.ps_lim_ref_eps (1e-8): the reference-scale floor on the slot
    // weight of the limiter's nondimensional inner product (O-32, docs/
    // PLAN_high_order.pdf Stage 2d).  A slot's local scale |U_L|+|U_R| is
    // floored at eps times a physical reference for the slot (mass slots:
    // rho_bar; momentum: rho_bar (|s_1| + c_bar); energy: rho_bar
    // (|s_1| + c_bar)^2; alpha: 1), the acoustic eigenvector normalisation
    // written as a floor, so that a slot at noise level cannot decide phi
    // (measured on the T-Blowdown tube: the y-momentum at round-off drove
    // a transverse instability; Stage 2c).  eps is 10x the EOS inversion
    // tolerance of hem_pr_state.H (1e-9): a slot below that fraction of its
    // reference is inside the noise of the state it is computed from.
    // 0 = the old rule (the A/B control).
    auto ps_lim_ref_eps_cached = []() -> amrex::Real
    {
        static const amrex::Real cached = []() -> amrex::Real {
            const amrex::Real v = ps_dial_real("ps_lim_ref_eps", amrex::Real(1.0e-8));
            if (v < amrex::Real(0.0)) {
                amrex::Abort("PSAMR.ps_lim_ref_eps must be >= 0");
            }
            return v; }();
        return cached;
    };
    const amrex::Real lim_ref_eps = ps_lim_ref_eps_cached();
    if (wp_mol_form != 0 && wp_transverse != 0) {
        amrex::Abort("PSAMR.ps_wp_transverse must be 0 under rk_order=3 "
                     "(the Runge-Kutta stages supply the cross term)");
    }
    if (wp_order == 4 && wp_transverse != 0) {
        amrex::Abort("PSAMR.ps_wp_transverse must be 0 under ps_wp_order=4 "
                     "(the reconstructed faces carry no transverse term)");
    }

    // PSAMR.ps_mu (0 = inviscid): dynamic viscosity [Pa s] for
    // ps_viscous_face; the pipe-break decks run an effective 2 Pa s.
    auto ps_mu_cached = []() -> amrex::Real {
        static const amrex::Real cached = []() -> amrex::Real {
            const amrex::Real v = ps_dial_real("ps_mu", 0.0);
            return (v > amrex::Real(0.0)) ? v : amrex::Real(0.0); }();
        return cached;
    };
#if (AMREX_SPACEDIM >= 2)
    const amrex::Real ps_mu = ps_mu_cached();
#else
    const amrex::Real ps_mu = amrex::Real(0.0);
    amrex::ignore_unused(ps_mu_cached);
#endif

    // Banner (once per rank per run).
    {
        static bool banner_shown = false;
        if (!banner_shown) {
            amrex::Print()
                << "  PS_umeth: Berger-LeVeque WP (fluctuation) flux — "
                << (wp_order == 2
                        ? "BL-2 limited correction fluxes (2nd order, the acceptance order)"
                   : wp_order == 4
                        ? "reconstructed faces with the in-cell term (PSAMR.ps_wp_order=4)"
                        : "1st-order fluctuations (BL-1, PSAMR.ps_wp_order=1)");
            if (wp_order == 4) {
                amrex::Print() << ", " << (wp_recon == 1 ? "ppm" : "weno5z")
                               << " on the primitive set, contact taper "
                               << wp_contact_taper << ", sharpen " << wp_sharpen
                               << ", conversions " << wp_mc;
            }
            amrex::Print() << "\n";
            banner_shown = true;
        }
    }

    // hydro_umdrv hands over pdivu uninitialised and hydro_consup adds it
    // to dsdt; wp has no separate P∇·u term (the pressure work sits in
    // F[UEDEN]), so pdivu must be zeroed here.
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        pdivu(i,j,k) = Real(0.0);
    });

    // ------ face-centered boxes ------------------------------------
    const Box xfbx = amrex::surroundingNodes(bx, 0);
#if (AMREX_SPACEDIM >= 2)
    const Box yfbx = amrex::surroundingNodes(bx, 1);
#endif
#if (AMREX_SPACEDIM == 3)
    const Box zfbx = amrex::surroundingNodes(bx, 2);
#endif

    // ========== wave-propagation interior ===============================
    //  wp is the only interior; ps_flux_selector() admits nothing else.
    {
        const bool o2 = (wp_order == 2);            // correction fluxes
        const bool o4 = (wp_order == 4);            // reconstructed faces
        const bool tv = (wp_transverse != 0);       // transverse term
        const bool unlim = (wp_unlimited != 0);     // bypass van Leer
        const bool store_w = o2 || tv;              // need the raw waves?
        const int  wc = store_w ? (3*NVAR + 3) : 1; // wave/speed store width
        const int  fc = o2 ? 3 : 1;                 // Ftilde store {α,UE1,UE2}
        const int  pc = o4 ? 16 : 6;                // per-face non-conserved store width
        const Real dt_l = dt;
        const int  recon_l = wp_recon;
        const int  taper_l = wp_contact_taper;
        const int  wpow_l  = wp_wpow;
        const int  sharp_l = wp_sharpen;
        const int  mc_l    = wp_mc;
        const int  mcb_l   = wp_mc_bdry;
        // The cells whose conversion stencil stays inside the domain: one
        // (point) and two (average) cells in from every non-periodic face;
        // a periodic face's ghosts are interior cells and no bound applies.
        amrex::GpuArray<int, 3> mc_lo1{{-2000000000, -2000000000, -2000000000}}, mc_hi1{{2000000000, 2000000000, 2000000000}};
        amrex::GpuArray<int, 3> mc_lo2 = mc_lo1, mc_hi2 = mc_hi1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (bclo[d] != Interior) { mc_lo1[d] = domlo[d] + 1; mc_lo2[d] = domlo[d] + 2; }
            if (bchi[d] != Interior) { mc_hi1[d] = domhi[d] - 1; mc_hi2[d] = domhi[d] - 2; }
        }
        auto mc_inside = [] (int i, int j, int k, const amrex::GpuArray<int, 3>& lo, const amrex::GpuArray<int, 3>& hi) noexcept -> bool {
            const int idx[3] = {i, j, k};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { if (idx[d] < lo[d] || idx[d] > hi[d]) return false; }
            return true;
        };

        // Per-face store of the non-conserved fluctuations A⁻/A⁺ (comps
        // 0/1 UALPHA1, 2/3 UE1, 4/5 UE2).  The wave store is grown by one
        // in the normal direction for the correction's upwind stencil and
        // isotropically when the transverse gather reads the perpendicular
        // neighbour faces; ghost faces come from uin_arr's ghost cells.
        auto wbox = [&](const amrex::Box& fb, int nrm) {
            if (tv)  return amrex::grow(fb, 1);
            if (o2)  return amrex::grow(fb, nrm, 1);
            return fb;
        };
        const amrex::Box wxbx = wbox(xfbx, 0);
        amrex::FArrayBox wp_fluct_x_fab(xfbx, pc, amrex::The_Async_Arena());
        amrex::FArrayBox wp_wave_x_fab (wxbx, wc, amrex::The_Async_Arena());
        amrex::FArrayBox wp_ft_x_fab   (xfbx, fc, amrex::The_Async_Arena());
        auto const& wp_fluct_x = wp_fluct_x_fab.array();
        auto const& wp_wave_x  = wp_wave_x_fab.array();
        auto const& wp_ft_x    = wp_ft_x_fab.array();
#if (AMREX_SPACEDIM >= 2)
        const amrex::Box wybx = wbox(yfbx, 1);
        amrex::FArrayBox wp_fluct_y_fab(yfbx, pc, amrex::The_Async_Arena());
        amrex::FArrayBox wp_wave_y_fab (wybx, wc, amrex::The_Async_Arena());
        amrex::FArrayBox wp_ft_y_fab   (yfbx, fc, amrex::The_Async_Arena());
        auto const& wp_fluct_y = wp_fluct_y_fab.array();
        auto const& wp_wave_y  = wp_wave_y_fab.array();
        auto const& wp_ft_y    = wp_ft_y_fab.array();
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Box wzbx = wbox(zfbx, 2);
        amrex::FArrayBox wp_fluct_z_fab(zfbx, pc, amrex::The_Async_Arena());
        amrex::FArrayBox wp_wave_z_fab (wzbx, wc, amrex::The_Async_Arena());
        amrex::FArrayBox wp_ft_z_fab   (zfbx, fc, amrex::The_Async_Arena());
        auto const& wp_fluct_z = wp_fluct_z_fab.array();
        auto const& wp_wave_z  = wp_wave_z_fab.array();
        auto const& wp_ft_z    = wp_ft_z_fab.array();
#endif

        // ---- Pass 0 (ps_wp_order=4): the two edge states of every cell
        //      of the box grown by one, per direction, from the five cell
        //      averages along it (PS_recon.H); the faces of the valid box
        //      read the high edge of the left cell and the low edge of the
        //      right cell.  Three ghost cells are read (five under the
        //      conversions); numGrow() is 4 (5).
        //  At ps_wp_mc=2 the edges are needed one cell further out (the
        //  faces grown transversally by one read their transverse
        //  neighbours' edges), and so are the converted averages.
        const int eg = (mc_l == 2) ? 1 : 0;
        const amrex::Box rbx = amrex::grow(bx, 1 + eg);
        amrex::FArrayBox rq_x_fab;
        if (o4) { rq_x_fab.resize(rbx, PS_RECON::RQ_NCOMP, amrex::The_Async_Arena()); }
        auto const& rq_x = rq_x_fab.array();
#if (AMREX_SPACEDIM >= 2)
        amrex::FArrayBox rq_y_fab;
        if (o4) { rq_y_fab.resize(rbx, PS_RECON::RQ_NCOMP, amrex::The_Async_Arena()); }
        auto const& rq_y = rq_y_fab.array();
#endif
#if (AMREX_SPACEDIM == 3)
        amrex::FArrayBox rq_z_fab;
        if (o4) { rq_z_fab.resize(rbx, PS_RECON::RQ_NCOMP, amrex::The_Async_Arena()); }
        auto const& rq_z = rq_z_fab.array();
#endif
        // The conversion passes (ps_wp_mc): the point-value primitive set on
        // the box grown by four (reading five cells out), the converted
        // averages on the box grown by three, which the edge pass reads.
        amrex::FArrayBox wc_fab, wq_fab;
        if (o4 && mc_l != 0) {
            wc_fab.resize(amrex::grow(bx, 4 + eg), PS_RECON::NPRIM + 3, amrex::The_Async_Arena());
            wq_fab.resize(amrex::grow(bx, 3 + eg), PS_RECON::NPRIM + 3, amrex::The_Async_Arena());
        }
        auto const& wc_arr = wc_fab.array();
        auto const& wq_arr = wq_fab.array();
        amrex::Array4<const Real> const wq_c = wq_fab.const_array();
        //  A phase density outside the EOS's own domain is not a state,
        //  whatever the branch solve says of it (follow-up 16c: a point
        //  value of 1901 kg/m3, past the covolume pole, had passed the
        //  solve); the bound is the fill's clamp's, no new constant.
        const Real mc_rho_lo = EOS::rho_min(), mc_rho_hi = EOS::rho_max();
        auto ps_mc_rho_in_domain = [=] AMREX_GPU_DEVICE (Real r) noexcept -> bool { return r >= mc_rho_lo && r <= mc_rho_hi; };
        if (o4 && mc_l != 0) {
            BL_PROFILE_VAR("PS::wp_mc_prims()", ps_wp_mc_prof);
            //  The point value is used only where it is a state: each
            //  present phase's (rho, e) must have a root on its branch
            //  (the EOS's own verdict, PS_corridor.H's pattern); elsewhere
            //  the average's primitive set stands.  Rule 2: a point value
            //  that is not a state is refused at creation, not repaired.
            amrex::ParallelFor(amrex::grow(bx, 4 + eg),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                bool raw = PS_RECON::point_prims(i, j, k, uin_arr, wc_arr, l_pres);
                if (wc_arr(i, j, k, PS_RECON::WC_PT) != Real(0.0)) {
                    Real Yp[NUM_SPECIES]; ps_pure_species(Yp);
                    Real Pq, Tq, Sq;
                    bool ok = true;
                    if (wc_arr(i, j, k, PS_RECON::WC_AB1) == Real(0.0))
                        ok = ps_mc_rho_in_domain(wc_arr(i, j, k, PS_RECON::IR1))
                          && EOS::REY2PTS_phase_try(wc_arr(i, j, k, PS_RECON::IR1), wc_arr(i, j, k, PS_RECON::IE1), Yp, hem::Phase3::Liquid, Pq, Tq, Sq);
                    if (ok && wc_arr(i, j, k, PS_RECON::WC_AB2) == Real(0.0))
                        ok = ps_mc_rho_in_domain(wc_arr(i, j, k, PS_RECON::IR2))
                          && EOS::REY2PTS_phase_try(wc_arr(i, j, k, PS_RECON::IR2), wc_arr(i, j, k, PS_RECON::IE2), Yp, hem::Phase3::Vapor, Pq, Tq, Sq);
                    if (!ok) {
                        Real U[NVAR];
                        for (int n = 0; n < NVAR; ++n) U[n] = uin_arr(i, j, k, n);
                        const PS_RECON::PsPrim Pa = PS_RECON::prim_from_cons(U, l_pres);
                        for (int m = 0; m < PS_RECON::NPRIM; ++m) wc_arr(i, j, k, m) = Pa.v[m];
                        wc_arr(i, j, k, PS_RECON::WC_PT) = Real(0.0);
                        raw = true;
                    }
                }
                if (raw) ps_guard::count_recon_mc_raw();
                if (mcb_l != 0 && wc_arr(i, j, k, PS_RECON::WC_PT) != Real(0.0) && !mc_inside(i, j, k, mc_lo1, mc_hi1)) {
                    Real U[NVAR];
                    for (int n = 0; n < NVAR; ++n) U[n] = uin_arr(i, j, k, n);
                    const PS_RECON::PsPrim Pa = PS_RECON::prim_from_cons(U, l_pres);
                    for (int m = 0; m < PS_RECON::NPRIM; ++m) wc_arr(i, j, k, m) = Pa.v[m];
                    wc_arr(i, j, k, PS_RECON::WC_PT) = Real(0.0);
                    ps_guard::count_recon_mc_bdry();
                }
            });
            //  The converted average must be a state as well (its phase
            //  energies are the point values' plus a 24th of their second
            //  difference); where it is not, the average's own set stands.
            amrex::Array4<const Real> const wc_c = wc_fab.const_array();
            amrex::ParallelFor(amrex::grow(bx, 3 + eg),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                PS_RECON::avg_prims(i, j, k, wc_c, wq_arr);
                if (wq_arr(i, j, k, PS_RECON::WC_PT) != Real(0.0)) {
                    Real Yp[NUM_SPECIES]; ps_pure_species(Yp);
                    Real Pq, Tq, Sq;
                    bool ok = true;
                    if (wq_arr(i, j, k, PS_RECON::WC_AB1) == Real(0.0))
                        ok = ps_mc_rho_in_domain(wq_arr(i, j, k, PS_RECON::IR1))
                          && EOS::REY2PTS_phase_try(wq_arr(i, j, k, PS_RECON::IR1), wq_arr(i, j, k, PS_RECON::IE1), Yp, hem::Phase3::Liquid, Pq, Tq, Sq);
                    if (ok && wq_arr(i, j, k, PS_RECON::WC_AB2) == Real(0.0))
                        ok = ps_mc_rho_in_domain(wq_arr(i, j, k, PS_RECON::IR2))
                          && EOS::REY2PTS_phase_try(wq_arr(i, j, k, PS_RECON::IR2), wq_arr(i, j, k, PS_RECON::IE2), Yp, hem::Phase3::Vapor, Pq, Tq, Sq);
                    if (!ok) {
                        Real U[NVAR];
                        for (int n = 0; n < NVAR; ++n) U[n] = uin_arr(i, j, k, n);
                        const PS_RECON::PsPrim Pa = PS_RECON::prim_from_cons(U, l_pres);
                        for (int m = 0; m < PS_RECON::NPRIM; ++m) wq_arr(i, j, k, m) = Pa.v[m];
                        wq_arr(i, j, k, PS_RECON::WC_PT) = Real(0.0);
                        ps_guard::count_recon_mc_raw();
                    }
                }
                if (mcb_l != 0 && wq_arr(i, j, k, PS_RECON::WC_PT) != Real(0.0) && !mc_inside(i, j, k, mc_lo2, mc_hi2)) {
                    Real U[NVAR];
                    for (int n = 0; n < NVAR; ++n) U[n] = uin_arr(i, j, k, n);
                    const PS_RECON::PsPrim Pa = PS_RECON::prim_from_cons(U, l_pres);
                    for (int m = 0; m < PS_RECON::NPRIM; ++m) wq_arr(i, j, k, m) = Pa.v[m];
                    wq_arr(i, j, k, PS_RECON::WC_PT) = Real(0.0);
                    ps_guard::count_recon_mc_bdry();
                }
            });
#if !defined(AMREX_USE_GPU)
            //  The trace (ps_wp_mc_trace under ps_pres_diag): the boundary
            //  cell and the first ghost of the named face, their average set
            //  against the point and converted sets (follow-up 16b).
            if (wp_mc_trace >= 0 && wp_mc_trace < 2 * AMREX_SPACEDIM) {
                const int td = wp_mc_trace / 2, thi = wp_mc_trace % 2;
                const int ib = thi ? domhi[td] : domlo[td];
                const int ig = thi ? domhi[td] + 1 : domlo[td] - 1;
                amrex::Array4<const Real> const wc_t = wc_fab.const_array();
                amrex::Array4<const Real> const wq_t = wq_fab.const_array();
                const amrex::Box tb = amrex::grow(bx, 3 + eg);
                amrex::LoopOnCpu(tb, [&] (int i, int j, int k) {
                    const int idx = (td == 0) ? i : (td == 1) ? j : k;
                    if (idx != ib && idx != ig) return;
                    Real U[NVAR];
                    for (int n = 0; n < NVAR; ++n) U[n] = uin_arr(i, j, k, n);
                    const PS_RECON::PsPrim Pa = PS_RECON::prim_from_cons(U, l_pres);
                    const Real Pm = ps_cell_state(U, l_pres).P_mix;
                    amrex::AllPrint() << std::setprecision(10)
                        << "[MC-TRACE] face=" << wp_mc_trace << (idx == ib ? " cell " : " ghost ")
                        << i << " " << j << " " << k
                        << " avg: u=" << Pa.v[PS_RECON::IU + td] << " a1=" << Pa.v[PS_RECON::IA1]
                        << " r1=" << Pa.v[PS_RECON::IR1] << " r2=" << Pa.v[PS_RECON::IR2] << " P=" << Pm
                        << " | pt(" << wc_t(i, j, k, PS_RECON::WC_PT) << "): u=" << wc_t(i, j, k, PS_RECON::IU + td)
                        << " a1=" << wc_t(i, j, k, PS_RECON::IA1) << " r1=" << wc_t(i, j, k, PS_RECON::IR1)
                        << " | cavg(" << wq_t(i, j, k, PS_RECON::WC_PT) << "): u=" << wq_t(i, j, k, PS_RECON::IU + td)
                        << " a1=" << wq_t(i, j, k, PS_RECON::IA1) << " r1=" << wq_t(i, j, k, PS_RECON::IR1)
                        << "\n";
                });
            }
#endif
            BL_PROFILE_VAR_STOP(ps_wp_mc_prof);
        }
        if (o4) {
            BL_PROFILE_VAR("PS::wp_recon_edges()", ps_wp_recon_prof);
            amrex::ParallelFor(rbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int rc = PS_RECON::cell_edges(0, i, j, k, uin_arr, rq_x, recon_l, taper_l, wpow_l, sharp_l, mc_l, wq_c, l_pres);
                if (rc == 1) { ps_guard::count_recon_fallback(); } else if (rc == 2) { ps_guard::count_recon_thinc(); }
            });
#if (AMREX_SPACEDIM >= 2)
            amrex::ParallelFor(rbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int rc = PS_RECON::cell_edges(1, i, j, k, uin_arr, rq_y, recon_l, taper_l, wpow_l, sharp_l, mc_l, wq_c, l_pres);
                if (rc == 1) { ps_guard::count_recon_fallback(); } else if (rc == 2) { ps_guard::count_recon_thinc(); }
            });
#endif
#if (AMREX_SPACEDIM == 3)
            amrex::ParallelFor(rbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int rc = PS_RECON::cell_edges(2, i, j, k, uin_arr, rq_z, recon_l, taper_l, wpow_l, sharp_l, mc_l, wq_c, l_pres);
                if (rc == 1) { ps_guard::count_recon_fallback(); } else if (rc == 2) { ps_guard::count_recon_thinc(); }
            });
#endif
            BL_PROFILE_VAR_STOP(ps_wp_recon_prof);
        }

        // ---- Pass 1: fluctuations → F* (conserved flx), A⁻/A⁺ (non-
        //      conserved store) and raw waves/speeds.  Loops over the grown
        //      wave box; flx/wpf are written only on the valid face box.
        //  EOS-heavy (per-face branch-locked solves).  On a GPU build the
        //  ParallelFors are async, so the timer is meaningful on CPU only.
        BL_PROFILE_VAR("PS::wp_face_riemann()", ps_wp_face_prof);
        //  The side states: the reconstructed edges under order 4 (the
        //  wave box equals the face box there), the raw averages otherwise.
        const int cL0 = o4 ? NVAR : 0;
        if (o4 && mc_l == 2) {
            //  The face point values on the face boxes grown transversally
            //  by one, into point stores; then the fluctuations and the
            //  flux converted back to face averages on the valid faces,
            //  F = F_pt + (1/24) sum over the transverse directions of the
            //  second difference of F_pt.
            auto tgrow = [](const amrex::Box& fb, int nrm) { amrex::Box b = amrex::grow(fb, 1); b.grow(nrm, -1); return b; };
            auto back = [=] AMREX_GPU_DEVICE (int idir, int i, int j, int k,
                                              amrex::Array4<const Real> const& fp, amrex::Array4<Real> const& f, int nc) noexcept {
                for (int n = 0; n < nc; ++n) {
                    Real v = fp(i, j, k, n);
                    for (int tdir = 0; tdir < AMREX_SPACEDIM; ++tdir) {
                        if (tdir == idir) continue;
                        const int oi = (tdir == 0), oj = (tdir == 1), ok = (tdir == 2);
                        v += (fp(i + oi, j + oj, k + ok, n) - Real(2.0) * fp(i, j, k, n) + fp(i - oi, j - oj, k - ok, n)) / Real(24.0);
                    }
                    f(i, j, k, n) = v;
                }
            };
            const amrex::Array4<const Real> rqc_x(rq_x);
            const amrex::Box xtb = tgrow(xfbx, 0);
            amrex::FArrayBox fpx_fab(xtb, NVAR, amrex::The_Async_Arena()), wpx_fab(xtb, pc, amrex::The_Async_Arena());
            auto const& fpx = fpx_fab.array(); auto const& wpx = wpx_fab.array();
            amrex::ParallelFor(xtb, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                ps_wp_face_pt(0, i, j, k, i-1, j, k, uin_arr, rqc_x, fpx, wpx, wp_wave_x, store_w, xtb, l_pres); });
            const amrex::Array4<const Real> fpxc(fpx), wpxc(wpx);
            amrex::ParallelFor(xfbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                back(0, i, j, k, fpxc, flx1, NVAR); back(0, i, j, k, wpxc, wp_fluct_x, pc); });
#if (AMREX_SPACEDIM >= 2)
            const amrex::Array4<const Real> rqc_y(rq_y);
            const amrex::Box ytb = tgrow(yfbx, 1);
            amrex::FArrayBox fpy_fab(ytb, NVAR, amrex::The_Async_Arena()), wpy_fab(ytb, pc, amrex::The_Async_Arena());
            auto const& fpy = fpy_fab.array(); auto const& wpy = wpy_fab.array();
            amrex::ParallelFor(ytb, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                ps_wp_face_pt(1, i, j, k, i, j-1, k, uin_arr, rqc_y, fpy, wpy, wp_wave_y, store_w, ytb, l_pres); });
            const amrex::Array4<const Real> fpyc(fpy), wpyc(wpy);
            amrex::ParallelFor(yfbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                back(1, i, j, k, fpyc, flx2, NVAR); back(1, i, j, k, wpyc, wp_fluct_y, pc); });
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Array4<const Real> rqc_z(rq_z);
            const amrex::Box ztb = tgrow(zfbx, 2);
            amrex::FArrayBox fpz_fab(ztb, NVAR, amrex::The_Async_Arena()), wpz_fab(ztb, pc, amrex::The_Async_Arena());
            auto const& fpz = fpz_fab.array(); auto const& wpz = wpz_fab.array();
            amrex::ParallelFor(ztb, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                ps_wp_face_pt(2, i, j, k, i, j, k-1, uin_arr, rqc_z, fpz, wpz, wp_wave_z, store_w, ztb, l_pres); });
            const amrex::Array4<const Real> fpzc(fpz), wpzc(wpz);
            amrex::ParallelFor(zfbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                back(2, i, j, k, fpzc, flx3, NVAR); back(2, i, j, k, wpzc, wp_fluct_z, pc); });
#endif
        } else {
        const amrex::Array4<const Real> src_x = o4 ? amrex::Array4<const Real>(rq_x) : uin_arr;
        amrex::ParallelFor(wxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            ps_wp_face(0, i, j, k, i-1, j, k, src_x, cL0, 0, flx1, wp_fluct_x, wp_wave_x, store_w, o4, xfbx, l_pres);
        });
#if (AMREX_SPACEDIM >= 2)
        const amrex::Array4<const Real> src_y = o4 ? amrex::Array4<const Real>(rq_y) : uin_arr;
        amrex::ParallelFor(wybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            ps_wp_face(1, i, j, k, i, j-1, k, src_y, cL0, 0, flx2, wp_fluct_y, wp_wave_y, store_w, o4, yfbx, l_pres);
        });
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Array4<const Real> src_z = o4 ? amrex::Array4<const Real>(rq_z) : uin_arr;
        amrex::ParallelFor(wzbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            ps_wp_face(2, i, j, k, i, j, k-1, src_z, cL0, 0, flx3, wp_fluct_z, wp_wave_z, store_w, o4, zfbx, l_pres);
        });
#endif
        }
        BL_PROFILE_VAR_STOP(ps_wp_face_prof);

        // ---- Pass 2: LeVeque limited correction fluxes (docs §3). ----
        //  F̃_f = Σ_l w_l ½|s_l|(1−|s_l|Δt/Δx) φ(θ_l) W_l, one scalar φ per
        //  wave, w_l the contact keep-weight.  Added to the conserved-slot
        //  flux (telescopes through consup) and stored for the {α,UE1,UE2}
        //  deposit.  Upwind neighbour per wave sign; a face with no in-box
        //  upwind neighbour drops the term (first order there).
        if (o2) {
            auto correct = [=] AMREX_GPU_DEVICE
                (int idir, int i, int j, int k, const amrex::Box fbox,
                 amrex::Array4<amrex::Real> const& flx,
                 amrex::Array4<amrex::Real> const& wv,
                 amrex::Array4<amrex::Real> const& ft, Real dxd) noexcept
            {
                const Real dtdx = dt_l / dxd;
                //  Contact-wave keep-weight wc_keep, one scalar on the whole
                //  wave l=1 (a scalar multiple keeps the linear identities).
                //  wc_keep = 1 − smoothstep(|Δα|/α_cond): a uniform region at
                //  any level (including a uniform trace) keeps the full
                //  correction, a jump ≥ α_cond drops it, and there is no
                //  on/off contour to imprint a kink.  docs §3.4.
                const int oi = (idir == 0) ? 1 : 0;
                const int oj = (idir == 1) ? 1 : 0;
                const int ok2 = (idir == 2) ? 1 : 0;
                const Real wc_keep = PS_RECON::contact_keep(
                    uin_arr(i-oi, j-oj, k-ok2, UALPHA1) - uin_arr(i, j, k, UALPHA1), l_pres.alpha_cond);
                //  Reference scales of the limiter's floor (O-32): the face
                //  mean density, the contact speed and the half spread of
                //  the acoustic speeds, from the stored waves of this face.
                Real rho_bar = Real(0.0), vref = Real(0.0);
                if (!unlim && lim_ref_eps > Real(0.0)) {
                    rho_bar = Real(0.5) * (uin_arr(i-oi, j-oj, k-ok2, URHO) + uin_arr(i, j, k, URHO));
                    const Real s_c  = std::abs(wv(i,j,k, 3*NVAR + 1));
                    const Real c_bar = amrex::max(Real(0.5) * (wv(i,j,k, 3*NVAR + 2)
                                                              - wv(i,j,k, 3*NVAR + 0)), Real(0.0));
                    vref = s_c + c_bar;
                }
                Real Ft[NVAR];
                for (int n = 0; n < NVAR; ++n) Ft[n] = Real(0.0);
                for (int l = 0; l < 3; ++l) {
                    const Real wl = (l == 1) ? wc_keep : Real(1.0);
                    if (l == 1 && wl <= Real(0.0)) continue;   // full drop
                    const Real sl = wv(i,j,k, 3*NVAR + l);
                    if (std::abs(sl) < Real(1.0e-30)) continue;
                    const int ni = i - ((idir==0) ? ((sl>Real(0.0))?1:-1) : 0);
                    const int nj = j - ((idir==1) ? ((sl>Real(0.0))?1:-1) : 0);
                    const int nk = k - ((idir==2) ? ((sl>Real(0.0))?1:-1) : 0);
                    if (!fbox.contains(amrex::IntVect(AMREX_D_DECL(ni,nj,nk)))) continue;
                    const Real asl   = std::abs(sl);
                    //  Lax-Wendroff factor (1 - |s| dt/dx) in the single-step
                    //  and Heun paths; none under rk_order=3 (method of lines).
                    const Real coef0 = Real(0.5) * asl *
                        ((wp_mol_form != 0) ? Real(1.0) : (Real(1.0) - asl * dtdx));
                    //  One scalar limiter per wave (LeVeque):
                    //      θ = <W_up, W_f>_D / <W_f, W_f>_D,   W̃ = φ(θ) W_f,
                    //  with <a,b>_D = Σ_n a_n b_n / (|U_L,n| + |U_R,n|)².
                    //  Scaling a whole wave
                    //  by one number preserves its direction in state space and
                    //  therefore the linear identities every raw wave satisfies;
                    //  per-component factors bend the wave and drift the
                    //  phase-energy split.  The nondimensional inner product
                    //  stops the energy components (orders of magnitude larger
                    //  than mass) from setting φ alone; it changes only which
                    //  scalar comes out, φ is still applied to the raw wave.
                    //  See docs/MODEL_AND_ALGORITHM.md §3.3.
                    Real phi = Real(1.0);
                    if (!unlim) {
                        Real wdotw = Real(0.0), wdotu = Real(0.0);
                        int ndrop = 0;
                        for (int n = 0; n < NVAR; ++n) {
                            if (n == UTEMP) continue;   // not a wave component
                            const Real Wf  = wv(i, j, k,  l*NVAR + n);
                            const Real Wup = wv(ni,nj,nk, l*NVAR + n);
                            //  A component identically zero in both cells
                            //  (an absent phase's slots) carries no
                            //  information and is skipped; if the wave is
                            //  non-zero there (phase birth) it is scaled by
                            //  its own magnitude.
                            //  A scale below the smallest normal number is
                            //  not inverted (its reciprocal is infinite and
                            //  phi would be NaN): it is treated as the zero
                            //  case and counted.  docs/DESIGN_DECISIONS.md O-31.
                            Real inv = Real(1.0);
                            {
                                constexpr Real tiny = std::numeric_limits<Real>::min();
                                const Real sc =
                                    std::abs(uin_arr(i-oi, j-oj, k-ok2, n))
                                  + std::abs(uin_arr(i,    j,    k,    n));
                                if (sc > Real(0.0) && sc < tiny) ps_guard::count_lim_slot_denorm();
                                Real scl = Real(0.0);
                                if (sc >= tiny)                scl = sc;
                                else if (std::abs(Wf) >= tiny) scl = std::abs(Wf);
                                else                           continue;
                                //  The reference-scale rule (O-32): mass
                                //  slots by rho_bar, momentum by rho_bar
                                //  (|s_1| + c_bar), energy by rho_bar
                                //  (|s_1| + c_bar)^2, alpha by 1; a slot
                                //  below eps times its reference is inside
                                //  the noise of the state and drops out of
                                //  the inner product, counted.  (Flooring
                                //  its weight instead was measured not to
                                //  help: in a sweep through a flow uniform
                                //  along that direction every other slot's
                                //  jump is round-off, so a floored slot
                                //  still decides phi; Stage 2d record.)
                                if (lim_ref_eps > Real(0.0)) {
                                    Real ref = rho_bar;
                                    if (n == UMX
#if (AMREX_SPACEDIM >= 2)
                                        || n == UMY
#endif
#if (AMREX_SPACEDIM == 3)
                                        || n == UMZ
#endif
                                        ) {
                                        ref = rho_bar * vref;
                                    } else if (n == UEDEN || n == UEINT || n == UE1 || n == UE2) {
                                        ref = rho_bar * vref * vref;
                                    } else if (n == UALPHA1) {
                                        ref = Real(1.0);
                                    }
                                    const Real fl = lim_ref_eps * ref;
                                    if (scl < fl) { ps_guard::count_lim_slot_floor(); ++ndrop; continue; }
                                }
                                inv = Real(1.0) / scl;
                            }
                            const Real wf = Wf * inv, wu = Wup * inv;
                            wdotw += wf * wf;
                            wdotu += wf * wu;
                        }
                        //  A wave every slot of which dropped out as noise
                        //  carries no smoothness information: no correction
                        //  (phi = 0), counted lim_wave_noise.
                        if (ndrop > 0 && wdotw == Real(0.0)) {
                            phi = Real(0.0);
                            ps_guard::count_lim_wave_noise();
                        }
                        //  A zero-strength wave contributes nothing whatever phi
                        //  is; leave it at 1 rather than dividing by zero.  A
                        //  wave whose theta is not finite enters at first
                        //  order, phi = 0, counted (O-31).
                        if (wdotw != Real(0.0)) {
                            const Real theta = wdotu / wdotw;
                            if (std::isfinite(theta)) {
                                //  van Leer, phi = (theta+|theta|)/(1+|theta|).
                                const Real at = std::abs(theta);
                                phi = (theta + at) / (Real(1.0) + at);
                            } else {
                                phi = Real(0.0);
                                ps_guard::count_lim_wave_drop();
                            }
                        }
                    }
                    for (int n = 0; n < NVAR; ++n) {
                        Ft[n] += wl * coef0 * phi * wv(i, j, k, l*NVAR + n);
                    }
                }
                //  Mixture correction derived from the phase corrections,
                //  F̃[ρ] = F̃[m1]+F̃[m2], F̃[ρE] = F̃[E1]+F̃[E2] (H-4).  Under
                //  scalar-per-wave limiting these hold already to round-off;
                //  the assignments still act on the unlimited path.
                Ft[URHO]  = Ft[UM1RHO1] + Ft[UM2RHO2];
                Ft[UEDEN] = Ft[UE1] + Ft[UE2];
                //  A correction with a non-finite component is dropped whole
                //  at this face (every component together, so the wave
                //  direction is kept) and the first-order flux is left as it
                //  is; counted (O-31).  This replaces the ps_finite_or on
                //  flx + F̃, which zeroed the whole face flux silently.
                {
                    bool finite = true;
                    for (int n = 0; n < NVAR; ++n) {
                        if (n == UTEMP) continue;
                        if (!std::isfinite(Ft[n])) { finite = false; break; }
                    }
                    if (!finite) {
                        for (int n = 0; n < NVAR; ++n) Ft[n] = Real(0.0);
                        ps_guard::count_lim_face_drop();
                    }
                }
                // Conserved slots: add F̃ to the recovered flux.
                for (int n = 0; n < NVAR; ++n) {
                    if (n==UTEMP || n==UALPHA1 || n==UE1 || n==UE2) continue;
                    flx(i,j,k,n) += Ft[n];
                }
                // Non-conserved slots: stash F̃ for the deposit.
                ft(i,j,k,0) = Ft[UALPHA1];
                ft(i,j,k,1) = Ft[UE1];
                ft(i,j,k,2) = Ft[UE2];
            };
            amrex::ParallelFor(xfbx, [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                correct(0, i,j,k, wxbx, flx1, wp_wave_x, wp_ft_x, dx[0]); });
#if (AMREX_SPACEDIM >= 2)
            amrex::ParallelFor(yfbx, [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                correct(1, i,j,k, wybx, flx2, wp_wave_y, wp_ft_y, dx[1]); });
#endif
#if (AMREX_SPACEDIM == 3)
            amrex::ParallelFor(zfbx, [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                correct(2, i,j,k, wzbx, flx3, wp_wave_z, wp_ft_z, dx[2]); });
#endif
        }

        // ---- Transverse term (ps_wp_tvterm), race-free per-face gather. ----
        //  Each d-face collects the contribution from every t≠d; conserved
        //  slots go to flx_d (refluxes), {α,UE1,UE2} to gtv_d for the Pass-3
        //  deposit.  In 3-D the single-transverse pairs are included; the
        //  double-transverse corner term is not.
        const int gc = tv ? 3 : 1;   // {α,UE1,UE2} transverse deposit store
        amrex::FArrayBox gtv_x_fab(xfbx, gc, amrex::The_Async_Arena());
        auto const& gtv_x = gtv_x_fab.array();
#if (AMREX_SPACEDIM >= 2)
        amrex::FArrayBox gtv_y_fab(yfbx, gc, amrex::The_Async_Arena());
        auto const& gtv_y = gtv_y_fab.array();
#endif
#if (AMREX_SPACEDIM == 3)
        amrex::FArrayBox gtv_z_fab(zfbx, gc, amrex::The_Async_Arena());
        auto const& gtv_z = gtv_z_fab.array();
#endif
#if (AMREX_SPACEDIM >= 2)
        if (tv) {
            // x-faces: transverse from y (+ z in 3D).
            amrex::ParallelFor(xfbx, [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                Real g[NVAR]; for(int n=0;n<NVAR;++n) g[n]=Real(0.0);
                ps_wp_tvterm(0,1,i,j,k, uin_arr, wp_wave_y, domlo,domhi, dt_l, dx[1], g);
#if (AMREX_SPACEDIM == 3)
                ps_wp_tvterm(0,2,i,j,k, uin_arr, wp_wave_z, domlo,domhi, dt_l, dx[2], g);
#endif
                for(int n=0;n<NVAR;++n){
                    if      (n==UALPHA1) gtv_x(i,j,k,0)=ps_finite_or(g[n],Real(0.0));
                    else if (n==UE1)     gtv_x(i,j,k,1)=ps_finite_or(g[n],Real(0.0));
                    else if (n==UE2)     gtv_x(i,j,k,2)=ps_finite_or(g[n],Real(0.0));
                    else if (n!=UTEMP)   flx1(i,j,k,n)=ps_finite_or(flx1(i,j,k,n)+g[n],Real(0.0));
                }
            });
            // y-faces: transverse from x (+ z in 3D).
            amrex::ParallelFor(yfbx, [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                Real g[NVAR]; for(int n=0;n<NVAR;++n) g[n]=Real(0.0);
                ps_wp_tvterm(1,0,i,j,k, uin_arr, wp_wave_x, domlo,domhi, dt_l, dx[0], g);
#if (AMREX_SPACEDIM == 3)
                ps_wp_tvterm(1,2,i,j,k, uin_arr, wp_wave_z, domlo,domhi, dt_l, dx[2], g);
#endif
                for(int n=0;n<NVAR;++n){
                    if      (n==UALPHA1) gtv_y(i,j,k,0)=ps_finite_or(g[n],Real(0.0));
                    else if (n==UE1)     gtv_y(i,j,k,1)=ps_finite_or(g[n],Real(0.0));
                    else if (n==UE2)     gtv_y(i,j,k,2)=ps_finite_or(g[n],Real(0.0));
                    else if (n!=UTEMP)   flx2(i,j,k,n)=ps_finite_or(flx2(i,j,k,n)+g[n],Real(0.0));
                }
            });
#if (AMREX_SPACEDIM == 3)
            // z-faces: transverse from x + y.
            amrex::ParallelFor(zfbx, [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                Real g[NVAR]; for(int n=0;n<NVAR;++n) g[n]=Real(0.0);
                ps_wp_tvterm(2,0,i,j,k, uin_arr, wp_wave_x, domlo,domhi, dt_l, dx[0], g);
                ps_wp_tvterm(2,1,i,j,k, uin_arr, wp_wave_y, domlo,domhi, dt_l, dx[1], g);
                for(int n=0;n<NVAR;++n){
                    if      (n==UALPHA1) gtv_z(i,j,k,0)=ps_finite_or(g[n],Real(0.0));
                    else if (n==UE1)     gtv_z(i,j,k,1)=ps_finite_or(g[n],Real(0.0));
                    else if (n==UE2)     gtv_z(i,j,k,2)=ps_finite_or(g[n],Real(0.0));
                    else if (n!=UTEMP)   flx3(i,j,k,n)=ps_finite_or(flx3(i,j,k,n)+g[n],Real(0.0));
                }
            });
#endif
        }
#endif

        // ---- Viscosity: flux-form term added to flx_d (refluxes and
        //      telescopes through consup). ----
#if (AMREX_SPACEDIM >= 2)
        if (ps_mu > Real(0.0)) {
            amrex::ParallelFor(xfbx, [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                ps_viscous_face(0, i,j,k, uin_arr, flx1, domlo,domhi, ps_mu, dx.data()); });
            amrex::ParallelFor(yfbx, [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                ps_viscous_face(1, i,j,k, uin_arr, flx2, domlo,domhi, ps_mu, dx.data()); });
#if (AMREX_SPACEDIM == 3)
            amrex::ParallelFor(zfbx, [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                ps_viscous_face(2, i,j,k, uin_arr, flx3, domlo,domhi, ps_mu, dx.data()); });
#endif
        }
#endif

        // ---- Pass 3: per-cell deposit of the non-conserved fluctuations.
        //  dsdt(n) = -(A⁺_lowface + A⁻_highface)/dx [- (F̃_hi - F̃_lo)/dx]
        //  [- (G_hi - G_lo)/dx] [- (in-cell term)/dx] summed over
        //  directions, n ∈ {UALPHA1, UE1, UE2}.  Low face of cell i is face
        //  index i (A⁺), high face i+1 (A⁻).  flx is 0 on these slots so
        //  consup adds nothing.
        //  The in-cell term (ps_wp_order=4): the integral of A(q) q_x over
        //  the cell's own reconstruction, between its low edge (the R side
        //  of its low face) and its high edge (the L side of its high face),
        //  with the cell's mean velocity u_bar and mass fractions Y_k: for
        //  alpha, u_bar (alpha_hi - alpha_lo) - K (u_hi - u_lo) with the
        //  Kapila K = alpha_1 alpha_2 (rho_2 c_2^2 - rho_1 c_1^2) /
        //  (alpha_1 rho_2 c_2^2 + alpha_2 rho_1 c_1^2) from the cell's own
        //  phase states (the closure the face solver's star states carry,
        //  MODEL section 2.3); for the phase energies, F_k(q_hi) - F_k(q_lo)
        //  plus Sigma_1 = -Sigma_2 = -u_bar [Y_2 d(alpha_1 P) - Y_1
        //  d(alpha_2 P)] at the mixture pressure (Pelanti-Shyue 2014 eq. 2d,
        //  H-3).  The edge values are the ones the fluctuations used, so for
        //  a uniform velocity the alpha update is the upwind difference of
        //  the edges and consistent with the mass update (the Abgrall
        //  condition).  docs/PLAN_high_order.pdf section 4, Stage 3.
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real da = Real(0.0), de1 = Real(0.0), de2 = Real(0.0);
            Real ub[3] = {Real(0.0), Real(0.0), Real(0.0)};
            Real Y1b = Real(0.0), Y2b = Real(0.0), Kap = Real(0.0);
            if (o4) {
                const Real rho = amrex::max(ps_finite_or(uin_arr(i,j,k,URHO), Real(ps_const::RHO_MIX_MIN)),
                                            Real(ps_const::RHO_MIX_MIN));
                ub[0] = ps_finite_or(uin_arr(i,j,k,UMX), Real(0.0)) / rho;
#if (AMREX_SPACEDIM >= 2)
                ub[1] = ps_finite_or(uin_arr(i,j,k,UMY), Real(0.0)) / rho;
#endif
#if (AMREX_SPACEDIM == 3)
                ub[2] = ps_finite_or(uin_arr(i,j,k,UMZ), Real(0.0)) / rho;
#endif
                const Real m1 = amrex::max(ps_finite_or(uin_arr(i,j,k,UM1RHO1), Real(0.0)), Real(0.0));
                const Real m2 = amrex::max(ps_finite_or(uin_arr(i,j,k,UM2RHO2), Real(0.0)), Real(0.0));
                Real ms = m1 + m2; if (!(ms > Real(1.0e-30))) ms = Real(1.0e-30);
                Y1b = m1 / ms; Y2b = Real(1.0) - Y1b;
                Real Uc[NVAR];
                for (int n = 0; n < NVAR; ++n) Uc[n] = uin_arr(i,j,k,n);
                const PsPhaseSpeeds sp = ps_phase_speeds_from_state(Uc, l_pres);
                if (sp.rg1 != PsRegime::Absent && sp.rg2 != PsRegime::Absent) {
                    const Real k1 = sp.rho1 * sp.c1 * sp.c1;
                    const Real k2 = sp.rho2 * sp.c2 * sp.c2;
                    const Real den = sp.alpha1 * k2 + sp.alpha2 * k1;
                    if (den > Real(0.0) && std::isfinite(den)) {
                        Kap = ps_finite_or(sp.alpha1 * sp.alpha2 * (k2 - k1) / den, Real(0.0));
                    }
                }
            }
            //  The in-cell term of one direction from the face store: lo is
            //  the cell's low face (its R-side comps), hi its high face (its
            //  L-side comps).
            auto incell = [&] (amrex::Array4<amrex::Real> const& w,
                               int li, int lj, int lk, int hi_, int hj, int hk,
                               Real u_bar, Real inv) noexcept {
                const Real aH = w(hi_,hj,hk,12), aL = w(li,lj,lk,13);
                const Real PH = w(hi_,hj,hk,10), PL = w(li,lj,lk,11);
                const Real dalpha = aH - aL;
                const Real du     = w(hi_,hj,hk,14) - w(li,lj,lk,15);
                const Real dF1    = w(hi_,hj,hk,6)  - w(li,lj,lk,8);
                const Real dF2    = w(hi_,hj,hk,7)  - w(li,lj,lk,9);
                const Real da1P   = ps_finite_or(aH * PH, Real(0.0)) - ps_finite_or(aL * PL, Real(0.0));
                const Real da2P   = ps_finite_or((Real(1.0) - aH) * PH, Real(0.0)) - ps_finite_or((Real(1.0) - aL) * PL, Real(0.0));
                const Real sig1   = -u_bar * (Y2b * da1P - Y1b * da2P);
                da  -= (u_bar * dalpha - Kap * du) * inv;
                de1 -= (dF1 + sig1) * inv;
                de2 -= (dF2 - sig1) * inv;
            };
            {
                const Real inv = Real(1.0) / dx[0];
                da  -= (wp_fluct_x(i,j,k,1) + wp_fluct_x(i+1,j,k,0)) * inv;
                de1 -= (wp_fluct_x(i,j,k,3) + wp_fluct_x(i+1,j,k,2)) * inv;
                de2 -= (wp_fluct_x(i,j,k,5) + wp_fluct_x(i+1,j,k,4)) * inv;
                if (o4) incell(wp_fluct_x, i,j,k, i+1,j,k, ub[0], inv);
                if (o2) {
                    da  -= (wp_ft_x(i+1,j,k,0) - wp_ft_x(i,j,k,0)) * inv;
                    de1 -= (wp_ft_x(i+1,j,k,1) - wp_ft_x(i,j,k,1)) * inv;
                    de2 -= (wp_ft_x(i+1,j,k,2) - wp_ft_x(i,j,k,2)) * inv;
                }
                if (tv) {
                    da  -= (gtv_x(i+1,j,k,0) - gtv_x(i,j,k,0)) * inv;
                    de1 -= (gtv_x(i+1,j,k,1) - gtv_x(i,j,k,1)) * inv;
                    de2 -= (gtv_x(i+1,j,k,2) - gtv_x(i,j,k,2)) * inv;
                }
            }
#if (AMREX_SPACEDIM >= 2)
            {
                const Real inv = Real(1.0) / dx[1];
                da  -= (wp_fluct_y(i,j,k,1) + wp_fluct_y(i,j+1,k,0)) * inv;
                de1 -= (wp_fluct_y(i,j,k,3) + wp_fluct_y(i,j+1,k,2)) * inv;
                de2 -= (wp_fluct_y(i,j,k,5) + wp_fluct_y(i,j+1,k,4)) * inv;
                if (o4) incell(wp_fluct_y, i,j,k, i,j+1,k, ub[1], inv);
                if (o2) {
                    da  -= (wp_ft_y(i,j+1,k,0) - wp_ft_y(i,j,k,0)) * inv;
                    de1 -= (wp_ft_y(i,j+1,k,1) - wp_ft_y(i,j,k,1)) * inv;
                    de2 -= (wp_ft_y(i,j+1,k,2) - wp_ft_y(i,j,k,2)) * inv;
                }
                if (tv) {
                    da  -= (gtv_y(i,j+1,k,0) - gtv_y(i,j,k,0)) * inv;
                    de1 -= (gtv_y(i,j+1,k,1) - gtv_y(i,j,k,1)) * inv;
                    de2 -= (gtv_y(i,j+1,k,2) - gtv_y(i,j,k,2)) * inv;
                }
            }
#endif
#if (AMREX_SPACEDIM == 3)
            {
                const Real inv = Real(1.0) / dx[2];
                da  -= (wp_fluct_z(i,j,k,1) + wp_fluct_z(i,j,k+1,0)) * inv;
                de1 -= (wp_fluct_z(i,j,k,3) + wp_fluct_z(i,j,k+1,2)) * inv;
                de2 -= (wp_fluct_z(i,j,k,5) + wp_fluct_z(i,j,k+1,4)) * inv;
                if (o4) incell(wp_fluct_z, i,j,k, i,j,k+1, ub[2], inv);
                if (o2) {
                    da  -= (wp_ft_z(i,j,k+1,0) - wp_ft_z(i,j,k,0)) * inv;
                    de1 -= (wp_ft_z(i,j,k+1,1) - wp_ft_z(i,j,k,1)) * inv;
                    de2 -= (wp_ft_z(i,j,k+1,2) - wp_ft_z(i,j,k,2)) * inv;
                }
                if (tv) {
                    da  -= (gtv_z(i,j,k+1,0) - gtv_z(i,j,k,0)) * inv;
                    de1 -= (gtv_z(i,j,k+1,1) - gtv_z(i,j,k,1)) * inv;
                    de2 -= (gtv_z(i,j,k+1,2) - gtv_z(i,j,k,2)) * inv;
                }
            }
#endif
            dsdt_arr(i,j,k, UALPHA1) = ps_finite_or(da,  Real(0.0));
            dsdt_arr(i,j,k, UE1)     = ps_finite_or(de1, Real(0.0));
            dsdt_arr(i,j,k, UE2)     = ps_finite_or(de2, Real(0.0));
        });

        // ---- Coarse-fine treatment ----
        //  Conserved slots reflux through the AMReX FluxRegister via flx
        //  (F* plus the flux-form correction, which Berger & LeVeque show
        //  refluxes conservatively even for a non-conservative first-order
        //  operator).  Alpha is corrected by the capacity-form co-move with
        //  its refluxed mass in PSAMR::reflux() (PSAMR.ps_bl_reflux), which
        //  needs no wp-specific data.  The first-order UE1/UE2 phase-split
        //  coarse-fine fix-up is not done (gaps ~1e-4, resolution-dominated).
        //  docs §4.9.
    } // ===== end wave-propagation interior =====
}

#else  // !USE_PS_HYDRO

void
PS_umeth(const amrex::Box& /*bx*/,
         const int* /*bclo*/, const int* /*bchi*/,
         const int* /*domlo*/, const int* /*domhi*/,
         amrex::Array4<const amrex::Real> const& /*uin_arr*/,
         amrex::Array4<const amrex::Real> const& /*q*/,
         amrex::Array4<const amrex::Real> const& /*qa*/,
         amrex::Array4<amrex::Real> const& /*dsdt_arr*/,
         AMREX_D_DECL(amrex::Array4<amrex::Real> const& /*flx1*/,
                      amrex::Array4<amrex::Real> const& /*flx2*/,
                      amrex::Array4<amrex::Real> const& /*flx3*/),
         AMREX_D_DECL(amrex::Array4<amrex::Real> const& /*q1*/,
                      amrex::Array4<amrex::Real> const& /*q2*/,
                      amrex::Array4<amrex::Real> const& /*q3*/),
         AMREX_D_DECL(amrex::Array4<const amrex::Real> const& /*a1*/,
                      amrex::Array4<const amrex::Real> const& /*a2*/,
                      amrex::Array4<const amrex::Real> const& /*a3*/),
         amrex::Array4<amrex::Real> const& /*pdivu*/,
         amrex::Array4<const amrex::Real> const& /*vol*/,
         const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*dx*/,
         const amrex::Real /*dt*/,
         const amrex::Real /*small*/,
         const amrex::Real /*small_dens*/,
         const amrex::Real /*small_pres*/,
         const amrex::Real /*smallu*/,
         const int /*slope_order*/,
         const PassMap* /*lpmap*/)
{
    amrex::Abort("PS_umeth called without USE_PS_HYDRO — bug in build system.");
}

#endif // USE_PS_HYDRO
