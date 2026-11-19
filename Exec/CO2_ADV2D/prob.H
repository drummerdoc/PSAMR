#ifndef PROB_H
#define PROB_H

// =====================================================================
//  Exec/CO2_ADV2D  —  SMOOTH 2D diagonal two-phase advection benchmark.
//
//  Uniform pressure P0 and uniform velocity (u0,v0); a smooth periodic
//  volume-fraction field
//     α₁(x,y) = α_mean + α_amp·sin(2πk_x x/Lx)·sin(2πk_y y/Ly)
//  is advected diagonally.  Phase 1 (liquid branch) at (P0,T1), phase 2
//  (vapor/SC branch) at (P0,T2); both at the SAME pressure, so there is
//  no acoustic wave and the exact solution is a pure translation.  Used
//  as the PS order-of-accuracy harness (task #21).  Periodic BCs.
// =====================================================================

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <cmath>

#include "IndexDefines.H"
#include "prob_parm.H"
#include "PSAMR_params.H"
#include "EOS.H"

// The analytic state at the point (x, y), into s[NVAR].
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
adv2d_state(amrex::Real x, amrex::Real y, const amrex::Real* prob_lo,
            amrex::Real Lx, amrex::Real Ly,
            ProbParmDevice const& prob_parm, amrex::Real* s)
{
    constexpr amrex::Real twopi = amrex::Real(6.283185307179586);

    // Smooth periodic volume-fraction field (kept safely inside (0,1)).
    // alpha_pow (default 1) applies a sign-preserving exponent to each sin
    // factor: pow=3 gives sin³·sin³, a C^∞ but steeper profile.
    amrex::Real sx = std::sin(twopi * prob_parm.k_x * (x - prob_lo[0]) / Lx);
    amrex::Real sy = (prob_parm.oned != 0) ? amrex::Real(1.0)
                   : std::sin(twopi * prob_parm.k_y * (y - prob_lo[1]) / Ly);
    if (prob_parm.alpha_pow != amrex::Real(1.0)) {
        sx = std::copysign(std::pow(std::abs(sx), prob_parm.alpha_pow), sx);
        sy = std::copysign(std::pow(std::abs(sy), prob_parm.alpha_pow), sy);
    }
    amrex::Real alpha_1 = prob_parm.alpha_mean
        + prob_parm.alpha_amp * sx * sy;
    constexpr amrex::Real afloor = amrex::Real(1.0e-6);
    if (alpha_1 < afloor)                    alpha_1 = afloor;
    if (alpha_1 > amrex::Real(1.0) - afloor) alpha_1 = amrex::Real(1.0) - afloor;
    const amrex::Real alpha_2 = amrex::Real(1.0) - alpha_1;

    const amrex::Real u0 = prob_parm.u0;
    const amrex::Real v0 = prob_parm.v0;

    amrex::Real Y[NUM_SPECIES];
    Y[0] = amrex::Real(1.0);
    for (int n = 1; n < NUM_SPECIES; ++n) Y[n] = amrex::Real(0.0);

#ifdef USE_PS_HYDRO
    // Both phases initialised from the SAME branch and SAME (P0,T1)
    // state, so ρ₁=ρ₂, e₁=e₂ and the per-phase pressures are IDENTICALLY
    // equal (P₁≡P₂=P0).  This guarantees exact mechanical equilibrium
    // and a uniform mixture pressure, so the flow is a pure passive
    // advection of the α₁ colour at constant velocity (no acoustics).
    // NOTE: liquid and vapor branches are different EOS roots that do
    // NOT round-trip to the same pressure, so mixing branches (as in a
    // real two-phase contact) breaks the equilibrium — see the design
    // note in prob_parm.H.  For an order-of-accuracy transport test we
    // want the clean colour-advection, hence a single branch.
    // The instruments of follow-up 12: a smooth phase-1 temperature
    // variation and a Gaussian pressure pulse (prob_parm.H); both leave the
    // plain case bitwise when their amplitudes are 0.
    const amrex::Real T1l = prob_parm.T1 + prob_parm.T1_amp * sx * sy;
    const amrex::Real xc = prob_lo[0] + amrex::Real(0.5) * Lx, yc = prob_lo[1] + amrex::Real(0.5) * Ly;
    const amrex::Real w2 = (prob_parm.p_width * Lx) * (prob_parm.p_width * Lx);
    const amrex::Real Pl = prob_parm.P0 * (amrex::Real(1.0)
        + prob_parm.p_amp * std::exp(-((x - xc) * (x - xc) + (y - yc) * (y - yc)) / w2));
    amrex::Real rho_1, e_1;
    EOS::PYT2RE_liquid(Pl, Y, T1l, rho_1, e_1);
    //  Phase 2 on its own (vapour) branch at the same P0: the EOS is
    //  branch-locked, so a phase-2 slot holding the liquid state evaluates
    //  to a different pressure (6.37 MPa against 8 MPa, measured 15 Nov
    //  2026) and the mixture pressure then varies with alpha_1, which
    //  launches acoustic waves and voids the translated-IC exact solution.
    amrex::Real rho_2, e_2;
    EOS::PYT2RE_vapor(Pl, Y, prob_parm.T2, rho_2, e_2);

    const amrex::Real m_1 = alpha_1 * rho_1;
    const amrex::Real m_2 = alpha_2 * rho_2;
    const amrex::Real rho = m_1 + m_2;

    const amrex::Real e_mix = (m_1 * e_1 + m_2 * e_2) / rho;
    const amrex::Real ke    = amrex::Real(0.5) * rho * (u0 * u0 + v0 * v0);

    s[URHO] = rho;
    s[UMX]  = rho * u0;
    s[UMY]  = rho * v0;
#if (AMREX_SPACEDIM == 3)
    s[UMZ]  = amrex::Real(0.0);
#endif
    s[UEINT] = rho * e_mix;
    s[UEDEN] = rho * e_mix + ke;
    // Mixture temperature is only a diagnostic seed; recovered downstream.
    s[UTEMP] = amrex::Real(0.5) * (prob_parm.T1 + prob_parm.T2);
    for (int n = 0; n < NUM_SPECIES; ++n) s[UFS + n] = Y[n] * rho;

    const amrex::Real half_u2 = amrex::Real(0.5) * (u0 * u0 + v0 * v0);
    s[UALPHA1] = alpha_1;
    s[UM1RHO1] = m_1;
    s[UM2RHO2] = m_2;
    s[UE1] = m_1 * (e_1 + half_u2);
    s[UE2] = m_2 * (e_2 + half_u2);
#else
    amrex::Real rho0, e0;
    EOS::PYT2RE(prob_parm.P0, Y, prob_parm.T1, rho0, e0);
    s[URHO] = rho0;
    s[UMX]  = rho0 * u0;
    s[UMY]  = rho0 * v0;
#if (AMREX_SPACEDIM == 3)
    s[UMZ]  = amrex::Real(0.0);
#endif
    const amrex::Real ke_sf = amrex::Real(0.5) * rho0 * (u0 * u0 + v0 * v0);
    s[UEINT] = rho0 * e0;
    s[UEDEN] = rho0 * e0 + ke_sf;
    s[UTEMP] = prob_parm.T1;
    for (int n = 0; n < NUM_SPECIES; ++n) s[UFS + n] = Y[n] * rho0;
    amrex::ignore_unused(alpha_2);
#endif
}

// Every cell: the analytic state at its centre (prob.avg_init = 0, the
// point sample the order harness has always used) or its cell average by
// the three-point Gauss rule per direction (avg_init = 1), which is what a
// finite-volume scheme with the average-to-point conversions expects.
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
PSAMR_initdata( int i, int j, int k,
               amrex::Array4<amrex::Real> const& state,
               amrex::GeometryData const& geomdata,
               ProbParmDevice const& prob_parm)
{
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx      = geomdata.CellSize();
    const amrex::Real  x       = prob_lo[0] + (amrex::Real(i) + amrex::Real(0.5)) * dx[0];
    const amrex::Real  y       = prob_lo[1] + (amrex::Real(j) + amrex::Real(0.5)) * dx[1];
    const amrex::Real Lx = prob_hi[0] - prob_lo[0];
    const amrex::Real Ly = prob_hi[1] - prob_lo[1];
    amrex::Real s[NVAR];
    for (int n = 0; n < NVAR; ++n) s[n] = amrex::Real(0.0);
    if (prob_parm.avg_init == 0) {
        adv2d_state(x, y, prob_lo, Lx, Ly, prob_parm, s);
    } else {
        const amrex::Real g = amrex::Real(0.5) * std::sqrt(amrex::Real(0.6));
        const amrex::Real xs[3] = { -g, amrex::Real(0.0), g };
        const amrex::Real w[3]  = { amrex::Real(5.0/18.0), amrex::Real(8.0/18.0), amrex::Real(5.0/18.0) };
        amrex::Real tmp[NVAR];
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                for (int n = 0; n < NVAR; ++n) tmp[n] = amrex::Real(0.0);
                adv2d_state(x + xs[a] * dx[0], y + xs[b] * dx[1], prob_lo, Lx, Ly, prob_parm, tmp);
                for (int n = 0; n < NVAR; ++n) s[n] += w[a] * w[b] * tmp[n];
            }
        }
    }
    for (int n = 0; n < NVAR; ++n) state(i,j,k,n) = s[n];
}


// Periodic domain — bcnormal never invoked on periodic faces, but the
// signature must exist.  Zero-gradient copy as a safe fallback.
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real* /*x*/,
  const amrex::Real s_int[NVAR],
  amrex::Real s_ext[NVAR],
  const int /*idir*/,
  const int /*sgn*/,
  const amrex::Real /*time*/,
  amrex::GeometryData const& /*geomdata*/,
  ProbParmDevice const& /*prob_parm*/)
{
    for (int n = 0; n < NVAR; ++n) s_ext[n] = s_int[n];
}

#endif
