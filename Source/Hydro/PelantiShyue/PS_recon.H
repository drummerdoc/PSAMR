#ifndef PS_RECON_H
#define PS_RECON_H
// =====================================================================
//  PS_recon.H  —  reconstructed face states for the wave-propagation
//                 interior (PSAMR.ps_wp_order=4, docs/PLAN_high_order.pdf
//                 section 4 and Stage 3).
//
//  Per cell and direction, the two edge states of the cell are built
//  from the five cell averages i-2..i+2 by WENO5-Z (ps_wp_recon=weno5z)
//  or PPM (ps_wp_recon=ppm), component by component on the primitive set
//  (alpha_1, rho_1, rho_2, u, v, w, e_1, e_2) and converted back to the
//  NVAR conserved array with the identities rho = m_1 + m_2 and
//  rhoE = E_1 + E_2 holding by construction.  The primitive set is the
//  one in which a uniform-velocity, uniform-pressure contact stays
//  uniform in every variable but alpha, so the reconstruction cannot
//  disturb it (the Abgrall condition, Stage 3 item 4); the characteristic
//  projection of the plan's section 4 needs the per-phase Grueneisen
//  derivatives, which the EOS interface does not expose, and is a named
//  follow-up (Stage 3 record).
//
//  Every edge formula is written in deviation form about the cell's own
//  average, so uniform data reconstruct bitwise uniform (the Stage 2b
//  lesson: a round-off asymmetry on uniform data is amplified by the 2-D
//  scheme), and the left edge is the right-edge formula on the reversed
//  stencil, so a mirrored stencil gives the mirrored value exactly.
//
//  Presence rules: a cell whose five-cell stencil has a phase Absent in
//  some cells and present in others takes its raw averages at both edges
//  (first order at a presence front, as the correction path is there);
//  a phase Absent in the cell is pinned absent at its edges; an edge
//  that leaves the admissible set (alpha outside [0,1], a present
//  phase's alpha at or below alpha_vanish, a non-positive or non-finite
//  phase density, a non-finite energy or velocity) returns the cell to
//  its raw averages at both edges.  Every fallback is counted
//  (recon_fallback, host builds).
//
//  The contact taper (ps_wp_contact_taper=1): the H-5 weight
//  w = 1 - smoothstep(|dalpha|/alpha_cond) of the face, dalpha on the
//  raw averages, scales the whole high-order increment of the edge
//  state on that face's side: V_edge = V_bar + w (V_recon - V_bar), so
//  at w = 0 the face sees the raw averages, the present first-order
//  behaviour at a material interface.  0 = the reconstruction's own
//  smoothness measure is the only interface treatment (the default,
//  Marc's decision on the Stage 3 record).
//
//  Device-safe: no statics, no ParmParse, no EOS calls (the primitive
//  set needs none; the EOS enters at the face through ps_wp_face as
//  before).
// =====================================================================
#ifdef USE_PS_HYDRO

#include <AMReX_REAL.H>
#include <AMReX_Array4.H>
#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>

#include "IndexDefines.H"
#include "PS_presence.H"
#include "PS_constants.H"
#include "PS_util.H"
#include "PS_guards.H"

namespace PS_RECON {

using amrex::Real;

constexpr int NPRIM = 5 + AMREX_SPACEDIM;   // alpha_1, rho_1, rho_2, e_1, e_2, u[DIM]
constexpr int IA1 = 0, IR1 = 1, IR2 = 2, IE1 = 3, IE2 = 4, IU = 5;

// ---------------------------------------------------------------------
//  WENO5-Z right-edge value of the centre cell c from the averages
//  (a, b, c, d, e) = (q_{i-2}, ..., q_{i+2}) (Borges, Carmona, Costa,
//  Don 2008: alpha_k = d_k (1 + tau_5 / (beta_k + eps)), tau_5 =
//  |beta_0 - beta_2|).  Deviation form: the three candidate values and
//  the smoothness indicators are built from differences to c, and the
//  result is c plus the weighted deviation, so uniform data give c
//  bitwise.  eps is 1e-40, below any smoothness indicator the data can
//  carry, so the weights are scale-invariant (the same weights for
//  alpha and for an energy with the same shape).  wpow is the power on
//  the ratio tau/beta_k (PSAMR.ps_wp_weno_power: 1 = Borges, 2 = the
//  higher power that pushes a marginal stencil harder toward ENO;
//  docs/PLAN_high_order.pdf follow-up 2); 1 is bitwise the plain ratio.
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real weno5z_right (Real a, Real b, Real c, Real d, Real e, int wpow = 1) noexcept
{
    const Real da = a - c, db = b - c, dd = d - c, de = e - c;
    // the three candidate values, as deviations from c
    const Real q0 = (Real(2.0)*da - Real(7.0)*db) / Real(6.0);   // (2a - 7b + 11c)/6
    const Real q1 = (-db + Real(2.0)*dd) / Real(6.0);            // (-b + 5c + 2d)/6
    const Real q2 = (Real(5.0)*dd - de) / Real(6.0);             // (2c + 5d - e)/6
    // Jiang-Shu smoothness indicators, every term a difference to c
    const Real s0 = da - Real(2.0)*db;    // a - 2b + c
    const Real t0 = da - Real(4.0)*db;    // a - 4b + 3c
    const Real s1 = db + dd;              // b - 2c + d
    const Real t1 = db - dd;              // b - d
    const Real s2 = de - Real(2.0)*dd;    // c - 2d + e
    const Real t2 = de - Real(4.0)*dd;    // 3c - 4d + e
    const Real k13 = Real(13.0) / Real(12.0);
    const Real beta0 = k13 * s0 * s0 + Real(0.25) * t0 * t0;
    const Real beta1 = k13 * s1 * s1 + Real(0.25) * t1 * t1;
    const Real beta2 = k13 * s2 * s2 + Real(0.25) * t2 * t2;
    const Real tau = std::abs(beta0 - beta2);
    constexpr Real eps = Real(1.0e-40);
    Real r0 = tau / (beta0 + eps), r1 = tau / (beta1 + eps), r2 = tau / (beta2 + eps);
    if (wpow == 2) { r0 *= r0; r1 *= r1; r2 *= r2; }
    const Real w0 = Real(0.1) * (Real(1.0) + r0);
    const Real w1 = Real(0.6) * (Real(1.0) + r1);
    const Real w2 = Real(0.3) * (Real(1.0) + r2);
    const Real ws = w0 + w1 + w2;
    return c + (w0 * q0 + w1 * q1 + w2 * q2) / ws;
}

//  Left-edge value of the centre cell: the right-edge formula on the
//  reversed stencil (mirror-exact).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real weno5z_left (Real a, Real b, Real c, Real d, Real e, int wpow = 1) noexcept
{
    return weno5z_right(e, d, c, b, a, wpow);
}

// ---------------------------------------------------------------------
//  PPM (Colella-Woodward 1984) edge values of the centre cell: the
//  fourth-order interface interpolant on both edges, then the
//  monotonicity limiter, written symmetrically so the mirrored stencil
//  gives the mirrored pair.  Deviation form as above.
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ppm_edges (Real a, Real b, Real c, Real d, Real e, Real& ql, Real& qr) noexcept
{
    const Real da = a - c, db = b - c, dd = d - c, de = e - c;
    Real dl = (Real(7.0)*db - da - dd) / Real(12.0);   // q_{i-1/2} - c
    Real dr = (Real(7.0)*dd - db - de) / Real(12.0);   // q_{i+1/2} - c
    if (dr * dl >= Real(0.0)) {
        // c is an extremum of the interpolant: flat
        dl = Real(0.0); dr = Real(0.0);
    } else {
        const Real dq = dr - dl;                 // qr - ql
        const Real m  = -(dl + dr);              // 2 (c - (ql+qr)/2)
        if (Real(3.0) * dq * m > dq * dq) {
            dl = -Real(2.0) * dr;                // ql = 3c - 2qr
        } else if (Real(3.0) * dq * m < -(dq * dq)) {
            dr = -Real(2.0) * dl;                // qr = 3c - 2ql
        }
    }
    ql = c + dl;
    qr = c + dr;
}

// ---------------------------------------------------------------------
//  The primitive set of one cell from its conserved array, without an
//  EOS call: the clamped alpha, the checked phase quotients (rho_k, e_k)
//  of PS_presence.H with the mixture placeholders of PS_state.H for a
//  phase with no state, the velocity.  ab_k says the phase is Absent.
// ---------------------------------------------------------------------
struct PsPrim {
    Real v[NPRIM];
    bool ab1, ab2;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PsPrim prim_from_cons (const Real U[NVAR], const PsPres& pr) noexcept
{
    PsPrim p;
    const Real rho = amrex::max(ps_finite_or(U[URHO], Real(ps_const::RHO_MIX_MIN)),
                                Real(ps_const::RHO_MIX_MIN));
    const Real inv = Real(1.0) / rho;
    const Real ux = ps_finite_or(U[UMX], Real(0.0)) * inv;
#if (AMREX_SPACEDIM >= 2)
    const Real uy = ps_finite_or(U[UMY], Real(0.0)) * inv;
#else
    const Real uy = Real(0.0);
#endif
#if (AMREX_SPACEDIM == 3)
    const Real uz = ps_finite_or(U[UMZ], Real(0.0)) * inv;
#else
    const Real uz = Real(0.0);
#endif
    const Real ke = ps_kinetic_energy(ux, uy, uz);
    const Real e_mix = ps_finite_or(U[UEINT], Real(0.0)) * inv;
    Real a1 = ps_finite_or(U[UALPHA1], Real(1.0));
    if (a1 < Real(0.0)) a1 = Real(0.0);
    if (a1 > Real(1.0)) a1 = Real(1.0);
    const Real a2 = Real(1.0) - a1;
    const Real m1 = amrex::max(ps_finite_or(U[UM1RHO1], Real(0.0)), Real(0.0));
    const Real m2 = amrex::max(ps_finite_or(U[UM2RHO2], Real(0.0)), Real(0.0));
    p.ab1 = (ps_regime(a1, m1, pr) == PsRegime::Absent);
    p.ab2 = (ps_regime(a2, m2, pr) == PsRegime::Absent);
    const PsPhaseQuot q1 = ps_phase_quot(a1, m1, ps_finite_or(U[UE1], Real(0.0)), ke, pr);
    const PsPhaseQuot q2 = ps_phase_quot(a2, m2, ps_finite_or(U[UE2], Real(0.0)), ke, pr);
    p.v[IA1] = a1;
    p.v[IR1] = q1.exists ? q1.rho : rho;
    p.v[IR2] = q2.exists ? q2.rho : rho;
    p.v[IE1] = q1.exists ? q1.e   : e_mix;
    p.v[IE2] = q2.exists ? q2.e   : e_mix;
    p.v[IU + 0] = ux;
#if (AMREX_SPACEDIM >= 2)
    p.v[IU + 1] = uy;
#endif
#if (AMREX_SPACEDIM == 3)
    p.v[IU + 2] = uz;
#endif
    return p;
}

//  The conserved array of a primitive set; Y[] the species mass
//  fractions carried from the cell.  rho = m_1 + m_2 and rhoE = E_1 +
//  E_2 by construction; UTEMP is 0 (recomputed at ctoprim, as the face
//  flux leaves it).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cons_from_prim (const Real v[NPRIM], bool ab1, bool ab2,
                     const Real Y[NUM_SPECIES], Real U[NVAR]) noexcept
{
    const Real a1 = ab1 ? Real(0.0) : (ab2 ? Real(1.0) : v[IA1]);
    const Real a2 = Real(1.0) - a1;
    const Real m1 = ab1 ? Real(0.0) : a1 * v[IR1];
    const Real m2 = ab2 ? Real(0.0) : a2 * v[IR2];
    const Real rho = m1 + m2;
    const Real ux = v[IU + 0];
#if (AMREX_SPACEDIM >= 2)
    const Real uy = v[IU + 1];
#else
    const Real uy = Real(0.0);
#endif
#if (AMREX_SPACEDIM == 3)
    const Real uz = v[IU + 2];
#else
    const Real uz = Real(0.0);
#endif
    const Real ke = ps_kinetic_energy(ux, uy, uz);
    for (int n = 0; n < NVAR; ++n) U[n] = Real(0.0);
    U[URHO ] = rho;
    U[UMX  ] = rho * ux;
#if (AMREX_SPACEDIM >= 2)
    U[UMY  ] = rho * uy;
#endif
#if (AMREX_SPACEDIM == 3)
    U[UMZ  ] = rho * uz;
#endif
    U[UE1  ] = m1 * (v[IE1] + ke);
    U[UE2  ] = m2 * (v[IE2] + ke);
    U[UEDEN] = U[UE1] + U[UE2];
    U[UEINT] = m1 * v[IE1] + m2 * v[IE2];
    U[UTEMP] = Real(0.0);
    for (int n = 0; n < NUM_SPECIES; ++n) U[UFS + n] = rho * Y[n];
    U[UALPHA1] = a1;
    U[UM1RHO1] = m1;
    U[UM2RHO2] = m2;
}

//  The H-5 keep-weight of a face from the alpha jump across it (MODEL
//  section 3.4): 1 - smoothstep(|dalpha| / alpha_cond).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real contact_keep (Real dalpha, Real alpha_cond) noexcept
{
    Real t = std::abs(dalpha) / amrex::max(alpha_cond, Real(1.0e-30));
    if (t < Real(0.0)) t = Real(0.0);
    if (t > Real(1.0)) t = Real(1.0);
    return Real(1.0) - t * t * (Real(3.0) - Real(2.0) * t);
}

// ---------------------------------------------------------------------
//  cell_edges: the two edge states (conserved arrays) of cell (i,j,k)
//  in direction idir, written to rq(i,j,k, 0..NVAR-1) (low edge) and
//  rq(i,j,k, NVAR..2NVAR-1) (high edge).  recon: 0 = WENO5-Z, 1 = PPM.
//  taper: the contact-taper option.  Returns true when the cell fell
//  back to its raw averages (the caller counts).
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool cell_edges (int idir, int i, int j, int k,
                 amrex::Array4<const Real> const& uin,
                 amrex::Array4<Real> const& rq,
                 int recon, int taper, const PsPres& pr, int wpow = 1) noexcept
{
    const int oi = (idir == 0), oj = (idir == 1), ok = (idir == 2);
    PsPrim P[5];
    Real Uc[NVAR];
    for (int s = -2; s <= 2; ++s) {
        Real U[NVAR];
        for (int n = 0; n < NVAR; ++n) U[n] = uin(i + s*oi, j + s*oj, k + s*ok, n);
        P[s + 2] = prim_from_cons(U, pr);
        if (s == 0) { for (int n = 0; n < NVAR; ++n) Uc[n] = U[n]; }
    }
    const PsPrim& C = P[2];
    // species mass fractions of the cell, carried to both edges
    Real Y[NUM_SPECIES];
    {
        const Real rho = amrex::max(ps_finite_or(Uc[URHO], Real(ps_const::RHO_MIX_MIN)),
                                    Real(ps_const::RHO_MIX_MIN));
        for (int n = 0; n < NUM_SPECIES; ++n)
            Y[n] = ps_finite_or(Uc[UFS + n], Real(0.0)) / rho;
    }

    bool fallback = false;
    // A presence front inside the stencil: raw averages.
    for (int s = 0; s < 5; ++s) {
        if (s == 2) continue;
        if (P[s].ab1 != C.ab1 || P[s].ab2 != C.ab2) { fallback = true; break; }
    }

    Real vl[NPRIM], vr[NPRIM];
    if (!fallback) {
        for (int m = 0; m < NPRIM; ++m) {
            const Real a = P[0].v[m], b = P[1].v[m], c = P[2].v[m],
                       d = P[3].v[m], e = P[4].v[m];
            if (recon == 1) {
                ppm_edges(a, b, c, d, e, vl[m], vr[m]);
            } else {
                vl[m] = weno5z_left (a, b, c, d, e, wpow);
                vr[m] = weno5z_right(a, b, c, d, e, wpow);
            }
        }
        if (taper != 0) {
            const Real wl = contact_keep(C.v[IA1] - P[1].v[IA1], pr.alpha_cond);
            const Real wr = contact_keep(P[3].v[IA1] - C.v[IA1], pr.alpha_cond);
            for (int m = 0; m < NPRIM; ++m) {
                vl[m] = C.v[m] + wl * (vl[m] - C.v[m]);
                vr[m] = C.v[m] + wr * (vr[m] - C.v[m]);
            }
        }
        // Admissibility of both edges.
        auto admissible = [&] (const Real* v) noexcept -> bool {
            if (!(v[IA1] >= Real(0.0) && v[IA1] <= Real(1.0))) return false;
            if (!C.ab1) {
                if (!(v[IA1] > pr.alpha_vanish)) return false;
                if (!(v[IR1] > Real(0.0)) || !std::isfinite(v[IR1]) || !std::isfinite(v[IE1])) return false;
            }
            if (!C.ab2) {
                if (!(Real(1.0) - v[IA1] > pr.alpha_vanish)) return false;
                if (!(v[IR2] > Real(0.0)) || !std::isfinite(v[IR2]) || !std::isfinite(v[IE2])) return false;
            }
            for (int m = 0; m < AMREX_SPACEDIM; ++m) if (!std::isfinite(v[IU + m])) return false;
            return true;
        };
        if (!admissible(vl) || !admissible(vr)) fallback = true;
    }
    if (fallback) {
        for (int m = 0; m < NPRIM; ++m) { vl[m] = C.v[m]; vr[m] = C.v[m]; }
    }
    Real UL[NVAR], UR[NVAR];
    cons_from_prim(vl, C.ab1, C.ab2, Y, UL);
    cons_from_prim(vr, C.ab1, C.ab2, Y, UR);
    for (int n = 0; n < NVAR; ++n) {
        rq(i, j, k, n)        = UL[n];
        rq(i, j, k, NVAR + n) = UR[n];
    }
    return fallback;
}

} // namespace PS_RECON

#endif // USE_PS_HYDRO
#endif // PS_RECON_H
