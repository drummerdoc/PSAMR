#ifndef PS_RECON_H
#define PS_RECON_H
// =====================================================================
//  PS_recon.H  —  reconstructed edge states for the wave-propagation
//                 interior (PSAMR.ps_wp_order=4).
//
//  Per cell and direction, the two edge states of the cell are built
//  from the five cell averages i-2..i+2 by WENO5-Z (ps_wp_recon=weno5z)
//  or PPM (ps_wp_recon=ppm), component by component on the primitive set
//  (alpha_1, rho_1, rho_2, e_1, e_2, u[DIM]), in which a uniform-velocity,
//  uniform-pressure contact is uniform in every variable but alpha, and
//  converted back to the conserved array with rho = m_1 + m_2 and
//  rhoE = E_1 + E_2 by construction.  Every edge formula is written in
//  deviation form about the cell's own average (uniform data reconstruct
//  bitwise uniform) and the left edge is the right-edge formula on the
//  reversed stencil (a mirrored stencil gives the mirrored value).
//
//  Presence rules: a cell whose stencil has a phase Absent in some cells
//  and present in others, or whose edge leaves the admissible set (alpha
//  outside [0,1] or at a present phase's vanish threshold, a non-positive
//  or non-finite phase density, a non-finite energy or velocity), takes
//  the round trip of its raw averages at both edges (identical up to
//  round-off, an absent phase's slots exactly zero), first order there;
//  a phase Absent in the cell is pinned absent at its edges.  The caller
//  counts the fallbacks.  The contact taper (ps_wp_contact_taper, at its
//  accessor in PS_umeth.cpp) scales the high-order increment of each
//  edge by the H-5 keep-weight of its face.
//
//  Device-safe: no statics, no ParmParse, no EOS calls (the EOS enters at
//  the face as before).  docs/PLAN_high_order.pdf section 4, Stage 3.
//
//  Interface sharpening (ps_wp_sharpen=1, at its accessor in
//  PS_umeth.cpp; follow-up 11): at a cell with both phases present whose
//  three alpha_1 averages are strictly monotone, a second candidate pair
//  of alpha_1 edges from the THINC profile (Xiao, Honma and Kono 2005)
//  replaces the reconstructed pair when its total boundary variation
//  against the neighbours' reconstructed edges is the smaller (the BVD
//  rule of Sun, Inaba and Xiao 2016); the other primitives are untouched.
//
//  The average-to-point conversions (ps_wp_mc >= 1; follow-up 12): the
//  primitive set the reconstruction works on is formed from the cell's
//  point value U_c = <U> - (1/24) sum_d d2_d <U> and converted back to a
//  cell average <W> = W(U_c) + (1/24) sum_d d2_d W(U_c) (McCorquodale and
//  Colella 2011), so a nonlinear map of the averages no longer leaves a
//  second-order term in the edge values.  point_prims and avg_prims are
//  the two passes; a cell beside a presence front, or whose point value
//  is inadmissible, keeps W(<U>) (counted recon_mc_raw).  At ps_wp_mc=2
//  the face solver takes the face point values of the two edge states
//  (the transverse second difference of the stored edges, PS_umeth.cpp
//  pass 1) and its fluctuations are converted back to face averages.
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
#include "PS_state.H"

namespace PS_RECON {

using amrex::Real;

constexpr int NPRIM = 5 + AMREX_SPACEDIM;   // alpha_1, rho_1, rho_2, e_1, e_2, u[DIM]
constexpr int IA1 = 0, IR1 = 1, IR2 = 2, IE1 = 3, IE2 = 4, IU = 5;

//  THINC steepness: the tanh profile crosses from 5 to 95 % of its jump in
//  1.8 cells; a constant of the method (the literature value), not a
//  threshold on the state.
constexpr Real THINC_BETA = 1.6;

// ---------------------------------------------------------------------
//  THINC right-edge value of the centre cell b from the three averages
//  (a, b, c), strictly monotone: the profile q_min + Delta/2 (1 + theta
//  tanh(beta (xi - x0))) whose cell average is b, evaluated at the right
//  edge (Deng, Boivin and Xiao 2018, eq. 2.14).  The left edge is this
//  function on the reversed stencil, thinc_right(c, b, a), so a mirrored
//  stencil gives the mirrored value bitwise.
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real thinc_right (Real a, Real b, Real c) noexcept
{
    const Real qmin = amrex::min(a, c), qmax = amrex::max(a, c);
    const Real d = qmax - qmin;
    const Real th = (c > a) ? Real(1.0) : Real(-1.0);
    const Real C = (b - qmin) / d;
    const Real tb = std::tanh(THINC_BETA), cb = std::cosh(THINC_BETA);
    const Real B = std::exp(th * THINC_BETA * (Real(2.0) * C - Real(1.0)));
    const Real A = (B / cb - Real(1.0)) / tb;
    return qmin + Real(0.5) * d * (Real(1.0) + th * (tb + A) / (Real(1.0) + A * tb));
}

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
//  the ratio tau/beta_k (PSAMR.ps_wp_weno_power: 1 = Borges, 2 the
//  higher power); 1 is bitwise the plain ratio.
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real weno5z_right (Real a, Real b, Real c, Real d, Real e, int wpow) noexcept
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
Real weno5z_left (Real a, Real b, Real c, Real d, Real e, int wpow) noexcept
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
//  The primitive set of one cell from its conserved array: the pre-EOS
//  cell state of PS_state.H (the clamped alpha, the checked phase
//  quotients with the mixture placeholders, the velocity).  ab_k says
//  the phase is Absent.
// ---------------------------------------------------------------------
struct PsPrim {
    Real v[NPRIM];
    bool ab1, ab2;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PsPrim prim_from_cons (const Real U[NVAR], const PsPres& pr) noexcept
{
    PsCellState s;
    ps_cell_prims(U, pr, s);
    PsPrim p;
    p.ab1 = (s.rg1 == PsRegime::Absent);
    p.ab2 = (s.rg2 == PsRegime::Absent);
    p.v[IA1] = s.alpha1;
    p.v[IR1] = s.rho1;
    p.v[IR2] = s.rho2;
    p.v[IE1] = s.e1;
    p.v[IE2] = s.e2;
    p.v[IU + 0] = s.ux;
#if (AMREX_SPACEDIM >= 2)
    p.v[IU + 1] = s.uy;
#endif
#if (AMREX_SPACEDIM == 3)
    p.v[IU + 2] = s.uz;
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
    if (t > Real(1.0)) t = Real(1.0);
    return Real(1.0) - t * t * (Real(3.0) - Real(2.0) * t);
}

//  Layout of the conversion stores: comps 0..NPRIM-1 the primitive set,
//  NPRIM and NPRIM+1 the Absent flags of the two phases, NPRIM+2 1 where
//  the point value was used (and, in the average store, where the
//  converted average stands).
constexpr int WC_AB1 = NPRIM, WC_AB2 = NPRIM + 1, WC_PT = NPRIM + 2;
//  The edge store: comps 0..NVAR-1 the low edge, NVAR..2NVAR-1 the high
//  edge, RQ_FB 1 where the cell took its raw averages, then the two
//  edges' primitive sets (RQ_PL, RQ_PH; the face point conversion of
//  ps_wp_mc=2 acts on these, before the map to the conserved array).
constexpr int RQ_FB = 2 * NVAR;
constexpr int RQ_PL = 2 * NVAR + 1, RQ_PH = RQ_PL + NPRIM, RQ_NCOMP = RQ_PH + NPRIM;

// ---------------------------------------------------------------------
//  point_prims: the primitive set of the cell's point value, W(U_c), at
//  (i,j,k), into wc; the Absent flags from the cell average itself.  A
//  cell with a presence change among its face neighbours, or whose point
//  value leaves the admissible set, stores W(<U>) with WC_PT = 0.
//  Returns true when an admissible-looking cell's point value was refused
//  (the caller counts).
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool point_prims (int i, int j, int k,
                  amrex::Array4<const Real> const& uin,
                  amrex::Array4<Real> const& wc, const PsPres& pr) noexcept
{
    Real U[NVAR], Uc[NVAR];
    for (int n = 0; n < NVAR; ++n) { U[n] = uin(i, j, k, n); Uc[n] = U[n]; }
    const PsPrim Pa = prim_from_cons(U, pr);
    bool front = false;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const int oi = (d == 0), oj = (d == 1), ok = (d == 2);
        Real Um[NVAR], Up[NVAR];
        for (int n = 0; n < NVAR; ++n) {
            Um[n] = uin(i - oi, j - oj, k - ok, n);
            Up[n] = uin(i + oi, j + oj, k + ok, n);
            Uc[n] -= (Up[n] - Real(2.0) * U[n] + Um[n]) / Real(24.0);
        }
        const PsPrim Pm = prim_from_cons(Um, pr), Pp = prim_from_cons(Up, pr);
        if (Pm.ab1 != Pa.ab1 || Pm.ab2 != Pa.ab2 || Pp.ab1 != Pa.ab1 || Pp.ab2 != Pa.ab2) front = true;
    }
    bool raw = front;
    PsPrim Pc;
    if (!raw) {
        Pc = prim_from_cons(Uc, pr);
        const Real* v = Pc.v;
        if (Pc.ab1 != Pa.ab1 || Pc.ab2 != Pa.ab2) raw = true;
        else if (!(v[IA1] >= Real(0.0) && v[IA1] <= Real(1.0))) raw = true;
        else if (!Pa.ab1 && (!(v[IR1] > Real(0.0)) || !std::isfinite(v[IR1]) || !std::isfinite(v[IE1]))) raw = true;
        else if (!Pa.ab2 && (!(v[IR2] > Real(0.0)) || !std::isfinite(v[IR2]) || !std::isfinite(v[IE2]))) raw = true;
        else { for (int m = 0; m < AMREX_SPACEDIM; ++m) if (!std::isfinite(v[IU + m])) raw = true; }
    }
    const PsPrim& W = raw ? Pa : Pc;
    for (int m = 0; m < NPRIM; ++m) wc(i, j, k, m) = W.v[m];
    wc(i, j, k, WC_AB1) = Pa.ab1 ? Real(1.0) : Real(0.0);
    wc(i, j, k, WC_AB2) = Pa.ab2 ? Real(1.0) : Real(0.0);
    wc(i, j, k, WC_PT)  = raw ? Real(0.0) : Real(1.0);
    return raw && !front;
}

// ---------------------------------------------------------------------
//  avg_prims: the cell average of the primitive set, <W> = W_c + (1/24)
//  sum_d d2_d W_c, from the point store wc into wq at (i,j,k); a cell
//  whose point value was not used carries W(<U>) through unchanged.
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void avg_prims (int i, int j, int k,
                amrex::Array4<const Real> const& wc,
                amrex::Array4<Real> const& wq) noexcept
{
    const bool pt = wc(i, j, k, WC_PT) != Real(0.0);
    for (int m = 0; m < NPRIM; ++m) {
        Real w = wc(i, j, k, m);
        if (pt) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                const int oi = (d == 0), oj = (d == 1), ok = (d == 2);
                w += (wc(i + oi, j + oj, k + ok, m) - Real(2.0) * wc(i, j, k, m)
                      + wc(i - oi, j - oj, k - ok, m)) / Real(24.0);
            }
        }
        wq(i, j, k, m) = w;
    }
    wq(i, j, k, WC_AB1) = wc(i, j, k, WC_AB1);
    wq(i, j, k, WC_AB2) = wc(i, j, k, WC_AB2);
    wq(i, j, k, WC_PT)  = pt ? Real(1.0) : Real(0.0);
}

// ---------------------------------------------------------------------
//  cell_edges: the two edge states (conserved arrays) of cell (i,j,k)
//  in direction idir, written to rq(i,j,k, 0..NVAR-1) (low edge) and
//  rq(i,j,k, NVAR..2NVAR-1) (high edge).  recon: 0 = WENO5-Z, 1 = PPM.
//  taper: the contact-taper option; wpow: the WENO power; sharpen: the
//  THINC-BVD option on alpha_1; mc: the stencil's primitive set is read
//  from the converted store wq (avg_prims) instead of being formed from
//  the averages.  Returns 1 when the cell fell back to its raw averages,
//  2 when the THINC candidate was chosen, 0 otherwise (the caller counts).
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int cell_edges (int idir, int i, int j, int k,
                amrex::Array4<const Real> const& uin,
                amrex::Array4<Real> const& rq,
                int recon, int taper, int wpow, int sharpen,
                int mc, amrex::Array4<const Real> const& wq,
                const PsPres& pr) noexcept
{
    const int oi = (idir == 0), oj = (idir == 1), ok = (idir == 2);
    PsPrim P[5];
    Real Uc[NVAR];
    for (int n = 0; n < NVAR; ++n) Uc[n] = uin(i, j, k, n);
    for (int s = -2; s <= 2; ++s) {
        const int ii = i + s*oi, jj = j + s*oj, kk = k + s*ok;
        if (mc != 0) {
            for (int m = 0; m < NPRIM; ++m) P[s + 2].v[m] = wq(ii, jj, kk, m);
            P[s + 2].ab1 = wq(ii, jj, kk, WC_AB1) != Real(0.0);
            P[s + 2].ab2 = wq(ii, jj, kk, WC_AB2) != Real(0.0);
        } else {
            Real U[NVAR];
            for (int n = 0; n < NVAR; ++n) U[n] = uin(ii, jj, kk, n);
            P[s + 2] = prim_from_cons(U, pr);
        }
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
    bool thinc = false;
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
        // The THINC candidate for alpha_1 where both phases are present
        // and the three averages are strictly monotone; chosen by the
        // smaller total boundary variation against the neighbours'
        // reconstructed edges (their stencils reach three cells out).
        if (sharpen != 0 && !C.ab1 && !C.ab2) {
            const Real a1 = P[1].v[IA1], b1 = C.v[IA1], c1 = P[3].v[IA1];
            if ((b1 - a1) * (c1 - b1) > Real(0.0)) {
                const Real tl = thinc_right(c1, b1, a1);
                const Real tr = thinc_right(a1, b1, c1);
                // The two cells three out: alpha_1 is its own slot of the
                // averages (the converted set differs from it at fourth
                // order, and this is a selection criterion, not a state).
                const Real am = uin(i - 3*oi, j - 3*oj, k - 3*ok, UALPHA1);
                const Real ap = uin(i + 3*oi, j + 3*oj, k + 3*ok, UALPHA1);
                const Real a0 = P[0].v[IA1], a4 = P[4].v[IA1];
                Real nl, nr;   // the left neighbour's right edge, the right neighbour's left edge
                if (recon == 1) {
                    Real dummy;
                    ppm_edges(am, a0, a1, b1, c1, dummy, nl);
                    ppm_edges(a1, b1, c1, a4, ap, nr, dummy);
                } else {
                    nl = weno5z_right(am, a0, a1, b1, c1, wpow);
                    nr = weno5z_left (a1, b1, c1, a4, ap, wpow);
                }
                const Real tbv_w = std::abs(vl[IA1] - nl) + std::abs(nr - vr[IA1]);
                const Real tbv_t = std::abs(tl - nl) + std::abs(nr - tr);
                if (tbv_t < tbv_w) { vl[IA1] = tl; vr[IA1] = tr; thinc = true; }
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
        thinc = false;
    }
    Real UL[NVAR], UR[NVAR];
    cons_from_prim(vl, C.ab1, C.ab2, Y, UL);
    cons_from_prim(vr, C.ab1, C.ab2, Y, UR);
    for (int n = 0; n < NVAR; ++n) {
        rq(i, j, k, n)        = UL[n];
        rq(i, j, k, NVAR + n) = UR[n];
    }
    rq(i, j, k, RQ_FB) = fallback ? Real(1.0) : Real(0.0);
    for (int m = 0; m < NPRIM; ++m) { rq(i, j, k, RQ_PL + m) = vl[m]; rq(i, j, k, RQ_PH + m) = vr[m]; }
    return fallback ? 1 : (thinc ? 2 : 0);
}

} // namespace PS_RECON

#endif // USE_PS_HYDRO
#endif // PS_RECON_H
