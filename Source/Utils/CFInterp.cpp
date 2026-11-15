#include "CFInterp.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_Loop.H>

using namespace amrex;

CFInterpQuarticLimited psamr_quartic_limited_interp;

namespace {

constexpr int NF = 1 << AMREX_SPACEDIM;   // fine cells per coarse cell

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real dev1 (Real qm2, Real qm1, Real q0, Real qp1, Real qp2) noexcept
{
    constexpr Real cL2 = Real(-0.01171875);
    constexpr Real cL1 = Real( 0.0859375);
    return Real(2.0) * (cL2 * (qm2 - q0) + cL1 * (qm1 - q0)
                      - cL1 * (qp1 - q0) - cL2 * (qp2 - q0));
}

// The limited quartic values of the 2^DIM fine cells of one coarse cell,
// written once as a host-device function so that the GPU launch and the
// CPU loop call the same code (a device-only lambda cannot be called from
// the host loop under nvcc).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void quartic_limited_cell (int ic, int jc, int kc, int n,
                           Array4<Real const> const& crsearr,
                           Array4<Real>       const& finearr,
                           Box                const& target) noexcept
{
    // The quartic values of the 2^DIM fine cells of this coarse cell,
    // AMReX's ccquartic stencil written in deviation form: along each
    // direction the pair of fine values is q_c -/+ dev with
    // dev = 2 * sum_{k != 0} cL(k) (q(k) - q_c)  (sum of cL is 1/2, so
    // this is the same polynomial), evaluated so that a field uniform
    // along a direction gives bitwise-identical fine values along it
    // and the pair average is q_c exactly.  AMReX's own evaluation
    // leaves a round-off asymmetry on uniform data, which the 2-D
    // scheme amplified on the T-Blowdown tube (Stage 2b record).
    // Stage 1 (z then y then x in 3-D; y then x in 2-D; x in 1-D):
    // at each stage the pair (low, high) = (c + d, c - d) with d from the
    // five values along that direction, applied to the values the
    // previous stage produced.
    Real v[NF];
    {
#if (AMREX_SPACEDIM == 3)
        // y-direction values of the 5x5 (i,j) column stencil after z.
        Real cz[5][5][2];   // [ii][jj][kz]
        for (int ii = -2; ii <= 2; ++ii) {
            for (int jj = -2; jj <= 2; ++jj) {
                const Real q0 = crsearr(ic+ii, jc+jj, kc, n);
                const Real d  = dev1(crsearr(ic+ii, jc+jj, kc-2, n), crsearr(ic+ii, jc+jj, kc-1, n), q0,
                                     crsearr(ic+ii, jc+jj, kc+1, n), crsearr(ic+ii, jc+jj, kc+2, n));
                cz[ii+2][jj+2][0] = q0 + d;
                cz[ii+2][jj+2][1] = q0 - d;
            }
        }
        Real cy[5][2][2];   // [ii][ky][kz]
        for (int ii = -2; ii <= 2; ++ii) {
            for (int kz = 0; kz < 2; ++kz) {
                const Real q0 = cz[ii+2][2][kz];
                const Real d  = dev1(cz[ii+2][0][kz], cz[ii+2][1][kz], q0, cz[ii+2][3][kz], cz[ii+2][4][kz]);
                cy[ii+2][0][kz] = q0 + d;
                cy[ii+2][1][kz] = q0 - d;
            }
        }
        int m = 0;
        for (int kz = 0; kz < 2; ++kz) {
            for (int ky = 0; ky < 2; ++ky) {
                const Real q0 = cy[2][ky][kz];
                const Real d  = dev1(cy[0][ky][kz], cy[1][ky][kz], q0, cy[3][ky][kz], cy[4][ky][kz]);
                // m ordering must match the write loop: kk outer, jj, ii inner
                v[(kz*2 + ky)*2 + 0] = q0 + d;
                v[(kz*2 + ky)*2 + 1] = q0 - d;
                ++m;
            }
        }
        amrex::ignore_unused(m);
#elif (AMREX_SPACEDIM == 2)
        Real cy[5][2];      // [ii][ky]
        for (int ii = -2; ii <= 2; ++ii) {
            const Real q0 = crsearr(ic+ii, jc, kc, n);
            const Real d  = dev1(crsearr(ic+ii, jc-2, kc, n), crsearr(ic+ii, jc-1, kc, n), q0,
                                 crsearr(ic+ii, jc+1, kc, n), crsearr(ic+ii, jc+2, kc, n));
            cy[ii+2][0] = q0 + d;
            cy[ii+2][1] = q0 - d;
        }
        for (int ky = 0; ky < 2; ++ky) {
            const Real q0 = cy[2][ky];
            const Real d  = dev1(cy[0][ky], cy[1][ky], q0, cy[3][ky], cy[4][ky]);
            v[ky*2 + 0] = q0 + d;
            v[ky*2 + 1] = q0 - d;
        }
#else
        const Real q0 = crsearr(ic, jc, kc, n);
        const Real d  = dev1(crsearr(ic-2, jc, kc, n), crsearr(ic-1, jc, kc, n), q0,
                             crsearr(ic+1, jc, kc, n), crsearr(ic+2, jc, kc, n));
        v[0] = q0 + d;
        v[1] = q0 - d;
#endif
    }
    const Real vmean = crsearr(ic, jc, kc, n);   // the pair averages are exact

    // The range of the coarse cell and its 3^DIM - 1 neighbours.
    Real mn = crsearr(ic, jc, kc, n);
    Real mx = mn;
#if (AMREX_SPACEDIM == 3)
    for (int dk = -1; dk <= 1; ++dk) {
#else
    { const int dk = 0;
#endif
#if (AMREX_SPACEDIM >= 2)
    for (int dj = -1; dj <= 1; ++dj) {
#else
    { const int dj = 0;
#endif
    for (int di = -1; di <= 1; ++di) {
        const Real q = crsearr(ic + di, jc + dj, kc + dk, n);
        mn = amrex::min(mn, q);
        mx = amrex::max(mx, q);
    }}}

    // One factor for all deviations: the largest in [0,1] that keeps
    // every fine value inside [mn, mx].
    Real theta = Real(1.0);
    for (int q = 0; q < NF; ++q) {
        const Real d = v[q] - vmean;
        if (d > Real(0.0) && v[q] > mx) {
            theta = amrex::min(theta, amrex::max(Real(0.0), (mx - vmean) / d));
        } else if (d < Real(0.0) && v[q] < mn) {
            theta = amrex::min(theta, amrex::max(Real(0.0), (mn - vmean) / d));
        }
    }
    {
        int m = 0;
#if (AMREX_SPACEDIM == 3)
        for (int kk = 0; kk < 2; ++kk) {
#else
        { const int kk = 0;
#endif
#if (AMREX_SPACEDIM >= 2)
        for (int jj = 0; jj < 2; ++jj) {
#else
        { const int jj = 0;
#endif
        for (int ii = 0; ii < 2; ++ii) {
            const int i = 2*ic + ii;
            const int j = AMREX_SPACEDIM >= 2 ? 2*jc + jj : jc;
            const int k = AMREX_SPACEDIM == 3 ? 2*kc + kk : kc;
            if (target.contains(IntVect(AMREX_D_DECL(i, j, k)))) {
                finearr(i, j, k, n) = vmean + theta * (v[m] - vmean);
            }
            ++m;
        }}}
    }
}

} // namespace

Box
CFInterpQuarticLimited::CoarseBox (const Box& fine, int ratio)
{
    Box crse(amrex::coarsen(fine, ratio));
    crse.grow(2);
    return crse;
}

Box
CFInterpQuarticLimited::CoarseBox (const Box& fine, const IntVect& ratio)
{
    Box crse = amrex::coarsen(fine, ratio);
    crse.grow(2);
    return crse;
}

void
CFInterpQuarticLimited::interp (const FArrayBox&  crse,
                                int               crse_comp,
                                FArrayBox&        fine,
                                int               fine_comp,
                                int               ncomp,
                                const Box&        fine_region,
                                const IntVect&    ratio,
                                const Geometry&   /*crse_geom*/,
                                const Geometry&   /*fine_geom*/,
                                Vector<BCRec> const& /*bcr*/,
                                int               /*actual_comp*/,
                                int               /*actual_state*/,
                                RunOn             runon)
{
    BL_PROFILE("CFInterpQuarticLimited::interp()");
    if (ratio != IntVect(2)) {
        amrex::Abort("CFInterpQuarticLimited: refinement ratio must be 2");
    }
    // The fine region need not be aligned to coarse cells (AMReX's derive
    // for tagging fills one ghost cell): every coarse cell touching it is
    // evaluated in full, so the bound and conservation are those of the
    // whole coarse cell, and only the fine cells inside the region are
    // written.
    const Box target = fine_region & fine.box();
    const Box cbox   = amrex::coarsen(target, 2);

    Array4<Real const> const& crsearr = crse.const_array(crse_comp);
    Array4<Real>       const& finearr = fine.array(fine_comp);

    if (runon == RunOn::Gpu && Gpu::inLaunchRegion()) {
        amrex::ParallelFor(cbox, ncomp,
            [=] AMREX_GPU_DEVICE (int ic, int jc, int kc, int n) noexcept {
                quartic_limited_cell(ic, jc, kc, n, crsearr, finearr, target);
            });
    } else {
        amrex::LoopOnCpu(cbox, ncomp,
            [=] (int ic, int jc, int kc, int n) noexcept {
                quartic_limited_cell(ic, jc, kc, n, crsearr, finearr, target);
            });
    }
}
