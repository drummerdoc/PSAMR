#ifndef PS_FLOOR_TABLE_H
#define PS_FLOOR_TABLE_H
// =====================================================================
//  PS_floor_table.H — the floor's pre-test table (docs/PLAN_hygiene.md,
//  item 2; 27 November 2026).
//
//  ps_apply_floor raises e_k to max(e_k, e(T_floor, rho_k), e(P_floor,
//  rho_k)) and paid the two evaluations in every cell every step where
//  the floor binds in a handful.  At fixed T the energy RTY2PE_auto
//  returns decreases with density over the whole domain: on each PR
//  branch (de/dv)_T = (a - T a') / (v^2 + 2 b v - b^2) > 0 since a > 0 and
//  a' < 0, and inside the dome it is the lever rule between the saturated
//  energies, linear in v with e_V > e_L and continuous at the saturated
//  densities; and its pressure does not decrease with density, (dP/drho)_T
//  > 0 on the sides of the branches the auto rule evaluates and P = Psat
//  inside the dome.  So on a grid rho_i, uniform in ln rho, the pair at
//  the grid point at or below rho_k bounds the pair at rho_k:
//  e(T_f, rho_k) <= e(T_f, rho_i) and P(T_f, rho_k) >= P(T_f, rho_i).  A
//  phase with e_k > e(T_f, rho_i) has T_k > T_f (e increases with T at
//  fixed rho), so the temperature floor cannot bind; and if P(T_f, rho_i)
//  >= P_floor the floor's own rule skips the pressure floor (P_tf >=
//  pfloor), so nothing would be written: such a cell is skipped without
//  the evaluations, its result bitwise the evaluated one.  Every other
//  cell evaluates as before.  The table is filled once by the floor's own
//  RTY2PE_auto (one source of truth); its monotonicity is checked at the
//  fill and a violation aborts (the derivation would not hold for that
//  fluid); the grid's size is a resolution, not a threshold: a coarser
//  grid skips fewer cells, never a wrong one.  No constant.  Device
//  resident; the view is captured by value into the floor's kernel.
// =====================================================================
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_Print.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <vector>
#include "IndexDefines.H"
#include "EOS.H"
#include "PS_util.H"      // ps_pure_species

namespace ps_floor_table {

struct View {
    const amrex::Real* e_tf = nullptr;   // e(T_floor, rho_i)
    const amrex::Real* p_tf = nullptr;   // P(T_floor, rho_i)
    amrex::Real lnr_lo = amrex::Real(0.0), dlnr = amrex::Real(1.0);
    int n = 0;                           // 0 = no table: evaluate
    //  True where neither floor can bind at (rho, e): the cell is skipped.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool cannot_bind(amrex::Real rho, amrex::Real e, amrex::Real pfloor) const noexcept
    {
        if (n < 2 || !(rho > amrex::Real(0.0))) return false;
        const amrex::Real x = (std::log(rho) - lnr_lo) / dlnr;
        if (!(x >= amrex::Real(0.0)) || !(x < amrex::Real(n - 1))) return false;   // outside the grid: evaluate
        const int i = static_cast<int>(x);                                          // the grid point at or below rho
        return (e > e_tf[i]) && (p_tf[i] >= pfloor);
    }
};

namespace detail {
struct Store {
    amrex::Gpu::DeviceVector<amrex::Real> e, p;
    amrex::Real tfloor = amrex::Real(-1.0);
    View view;
};
inline Store*& store() { static Store* s = nullptr; return s; }
}

constexpr int N = 4096;

//  Host only: the view for this floor temperature, filled on first use.
inline View view(amrex::Real tfloor)
{
    using amrex::Real;
    detail::Store*& s = detail::store();
    if (s != nullptr && s->tfloor == tfloor) return s->view;
    if (s == nullptr) {
        s = new detail::Store();
        amrex::ExecOnFinalize([] () { delete detail::store(); detail::store() = nullptr; });
    }
    const Real r_lo = EOS::rho_min(), r_hi = EOS::rho_max();
    const Real lnr_lo = std::log(r_lo), dlnr = (std::log(r_hi) - lnr_lo) / Real(N - 1);
    std::vector<Real> he(N), hp(N);
    Real Y[NUM_SPECIES]; ps_pure_species(Y);
    for (int i = 0; i < N; ++i) {
        const Real rho = std::exp(lnr_lo + Real(i) * dlnr);
        Real P, E;
        EOS::RTY2PE_auto(rho, tfloor, Y, P, E);
        if (!std::isfinite(E) || !std::isfinite(P)) {
            amrex::Abort("[PS-FLOOR] pre-test table: non-finite e or P at the floor temperature");
        }
        he[i] = E; hp[i] = P;
        if (i > 0 && (he[i] > he[i - 1] || hp[i] < hp[i - 1])) {
            amrex::Print() << "[PS-FLOOR] pre-test table: at rho = " << rho << " e " << he[i-1] << " -> " << he[i]
                           << ", P " << hp[i-1] << " -> " << hp[i] << "\n";
            amrex::Abort("[PS-FLOOR] pre-test table: e(T_floor, rho) rises or P(T_floor, rho) falls with density; "
                         "the derivation does not hold for this fluid, run with PSAMR.ps_floor_pretest = 0");
        }
    }
    s->e.resize(N); s->p.resize(N);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, he.begin(), he.end(), s->e.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, hp.begin(), hp.end(), s->p.begin());
    amrex::Gpu::streamSynchronize();
    s->tfloor = tfloor;
    s->view.e_tf = s->e.data(); s->view.p_tf = s->p.data();
    s->view.lnr_lo = lnr_lo; s->view.dlnr = dlnr; s->view.n = N;
    amrex::Print() << "[PS-FLOOR] pre-test table: " << N << " points in ln rho over [" << r_lo << ", " << r_hi
                   << "] kg/m3 at T_floor = " << tfloor << " K; e from " << he[0] << " to " << he[N - 1]
                   << " J/kg, P from " << hp[0] << " to " << hp[N - 1] << " Pa\n";
    return s->view;
}

}  // namespace ps_floor_table
#endif  // PS_FLOOR_TABLE_H
