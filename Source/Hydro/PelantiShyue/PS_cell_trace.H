#ifndef PS_CELL_TRACE_H
#define PS_CELL_TRACE_H
// =====================================================================
//  ps_cell_trace: one cell's PS state printed with a label (follow-up 24;
//  PSAMR.ps_trace_cell = i j k, PSAMR.ps_trace_level = 0).  Called from
//  the stage loop (PSAMR_advance.cpp) and from the coarse-fine sequence
//  (PSAMR.cpp: after the reflux, after the average-down, around the
//  regrid pass).  Host builds only; reads only; inert when the dial is
//  absent.
// =====================================================================
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <iomanip>
#include <vector>
#include "IndexDefines.H"
#include "PS_dials.H"

inline const amrex::IntVect& ps_trace_cell_iv()
{
    static const amrex::IntVect iv = []() -> amrex::IntVect {
        amrex::ParmParse pp("PSAMR");
        std::vector<int> v;
        if (pp.queryarr("ps_trace_cell", v) && static_cast<int>(v.size()) >= AMREX_SPACEDIM) {
            return amrex::IntVect(AMREX_D_DECL(v[0], v[1], v[2]));
        }
        return amrex::IntVect(AMREX_D_DECL(-1000000, -1000000, -1000000));
    }();
    return iv;
}
inline int ps_trace_cell_level() { static const int l = ps_dial_int("ps_trace_level", 0); return l; }
inline bool ps_trace_cell_on(int level) { return ps_trace_cell_iv()[0] > -1000000 && level == ps_trace_cell_level(); }

inline void ps_cell_trace(const amrex::MultiFab& S, int level, int step, const char* label)
{
#if defined(USE_PS_HYDRO) && !defined(AMREX_USE_GPU)
    if (!ps_trace_cell_on(level)) return;
    const amrex::IntVect& iv = ps_trace_cell_iv();
    for (amrex::MFIter mfi(S); mfi.isValid(); ++mfi) {
        if (!mfi.validbox().contains(iv)) continue;
        auto const& sa = S.const_array(mfi);
        const amrex::Real a1 = sa(iv, UALPHA1), m1 = sa(iv, UM1RHO1), m2 = sa(iv, UM2RHO2);
        const amrex::Real rho = sa(iv, URHO);
        const amrex::Real ux = sa(iv, UMX) / rho;
        amrex::Real ke = amrex::Real(0.5) * ux * ux;
#if (AMREX_SPACEDIM >= 2)
        const amrex::Real uy = sa(iv, UMY) / rho; ke += amrex::Real(0.5) * uy * uy;
#endif
        const amrex::Real e1 = (m1 > 0.0) ? sa(iv, UE1) / m1 - ke : 0.0;
        const amrex::Real e2 = (m2 > 0.0) ? sa(iv, UE2) / m2 - ke : 0.0;
        amrex::AllPrint() << std::setprecision(9) << "[CELL-TRACE] step " << step << " " << label
                          << ": a1=" << a1 << " m1=" << m1 << " rho1=" << (a1 > 0.0 ? m1 / a1 : -1.0) << " e1=" << e1
                          << " | m2=" << m2 << " rho2=" << (a1 < 1.0 ? m2 / (1.0 - a1) : -1.0) << " e2=" << e2
                          << " | rho=" << rho << " ux=" << ux << "\n";
    }
#else
    amrex::ignore_unused(S, level, step, label);
#endif
}
#endif
