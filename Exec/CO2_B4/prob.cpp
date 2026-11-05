#include "AMReX_PROB_AMR_F.H"
#include "AMReX_ParmParse.H"
#include "PSAMR.H"
#include "prob.H"

extern "C" {
    void amrex_probinit(const int* /*init*/,
                        const int* /*name*/,
                        const int* /*namelen*/,
                        const amrex_real* /*problo*/,
                        const amrex_real* /*probhi*/)
    {
        amrex::ParmParse pp("prob");

        pp.query("p_L",      PSAMR::h_prob_parm->p_L);
        pp.query("T_L",      PSAMR::h_prob_parm->T_L);
        pp.query("u_L",      PSAMR::h_prob_parm->u_L);

        pp.query("p_R",      PSAMR::h_prob_parm->p_R);
        pp.query("T_R",      PSAMR::h_prob_parm->T_R);
        pp.query("u_R",      PSAMR::h_prob_parm->u_R);

        pp.query("x_diaph",  PSAMR::h_prob_parm->x_diaph);

        // Two-face NSCBC ambient targets default to the same-side
        // initial-state pressure so that a wave reaching either
        // boundary sees a "matched" ambient and leaves cleanly.  Can
        // be overridden explicitly if a Riemann analytic solution
        // predicts a different asymptotic state at either far-field.
        PSAMR::h_prob_parm->p_amb_lo = PSAMR::h_prob_parm->p_L;
        PSAMR::h_prob_parm->p_amb_hi = PSAMR::h_prob_parm->p_R;
        pp.query("p_amb_lo", PSAMR::h_prob_parm->p_amb_lo);
        pp.query("p_amb_hi", PSAMR::h_prob_parm->p_amb_hi);

        // NSCBC-path fallback (single ambient).  Default to the
        // higher-side value which matches T-Blowdown-style setups
        // where only the high face is an outflow.  For B4 the
        // sgn-aware bcnormal path is preferred — see prob.H.
        PSAMR::h_prob_parm->p_amb = PSAMR::h_prob_parm->p_amb_hi;
        pp.query("p_amb", PSAMR::h_prob_parm->p_amb);

        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         PSAMR::h_prob_parm, PSAMR::h_prob_parm+1,
                         PSAMR::d_prob_parm);
    }
}
