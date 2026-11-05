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

        pp.query("p0",    PSAMR::h_prob_parm->p0);
        pp.query("T0",    PSAMR::h_prob_parm->T0);
        pp.query("u0",    PSAMR::h_prob_parm->u0);
        pp.query("p_amb", PSAMR::h_prob_parm->p_amb);

        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         PSAMR::h_prob_parm, PSAMR::h_prob_parm+1,
                         PSAMR::d_prob_parm);
    }
}
