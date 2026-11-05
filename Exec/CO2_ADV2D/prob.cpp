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

        pp.query("P0", PSAMR::h_prob_parm->P0);
        pp.query("T1", PSAMR::h_prob_parm->T1);
        pp.query("T2", PSAMR::h_prob_parm->T2);

        pp.query("u0", PSAMR::h_prob_parm->u0);
        pp.query("v0", PSAMR::h_prob_parm->v0);

        pp.query("alpha_mean", PSAMR::h_prob_parm->alpha_mean);
        pp.query("alpha_amp",  PSAMR::h_prob_parm->alpha_amp);
        pp.query("k_x",        PSAMR::h_prob_parm->k_x);
        pp.query("k_y",        PSAMR::h_prob_parm->k_y);
        pp.query("alpha_pow",  PSAMR::h_prob_parm->alpha_pow);

        PSAMR::h_prob_parm->p_amb = PSAMR::h_prob_parm->P0;

        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         PSAMR::h_prob_parm, PSAMR::h_prob_parm+1,
                         PSAMR::d_prob_parm);
    }
}
