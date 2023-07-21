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
        // Parse params
        amrex::ParmParse pp("prob");

        pp.query("type",  PSAMR::h_prob_parm->type);
        pp.query("p_l",   PSAMR::h_prob_parm->p_l);
        pp.query("p_r",   PSAMR::h_prob_parm->p_r);
        pp.query("rho_l", PSAMR::h_prob_parm->rho_l);
        pp.query("rho_r", PSAMR::h_prob_parm->rho_r);
        pp.query("u_l",   PSAMR::h_prob_parm->u_l);
        pp.query("u_r",   PSAMR::h_prob_parm->u_r);

        amrex::Gpu::copy(amrex::Gpu::hostToDevice, PSAMR::h_prob_parm, PSAMR::h_prob_parm+1, PSAMR::d_prob_parm);
    }
}
