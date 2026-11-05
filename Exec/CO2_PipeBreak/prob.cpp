#include "AMReX_PROB_AMR_F.H"
#include "AMReX_ParmParse.H"
#include "PSAMR.H"
#include "prob.H"
#include "EOS.H"

extern "C" {
    void amrex_probinit(const int* /*init*/,
                        const int* /*name*/,
                        const int* /*namelen*/,
                        const amrex_real* /*problo*/,
                        const amrex_real* /*probhi*/)
    {
        amrex::ParmParse pp("prob");

        pp.query("p0",       PSAMR::h_prob_parm->p0);
        pp.query("T0",       PSAMR::h_prob_parm->T0);
        pp.query("u0",       PSAMR::h_prob_parm->u0);
        pp.query("alpha_trace", PSAMR::h_prob_parm->alpha_trace);   // S3/4.4
        pp.query("p_amb",    PSAMR::h_prob_parm->p_amb);
        pp.query("p_res",    PSAMR::h_prob_parm->p_res);
        pp.query("T_res",    PSAMR::h_prob_parm->T_res);
        pp.query("gap_yc",    PSAMR::h_prob_parm->gap_yc);
        pp.query("gap_half",  PSAMR::h_prob_parm->gap_half);
        pp.query("gap_taper", PSAMR::h_prob_parm->gap_taper);
        pp.query("gap_bell",  PSAMR::h_prob_parm->gap_bell);
        pp.query("ramp_time", PSAMR::h_prob_parm->ramp_time);
        pp.query("res_alpha1", PSAMR::h_prob_parm->res_alpha1);
        pp.query("res_dome_taper", PSAMR::h_prob_parm->res_dome_taper);
        pp.query("res_u",      PSAMR::h_prob_parm->res_u);
        pp.query("res_char_inflow", PSAMR::h_prob_parm->res_char_inflow);
        pp.query("res_mach_max",    PSAMR::h_prob_parm->res_mach_max);
        pp.query("sym_ylo",    PSAMR::h_prob_parm->sym_ylo);
        pp.query("amb_vapor",  PSAMR::h_prob_parm->amb_vapor);

        // Two-phase saturated reservoir (rung 4): precompute the saturated
        // liquid/vapor per-phase (rho,e) at T_res once, so the per-cell
        // bcnormal fill is cheap.
        if (PSAMR::h_prob_parm->res_alpha1 < amrex::Real(1.0)) {
            amrex::Real rhoL, eL, rhoV, eV;
            EOS::co2_sat_LV(PSAMR::h_prob_parm->T_res, rhoL, eL, rhoV, eV);
            PSAMR::h_prob_parm->res_rhoL = rhoL;
            PSAMR::h_prob_parm->res_eL   = eL;
            PSAMR::h_prob_parm->res_rhoV = rhoV;
            PSAMR::h_prob_parm->res_eV   = eV;
            PSAMR::h_prob_parm->res_Psat = EOS::Psat(PSAMR::h_prob_parm->T_res);
            amrex::Print() << "  CO2_PipeBreak: saturated two-phase reservoir @ T_res="
                           << PSAMR::h_prob_parm->T_res << " K, alpha1_liq="
                           << PSAMR::h_prob_parm->res_alpha1
                           << "  rhoL=" << rhoL << " rhoV=" << rhoV
                           << " eL=" << eL << " eV=" << eV << std::endl;

#ifdef USE_PS_HYDRO
            // Mixture (Wood) sound speed of the reservoir ghost -- the SAME
            // harmonic formula the PS solver uses for c_frozen
            // (hem_pelanti_shyue.H, PS_C_MODE=wood), so the characteristic
            // inlet is consistent with the scheme's own wave speeds instead of
            // the equilibrium single-fluid REY2Gam value (which reads ~20%
            // low here: 154 vs 185 m/s for the satjet reservoir).  Used as the
            // choked-inlet cap for res_char_inflow.
            {
                amrex::Real Y[NUM_SPECIES];
                Y[0] = amrex::Real(1.0);
                for (int n = 1; n < NUM_SPECIES; ++n) Y[n] = amrex::Real(0.0);
                const amrex::Real a1 = PSAMR::h_prob_parm->res_alpha1;
                const amrex::Real a2 = amrex::Real(1.0) - a1;
                const amrex::Real rho_mix = a1*rhoL + a2*rhoV;
                amrex::Real c1 = 0.0, c2 = 0.0;
                EOS::REY2Cs_liquid(rhoL, eL, Y, c1);
                EOS::REY2Cs_vapor (rhoV, eV, Y, c2);
                if (c1 > amrex::Real(0.0) && c2 > amrex::Real(0.0)
                    && rho_mix > amrex::Real(0.0)) {
                    const amrex::Real inv_rc2 = a1/(rhoL*c1*c1) + a2/(rhoV*c2*c2);
                    if (inv_rc2 > amrex::Real(0.0))
                        PSAMR::h_prob_parm->res_cfrozen =
                            std::sqrt(amrex::Real(1.0)/(rho_mix*inv_rc2));
                }
                amrex::Print() << "  CO2_PipeBreak: reservoir Psat="
                               << PSAMR::h_prob_parm->res_Psat/1.0e5 << " bar"
                               << "  c_liq=" << c1 << " c_vap=" << c2
                               << "  c_frozen(mix)=" << PSAMR::h_prob_parm->res_cfrozen
                               << " m/s";
                if (PSAMR::h_prob_parm->res_char_inflow != 0
                    && PSAMR::h_prob_parm->res_mach_max > amrex::Real(0.0))
                    amrex::Print() << "  -> characteristic inlet capped at u="
                                   << PSAMR::h_prob_parm->res_mach_max
                                      * PSAMR::h_prob_parm->res_cfrozen << " m/s (M="
                                   << PSAMR::h_prob_parm->res_mach_max << ")";
                amrex::Print() << std::endl;
            }
#endif
        }

        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         PSAMR::h_prob_parm, PSAMR::h_prob_parm+1,
                         PSAMR::d_prob_parm);
    }
}
