
// This file is automatically created by parse_PSAMR_params.py.  To update
// or add runtime parameters, please edit _cpp_parameters and then run
// mk_params.sh

int PSAMR::state_nghost = 0;
int PSAMR::do_reflux = 1;
int PSAMR::do_avg_down = 1;
int PSAMR::refine_cut_cells = 0;
amrex::Real PSAMR::difmag = 0.1;
amrex::Real PSAMR::small = 1.e-8;
amrex::Real PSAMR::small_dens = 1.e-7;
amrex::Real PSAMR::small_temp = 1.e2;
amrex::Real PSAMR::small_pres = 1.e-200;
int PSAMR::do_mol = 0;
int PSAMR::do_react = 0;
int PSAMR::add_ext_src = 0;
int PSAMR::add_grav_src = 1;
amrex::Real PSAMR::const_grav = 0.0;
int PSAMR::ppm_type = 0;
int PSAMR::ppm_trace_sources = 0;
int PSAMR::ppm_temp_fix = 0;
int PSAMR::ppm_predict_gammae = 0;
int PSAMR::plm_iorder = 2;
int PSAMR::use_flattening = 1;
int PSAMR::dual_energy_update_E_from_e = 1;
amrex::Real PSAMR::dual_energy_eta1 = 1.0e0;
amrex::Real PSAMR::dual_energy_eta2 = 1.0e-4;
int PSAMR::use_pslope = 0;
int PSAMR::fix_mass_flux = 0;
int PSAMR::limit_fluxes_on_small_dens = 0;
int PSAMR::allow_negative_energy = 1;
int PSAMR::allow_small_energy = 1;
int PSAMR::transverse_reset_density = 1;
amrex::Real PSAMR::eb_boundary_T = 1.0;
int PSAMR::eb_isothermal = 1;
int PSAMR::eb_noslip = 1;
int PSAMR::eb_weights_type = 2;
std::string PSAMR::redistribution_type = "FluxRedist";
amrex::Real PSAMR::fixed_dt = -1.0;
amrex::Real PSAMR::initial_dt = -1.0;
amrex::Real PSAMR::dt_cutoff = 0.0;
amrex::Real PSAMR::max_dt = 1.e200;
amrex::Real PSAMR::cfl = 0.8;
amrex::Real PSAMR::init_shrink = 1.0;
amrex::Real PSAMR::change_max = 1.1;
amrex::Real PSAMR::retry_neg_dens_factor = 1.e-1;
int PSAMR::bndry_func_thread_safe = 1;
int PSAMR::sum_interval = -1;
amrex::Real PSAMR::sum_per = -1.0e0;
int PSAMR::hard_cfl_limit = 1;
std::string PSAMR::job_name = "";
int PSAMR::write_eb = 0;
