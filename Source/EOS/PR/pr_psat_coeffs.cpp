// =====================================================================
//  pr_psat_coeffs.cpp -- the one definition of the saturation-pressure
//  spline arrays on a GPU build (device constant memory); every other
//  translation unit sees extern declarations (pr_psat_tab.H).  On a host
//  build the arrays are header constexpr and this file is inert.
// =====================================================================
#include <AMReX_Config.H>
#define PR_PSAT_DEFINE
#include "pr_psat_tab.H"
