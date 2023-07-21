#ifndef PSAMR_CONSTANTS_H
#define PSAMR_CONSTANTS_H

#include <AMReX_REAL.H>

struct PSAMRConstants
{
    static constexpr amrex::Real small     = 1.e-12;
    static constexpr amrex::Real smallu    = 1.e-12;
    static constexpr amrex::Real small_num = 1.e-8;

    static constexpr int level_mask_interior   = 0; // valid cells
    static constexpr int level_mask_covered    = 1; // ghost cells covered by valid cells of this level
    static constexpr int level_mask_notcovered = 2; // ghost cells not covered
    static constexpr int level_mask_physbnd    = 3; // outside domain
};
#endif
