#ifndef PSAMR_CFINTERP_H
#define PSAMR_CFINTERP_H

#include <AMReX_Interpolater.H>

//  The McCorquodale--Colella limited quartic coarse-to-fine interpolant
//  (PSAMR.cf_interp = quartic_limited; docs/PLAN_high_order.pdf, Stage 2b).
//
//  Fine values come from AMReX's conservative quartic stencil
//  (ccquartic_interp: the degree-four tensor polynomial whose average over
//  every cell of the five-point stencil is that cell's value).  For each
//  coarse cell and component the 2^DIM fine values are the coarse average
//  plus deviations that sum to zero; the deviations are scaled by one
//  factor theta in [0,1], the largest for which every fine value lies
//  within the range of the coarse cell and its 3^DIM-1 neighbours
//  (McCorquodale and Colella, CAMCoS 6, 2011, section 3.2).  A common
//  factor on zero-mean deviations keeps conservation exactly; the bound
//  keeps every fine value inside the convex hull of its coarse
//  neighbourhood, so a non-negative field stays non-negative and a field
//  uniform in one direction stays uniform.  Away from extrema of a smooth
//  field the bound is inactive and the interpolant is the quartic; at an
//  extremum it clips to the neighbour range, second order there, as the
//  van Leer slope of cell_cons_interp does.  Refinement ratio 2 only.
//  A fine region not aligned to coarse cells (AMReX's derive for tagging
//  fills one ghost cell) is handled by evaluating every coarse cell it
//  touches in full and writing only the fine cells inside it.
class CFInterpQuarticLimited : public amrex::Interpolater
{
public:
    amrex::Box CoarseBox (const amrex::Box& fine, int ratio) override;
    amrex::Box CoarseBox (const amrex::Box& fine, const amrex::IntVect& ratio) override;
    void interp (const amrex::FArrayBox& crse,
                 int                     crse_comp,
                 amrex::FArrayBox&       fine,
                 int                     fine_comp,
                 int                     ncomp,
                 const amrex::Box&       fine_region,
                 const amrex::IntVect&   ratio,
                 const amrex::Geometry&  crse_geom,
                 const amrex::Geometry&  fine_geom,
                 amrex::Vector<amrex::BCRec> const& bcr,
                 int                     actual_comp,
                 int                     actual_state,
                 amrex::RunOn            runon) override;
};

extern CFInterpQuarticLimited psamr_quartic_limited_interp;

#endif
