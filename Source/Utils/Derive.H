#ifndef Derive_H
#define Derive_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBCellFlag.H>
#endif

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
get_idx(const int i, int& im, int& ip)
{
  im = i - 1;
  ip = i + 1;
}

#ifdef AMREX_USE_EB
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
get_idx(
  const int i,
  const int dir,
  const bool all_regular,
  const amrex::EBCellFlag& flag,
  int& im,
  int& ip)
{
  if (all_regular) {
    get_idx(i, im, ip);
  } else {
    if (flag.isCovered()) {
      im = i;
      ip = i;
    } else {
      const amrex::IntVect ivm = -amrex::IntVect::TheDimensionVector(dir);
      const amrex::IntVect ivp = amrex::IntVect::TheDimensionVector(dir);
      im = i - flag.isConnected(ivm);
      ip = i + flag.isConnected(ivp);
    }
  }
}
#endif

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
get_weight(const int im, const int ip)
{
  const int diff = ip - im;
  return diff == 0 ? 0.0 : (diff == 1 ? 1.0 : 0.5);
}

void PSAMR_dernull(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_dervelx(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_dervely(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_dervelz(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_dermagvel(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_dermagmom(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_derkineng(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_dereint1(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_dereint2(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_derlogden(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_derspec(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

#ifdef USE_PS_HYDRO
// Pelanti-Shyue phase-1 volume fraction alpha_1 as a derive (with the
// standard ghost FillPatch) so AMR can refine on the phase/flash front.
// Not the EB fluid volume fraction "vfrac"; datfab component 0 = UALPHA1.
void PSAMR_der_ps_alpha1(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);
#endif

void PSAMR_dermagvort(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_derdivu(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_derenstrophy(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_dersoundspeed(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_dermachnumber(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

void PSAMR_derpres(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);


void PSAMR_dertemp(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);

#ifdef USE_PS_HYDRO
// Per-phase temperatures (liquid / vapor); vanished phase -> mixture T.
void PSAMR_dertemp1(
  const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
  const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
  amrex::Real time, const int* bcrec, const int level);
void PSAMR_dertemp2(
  const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
  const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
  amrex::Real time, const int* bcrec, const int level);
#endif

void PSAMR_dercp(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real time,
  const int* bcrec,
  const int level);
#endif
