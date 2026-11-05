#include <AMReX_LevelBld.H>

#include "PSAMR.H"

class PSAMRBld : public amrex::LevelBld
{
  void variableSetUp() override;
  void variableCleanUp() override;
  amrex::AmrLevel* operator()() override;
  amrex::AmrLevel* operator()(
    amrex::Amr& papa,
    int lev,
    const amrex::Geometry& level_geom,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    amrex::Real time) override;
};

PSAMRBld PSAMR_bld;

amrex::LevelBld*
getLevelBld()
{
  return &PSAMR_bld;
}

void
PSAMRBld::variableSetUp()
{
  PSAMR::variableSetUp();
}

void
PSAMRBld::variableCleanUp()
{
  PSAMR::variableCleanUp();
}

amrex::AmrLevel*
PSAMRBld::operator()()
{
  return new PSAMR;
}

amrex::AmrLevel*
PSAMRBld::operator()(
  amrex::Amr& papa,
  int lev,
  const amrex::Geometry& level_geom,
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm,
  amrex::Real time)
{
  return new PSAMR(papa, lev, level_geom, ba, dm, time);
}
