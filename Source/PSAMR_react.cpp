
#include "PSAMR.H"
#include "EOS.H"

using std::string;

using namespace amrex;

void
PSAMR::react(MultiFab& S)
{
    BL_PROFILE("PSAMR::react()");
 
    if (do_react) {
        amrex::Print() << "This is a stub for reactions but nothing is happening yet" << std::endl;
    }
}
