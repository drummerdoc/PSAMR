PSAMR_HOME      ?= ../..
AMREX_HOME     ?= ../../../amrex

USE_PS_HYDRO = TRUE

# AMReX.  Phase-0 pipe-break is intrinsically 2-D (the flashing jet), so
# DIM defaults to 2.  DIM=1 still builds (degenerates to a slot along x).
DIM ?= 2
COMP = gnu
PRECISION = DOUBLE

# Profiling
PROFILE = FALSE
COMM_PROFILE = FALSE
TRACE_PROFILE = FALSE
MEM_PROFILE = FALSE
TINY_PROFILE = FALSE

# Performance
USE_MPI = TRUE
USE_OMP = FALSE
USE_CUDA = FALSE
USE_HIP = FALSE
USE_SYCL = FALSE

# Debugging
DEBUG = FALSE

# PSAMR
USE_EB = FALSE
Eos_Model := PR

# GNU Make
Bpack := ./Make.package
Blocs := .

include $(PSAMR_HOME)/Exec/Make.PSAMR
