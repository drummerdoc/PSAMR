PSAMR_HOME      ?= ../..
AMREX_HOME     ?= ../../../amrex

USE_PS_HYDRO = TRUE
DIM = 2
COMP = gnu
PRECISION = DOUBLE

# Profiling
PROFILE = FALSE
COMM_PROFILE = FALSE
TRACE_PROFILE = FALSE
MEM_PROFILE = FALSE
TINY_PROFILE = TRUE

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

# Opt-in diagnostic probes (FINDINGS Addenda 10b-10e): dt-limiter reporter,
# per-stage cell dump, LLF-fallback decomposition.  COMPILE-TIME gated so a
# default build is byte-identical -- the B9 stiff leg is chaotically
# sensitive and even unexecuted added code shifts it.  Build with
#   make USE_PS_DIAG=TRUE ...
# then enable at runtime with PSAMR.ps_dt_diag / ps_cell_diag / ps_llf_diag.
USE_PS_DIAG ?= FALSE
ifeq ($(USE_PS_DIAG),TRUE)
  DEFINES += -DPSAMR_PS_DIAG
endif
