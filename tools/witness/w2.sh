#!/bin/bash
# w2.sh tag ref : N1b + battery defaults vs BASE (3rd arg, default BASE4_nc_defaults)
PSAMR_HOME=${PSAMR_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}
$PSAMR_HOME/tools/witness/wit.sh N1b $1 $2 | tail -1
cd $PSAMR_HOME/Exec/CO2_RiemannSuite && export PSAMR_EXE=$PWD/exe_keep/nc_$1.ex
timeout 75 python3 characterize.py record AUD_$1_nc --defaults 2>/dev/null | tail -1
python3 characterize.py compare ${3:-BASE4_nc_defaults} AUD_$1_nc 2>/dev/null | grep "identical,\|CHANGED" | cut -c1-160
