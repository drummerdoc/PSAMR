#!/bin/bash
# w1.sh tag ref : RS build + FF N1 N4
PSAMR_HOME=${PSAMR_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}
BT=100 $PSAMR_HOME/tools/witness/build.sh RS $1; $PSAMR_HOME/tools/witness/wit.sh FF $1 $2 | tail -2; $PSAMR_HOME/tools/witness/wit.sh N1 $1 $2 | tail -1; $PSAMR_HOME/tools/witness/wit.sh N4 $1 $2 | tail -1
