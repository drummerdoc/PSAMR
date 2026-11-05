#!/bin/sh
# move git lock/tmp debris into _to_delete/
PSAMR_HOME=${PSAMR_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}
cd $PSAMR_HOME; mkdir -p _to_delete
for f in $(find .git -name "*.lock" -o -name "tmp_obj_*" 2>/dev/null); do
  mv -n "$f" "_to_delete/$(basename $f).$(date +%s%N)"; done
ls _to_delete | wc -l
