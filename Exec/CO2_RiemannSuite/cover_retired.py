#!/usr/bin/env python3
"""cover_retired.py -- every retired input key and value aborts by name.

Reads the table of retired keys in Source/Hydro/PelantiShyue/PS_retired_keys.H
and runs the 1-D executable once per row with the key set (max_step=0: the
table is checked at startup), asserting a non-zero exit and the row's own
"PSAMR.<key> is retired" message.  Then the retired values of live keys,
which abort where the key is read (max_step=1): ps_relax_mode 1, 3, 4;
ps_coexist_action 2, 3, 4; ps_mt_tau_model 1 (under relax mode 2); a
ps_flux other than wp; ps_bc_nscbc_form=value.  Not covered here: ps_wp_transverse=2, which is read only by the 2-D and 3-D
solver.  Last, a ps_* key under the
wrong prefix, and the bare deck, which must start.  Prints one line per
failure and a PASS/FAIL summary; the exit code is 0 on PASS.

Usage:  [PSAMR_EXE=...] python3 cover_retired.py
        (default: the newest 1-D executable here, as full_suite.py finds it)
"""
import os, re, sys, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import full_suite as F
TABLE = os.path.join(HERE, '..', '..', 'Source', 'Hydro', 'PelantiShyue', 'PS_retired_keys.H')

def run(args, max_step):
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([F.PSAMR, os.path.join(HERE, 'inputs'), 'max_step=%d' % max_step,
                            'amr.plot_int=-1', 'amr.checkpoint_files_output=0'] + args,
                           cwd=d, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr

src = open(TABLE).read()
keys = re.findall(r'\{\s*"([a-z_0-9]+)",\s*"', src)
cases = [(['PSAMR.%s=0' % k], 0, 'PSAMR.%s is retired' % k) for k in keys]
vals = [(['PSAMR.ps_relax_mode=%d' % v], 1, 'PSAMR.ps_relax_mode = %d is retired' % v) for v in (1, 3, 4)]
vals += [(['PSAMR.ps_coexist_action=%d' % v], 1, 'PSAMR.ps_coexist_action=%d is retired' % v) for v in (2, 3, 4)]
vals += [(['PSAMR.ps_relax_mode=2', 'PSAMR.ps_mt_tau_model=1'], 1, '1 is retired'),
         (['PSAMR.ps_flux=hllc'], 1, "PSAMR.ps_flux must be 'wp'"),
         (['PSAMR.ps_bc_use_nscbc=1', 'PSAMR.ps_bc_nscbc_xhi=1', 'PSAMR.ps_bc_nscbc_form=value'], 1, '=value is retired'),
         (['ps_flux=wp'], 0, 'read only under the PSAMR prefix')]
cases += vals
bad = 0
for args, ms, msg in cases:
    rc, out = run(args, ms)
    if rc == 0 or msg not in out:
        bad += 1
        print('FAIL  %-60s rc=%d  expected "%s"' % (' '.join(args), rc, msg))
rc, out = run([], 0)
if rc != 0:
    bad += 1
    print('FAIL  the bare deck does not start (rc=%d)' % rc)
print('%d retired keys, %d retired values, bare deck: %s'
      % (len(keys), len(vals), 'PASS' if bad == 0 else 'FAIL (%d)' % bad))
sys.exit(0 if bad == 0 else 1)
