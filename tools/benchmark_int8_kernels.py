#!/usr/bin/env python3
"""Measure fixed synthetic INT8 shapes on the selected project's local canister."""
import argparse
import hashlib
import json
import re
import time
from canister_infer import Icp, ROOT, decode_blobs, parse_costs, require_local_network, network_status

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--label',required=True)
p.add_argument('--tokens',default='85,86,87,88,128')
args=p.parse_args()
assert re.fullmatch(r'[a-z0-9-]+',args.label)
icp=Icp(ROOT,'local','ic-laya-int8')
require_local_network(icp)
assert network_status(icp)
status=json.loads(icp.run(['canister','status','decision-engine','-e','local','--json']))
out=ROOT/'artifacts/int8_optimization_v2'
report=dict(label=args.label,module_hash=status['module_hash'],
            source_sha256=hashlib.sha256((ROOT/'crates/laya-candle/src/int8.rs').read_bytes()).hexdigest(),rows=[])
(out/f'{args.label}.rs').write_bytes((ROOT/'crates/laya-candle/src/int8.rs').read_bytes())
if 'mod quant;' in (ROOT/'crates/laya-candle/src/int8.rs').read_text():
    quant=(ROOT/'crates/laya-candle/src/int8_quant.rs').read_bytes()
    report['quant_source_sha256']=hashlib.sha256(quant).hexdigest()
    (out/f'{args.label}-quant.rs').write_bytes(quant)
basepath=out/'baseline.json'
baseline=json.loads(basepath.read_text()) if args.label!='baseline' else None
for t in map(int,args.tokens.split(',')):
    for rows,cols in [(3072,1024),(5248,1024),(1024,2624)]:
        started=time.monotonic()
        result=icp.call('decision-engine','benchmark_int8_kernel',f'({t} : nat32, {rows} : nat32, {cols} : nat32)',timeout=600)
        assert 'Ok' in result,result
        costs=parse_costs(result)
        total=max(int(v.replace('_','')) for v in re.findall(r'instructions\s*=\s*([\d_]+)',result))
        checksum=decode_blobs(result)[0].hex()
        if baseline:
            old=next(r for r in baseline['rows'] if (r['tokens'],r['rows'],r['cols'])==(t,rows,cols))
            assert old['checksum']==checksum,(t,rows,cols)
        row=dict(tokens=t,rows=rows,cols=cols,instructions=total,checksum=checksum,costs=costs,wall_seconds=time.monotonic()-started)
        assert sum(c['instructions'] for c in costs)<=total
        report['rows'].append(row)
        (out/f'{args.label}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(args.label,t,rows,cols,total,flush=True)
assert json.loads(icp.run(['canister','status','decision-engine','-e','local','--json']))['module_hash']==report['module_hash']
