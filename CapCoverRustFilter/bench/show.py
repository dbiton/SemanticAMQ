import sys, json
for l in sys.stdin:
    if not l.startswith("{"): continue
    r = json.loads(l); print(r['filter'], 'dis', r['disagree'], *(f'{k[4:]}={v:.0f}' for k, v in r.items() if k.startswith('new')))
