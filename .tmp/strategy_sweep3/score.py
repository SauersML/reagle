import subprocess,sys,math
truth,imp=sys.argv[1],sys.argv[2]

def rows(vcf, fmt):
    p=subprocess.Popen(["bcftools","query","-f",fmt,vcf],stdout=subprocess.PIPE,text=True)
    for line in p.stdout:
        f=line.rstrip("\n").split("\t")
        if len(f)<5: continue
        yield f
    p.wait()

def parse_ds(gt,ds,gp):
    if ds and ds!='.':
        if ',' in ds:
            s=0.0
            ok=False
            for t in ds.split(','):
                if t and t!='.':
                    s+=float(t); ok=True
            if ok: return s
        else:
            return float(ds)
    if gp and gp!='.':
        toks=gp.split(',')
        if len(toks)>=3:
            p0,p1,p2=map(float,toks[:3]); s=p0+p1+p2
            if s>0: return (p1+2*p2)/s
    if gt in ('','.','./.','.|.'): return None
    sep='|' if '|' in gt else '/'
    ps=gt.split(sep)
    if len(ps)!=2: return None
    try:
        a,b=int(ps[0]),int(ps[1])
    except:
        return None
    return (1.0 if a!=0 else 0.0)+(1.0 if b!=0 else 0.0)

def gt_ds(gt):
    if gt in ('','.','./.','.|.'): return None
    sep='|' if '|' in gt else '/'
    ps=gt.split(sep)
    if len(ps)!=2: return None
    try: a,b=int(ps[0]),int(ps[1])
    except: return None
    return (1.0 if a!=0 else 0.0)+(1.0 if b!=0 else 0.0)

it=rows(truth,"%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n")
ii=rows(imp,"%CHROM\t%POS\t%REF\t%ALT[\t%GT:%DS:%GP]\n")
try: t=next(it)
except StopIteration: t=None
try: i=next(ii)
except StopIteration: i=None
sx=sy=sxx=syy=sxy=0.0;n=0
while t and i:
    kt=(t[0],int(t[1])); ki=(i[0],int(i[1]))
    if kt<ki:
        try:t=next(it)
        except StopIteration:t=None
        continue
    if kt>ki:
        try:i=next(ii)
        except StopIteration:i=None
        continue
    t_ref,t_alt=t[2],t[3]; i_ref,i_alt=i[2],i[3]
    bial=(',' not in t_alt)
    swapped=bial and t_ref==i_alt and t_alt==i_ref
    if bial and (swapped or (t_ref==i_ref and t_alt==i_alt)):
        tv=gt_ds(t[4])
        gt,ds,gp=(i[4].split(':')+[None,None,None])[:3]
        iv=parse_ds(gt,ds,gp)
        if tv is not None and iv is not None:
            if swapped: iv=2.0-iv
            sx+=tv; sy+=iv; sxx+=tv*tv; syy+=iv*iv; sxy+=tv*iv; n+=1
    try:t=next(it)
    except StopIteration:t=None
    try:i=next(ii)
    except StopIteration:i=None
mx=sx/n; my=sy/n
vx=sxx/n-mx*mx; vy=syy/n-my*my; cov=sxy/n-mx*my
r2=(cov*cov)/(vx*vy) if n>1 and vx>0 and vy>0 else float('nan')
print(f"n={n} r2={r2:.6f}")
