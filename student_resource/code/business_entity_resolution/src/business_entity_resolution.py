#!/usr/bin/env python3
"""Train and run a lightweight business entity resolution pipeline.

Uses only the supplied challenge data. Candidate generation is country-aware
and combines rare name/address token overlap with compact prefix/signature
blocks; a NumPy logistic pair model is learned from labeled and hard-negative
pairs. See README.md for run instructions and limitations.
"""
from __future__ import annotations
import argparse, csv, hashlib, heapq, json, math, os, pickle, random, re, shutil, struct, sys, time, unicodedata
from array import array
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
SUFFIX = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "pvt", "private"}

def norm(s):
    s = unicodedata.normalize("NFKD", (s or "").casefold()).encode("ascii", "ignore").decode()
    return " ".join(TOKEN.findall(s))

def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def read_ids(path):
    """Read only the entity_id column; avoids materializing every field."""
    with open(path, encoding="utf-8", newline="") as f:
        reader=csv.reader(f,delimiter="\t"); header=next(reader,None)
        if not header: return []
        try: index=header.index("entity_id")
        except ValueError: raise ValueError(f"{path} has no entity_id column: {header}")
        return [row[index] for row in reader if row]

def load_rows_for_ids(path, wanted):
    """Normalize only the rows whose IDs are wanted, streaming the rest."""
    found={}
    with open(path, encoding="utf-8", newline="") as f:
        reader=csv.reader(f,delimiter="\t"); header=next(reader,None)
        if not header: raise ValueError(f"{path} is empty")
        columns={name:i for i,name in enumerate(header)}
        index=columns["entity_id"]
        for row in reader:
            if row and row[index] in wanted:
                found[row[index]]=prep_one(dict(zip(header,row)))
    return found

def load_labels_for_ids(path, wanted):
    """Read ground-truth labels only for the selected Source 1 entities."""
    labels={}
    with open(path, encoding="utf-8", newline="") as f:
        reader=csv.reader(f,delimiter="\t"); header=next(reader,None)
        if not header: raise ValueError(f"{path} is empty")
        columns={name:i for i,name in enumerate(header)}
        s1_index=columns["source1_entity_id"]; match_index=columns["matched_entity_ids"]
        for row in reader:
            if row and row[s1_index] in wanted:
                labels[row[s1_index]]=set(filter(None,row[match_index].split(",")))
    return labels

def reservoir_sample(path, positive_ids, pool_size, seed):
    """Stream one reference source and keep a bounded random sample plus positives.

    The full source table is never normalized or held in memory. Labeled
    references for the selected Source 1 rows are always retained so training
    and validation keep their true positives even when blocking misses them.
    pool_size <= 0 keeps every row.
    """
    keep=[]; positives={}; seen=0
    rng=random.Random(seed)
    with open(path, encoding="utf-8", newline="") as f:
        reader=csv.reader(f,delimiter="\t"); header=next(reader,None)
        if not header: raise ValueError(f"{path} is empty")
        columns={name:i for i,name in enumerate(header)}
        index=columns["entity_id"]
        unlimited=pool_size<=0
        for row in reader:
            if not row: continue
            entity_id=row[index]
            if entity_id in positive_ids:
                positives[entity_id]=row
            elif unlimited:
                keep.append(row)
            else:
                seen+=1
                if len(keep)<pool_size:
                    keep.append(row)
                else:
                    slot=rng.randrange(seen)
                    if slot<pool_size: keep[slot]=row
    header_names=list(columns)
    ordered=list(positives.values())+keep
    return [prep_one(dict(zip(header_names,row))) for row in ordered]

def words(s, drop_suffix=False):
    w = {sys.intern(t) for t in TOKEN.findall(s)}
    if drop_suffix: w -= SUFFIX
    return w

def prep_one(r):
    n=norm(r.get("business_name")); a=norm(r.get("business_address"))
    nt=words(n, True); at=words(a)
    digits="".join(c for c in a if c.isdigit())
    postal=[t for t in at if len(t) in (5,6) and t.isdigit()]
    return {"id":r["entity_id"],"country":sys.intern(r.get("country","").casefold()),"name":n,"addr":a,"nt":nt,"at":at,"digits":digits,"postal":postal}

def prep(rows):
    return [prep_one(r) for r in rows]

def iter_prep(path):
    """Normalize source rows lazily to keep the large target set out of RAM."""
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            yield prep_one(row)

def repair_and_count_rows(path):
    """Count completed interim records and discard a possibly partial last line."""
    if not path.exists(): return 0
    with open(path,"r+b") as f:
        f.seek(0,os.SEEK_END); end=f.tell()
        if end:
            f.seek(-1,os.SEEK_END)
            if f.read(1)!=b"\n":
                pos=end
                while pos:
                    size=min(65536,pos); pos-=size; f.seek(pos)
                    chunk=f.read(size); cut=chunk.rfind(b"\n")
                    if cut>=0:
                        f.truncate(pos+cut+1); break
                else: f.truncate(0)
    count=0
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): count+=chunk.count(b"\n")
    return count

def block_keys(r):
    """Yield every blocking key for one prepared record.

    Index construction and query expansion both call this, so a reference and a
    target always agree on which keys exist. Kinds: n/a (name/address tokens),
    p (name prefix), d (address digit prefix), z (postal), s (name signature).
    """
    for t in r["nt"]: yield (r["country"],"n",t)
    for t in r["at"]:
        if len(t)>2: yield (r["country"],"a",t)
    if r["name"]: yield (r["country"],"p",r["name"][:4])
    if r["digits"]: yield (r["country"],"d",r["digits"][:5])
    for p in r["postal"]: yield (r["country"],"z",p)
    # Order-insensitive name twin: the two alphabetically first name tokens.
    # Catches pairs that share distinctive name words but no 4-character prefix
    # ("acme plumbing" vs "plumbing acme") at one posting list per record.
    if r["nt"]: yield (r["country"],"s"," ".join(sorted(r["nt"])[:2]))

def build_index(refs):
    """Build only queryable posting lists using compact integer row IDs."""
    cap=max(500, min(1200, int(len(refs)*0.00015)))
    other_cap=max(cap,2000)
    # candidate_for discards postings above these exact caps. Counting first
    # lets us avoid storing the large lists that would never be queried.
    freq=Counter()
    for r in refs: freq.update(block_keys(r))
    inv=defaultdict(lambda:array("I"))
    for j,r in enumerate(refs):
        for key in block_keys(r):
            limit=cap if key[1] in ("n","a") else other_cap
            if freq[key]<=limit: inv[key].append(j)
    del freq
    return inv,cap

def candidate_for(s, refs, inv, cap, limit):
    """Return one target's ranked candidate reference row indices.

    Ranking only reads the postings, so `refs` is accepted for call-site
    compatibility and may be None when the caller streams reference rows.
    """
    # Rank block keys by posting-list size before expanding them. Querying all
    # address/name tokens against common postings was the main full-corpus
    # runtime bottleneck. Several independent rare keys preserve useful recall.
    keys=list(block_keys(s))
    available=[]
    for key in keys:
        post=inv.get(key,())
        per_key_cap=cap if key[1] in ("n","a") else max(cap,2000)
        if not post or len(post)>per_key_cap: continue
        available.append((len(post),key,post))
    # Prefer the strongest blocks while retaining both name and address evidence.
    available.sort(key=lambda item:item[0])
    selected=[]
    for kind in ("n","a","p","d","z","s"):
        candidates=[item for item in available if item[1][1]==kind]
        selected.extend(candidates[:(2 if kind in ("n","a") else 1)])
    selected.sort(key=lambda item:item[0])
    chosen=selected[:7]
    if not chosen: return []
    # Aggregate postings in NumPy rather than updating a Python Counter for
    # every row ID. This matters at multi-million-target inference scale.
    id_parts=[]; weight_parts=[]
    for npost,key,post in chosen:
        id_parts.append(np.frombuffer(post,dtype=np.uint32))
        weight_parts.append(np.full(npost,1.0/math.log2(2+npost),dtype=np.float64))
    ids=np.concatenate(id_parts)
    weights=np.concatenate(weight_parts)
    unique,inverse=np.unique(ids,return_inverse=True)
    scores=np.bincount(inverse,weights=weights,minlength=len(unique))
    first=np.full(len(unique),len(ids),dtype=np.int64)
    np.minimum.at(first,inverse,np.arange(len(ids),dtype=np.int64))
    order=np.lexsort((first,-scores))[:limit]
    return unique[order].astype(np.int64,copy=False).tolist()

def blocks(targets, refs, limit=300):
    """Multi-pass inverted-token blocking with common-token suppression."""
    inv,cap=build_index(refs)
    result=[]
    for s in targets:
        result.append(candidate_for(s,refs,inv,cap,limit))
    return result

@lru_cache(maxsize=250000)
def chargrams(s):
    return frozenset(s[i:i+3] for i in range(max(0,len(s)-2)))

def features(a,b):
    def jac(x,y): return len(x&y)/(len(x|y) or 1)
    ns=jac(a["nt"],b["nt"]); ads=jac(a["at"],b["at"])
    # Character 3-gram similarity is robust to small spelling noise.
    ng=jac(chargrams(a["name"]),chargrams(b["name"]))
    ag=jac(chargrams(a["addr"]),chargrams(b["addr"]))
    name_cont=min(len(a["nt"]&b["nt"])/(min(len(a["nt"]),len(b["nt"])) or 1),1)
    addr_cont=min(len(a["at"]&b["at"])/(min(len(a["at"]),len(b["at"])) or 1),1)
    return [ns,ads,ng,ag,name_cont,addr_cont,float(a["name"]==b["name"]),float(a["addr"]==b["addr"]),float(bool(set(a["postal"])&set(b["postal"]))),float(a["country"]==b["country"])]

def fit(X,y,steps=180,lr=.08):
    X=np.asarray(X,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1
    Z=(X-mu)/sd; Z=np.column_stack((np.ones(len(Z)),Z)); w=np.zeros(Z.shape[1])
    # Balanced class weights; threshold tuning handles the precision-heavy metric.
    pos=max(1,int(y.sum())); neg=max(1,len(y)-pos)
    sw=np.where(y>0, len(y)/(2*pos),len(y)/(2*neg))
    for _ in range(steps):
        p=1/(1+np.exp(-np.clip(Z@w,-30,30)))
        w-=lr*(Z.T@((p-y)*sw)/len(y)+np.r_[0,w[1:]]*.002)
    return {"mu":mu.tolist(),"sd":sd.tolist(),"w":w.tolist()}

def probs(model,X):
    Z=(np.asarray(X)-model["mu"])/model["sd"]
    Z=np.column_stack((np.ones(len(Z)),Z)); return 1/(1+np.exp(-np.clip(Z@model["w"],-30,30)))

def f05(truth,pred):
    tp=len(truth&pred); p=tp/len(pred) if pred else float(not truth); r=tp/len(truth) if truth else float(not pred)
    return 1.25*p*r/(.25*p+r) if (.25*p+r) else 0.0

def train(args):
    import json, time
    started=time.monotonic()
    def log(message): print(f"[{time.monotonic()-started:6.1f}s] {message}",flush=True)
    d=Path(args.data); tr=d/"train"
    source1_path=tr/"train_source1.tsv"
    # Pass 1: the seeded entity split only needs the ID column.
    ids=read_ids(source1_path)
    if not ids: raise ValueError(f"No Source 1 entities found in {source1_path}")
    rng=random.Random(args.seed); rng.shuffle(ids)
    val_count=min(args.validation_size,len(ids)//5)
    val=ids[:val_count]
    sample=ids[val_count:val_count+args.train_size]
    if not sample: raise ValueError("No training entities selected; check --train-size")
    selected=set(sample)|set(val)
    log(f"Selected {len(sample):,} training and {len(val):,} validation S1 entities from {len(ids):,} rows")
    # Pass 2: normalize and label only the selected Source 1 rows.
    s1map=load_rows_for_ids(source1_path,selected)
    missing=selected-set(s1map)
    if missing: raise ValueError(f"Missing {len(missing)} selected Source 1 rows")
    labels=load_labels_for_ids(tr/"train_ground_truth.tsv",selected)
    # Bounded reference pools: a seeded reservoir per source plus every labeled
    # positive for the selected entities. The multi-gigabyte reference tables
    # are streamed once and never fully materialized or normalized.
    source_pools={}
    for pref,number in (("S2",2),("S3",3)):
        positive_ids={x for sid in selected for x in labels.get(sid,set()) if x.startswith(pref+"-")}
        path=tr/f"train_source{number}.tsv"
        log(f"Sampling up to {args.reference_pool_size:,} {pref} references (+{len(positive_ids):,} labeled positives) from {path.name}...")
        refs=reservoir_sample(path,positive_ids,args.reference_pool_size,args.seed+number)
        source_pools[pref]=refs
        log(f"  {pref} pool ready: {len(refs):,} normalized references")
    # Training targets plus labeled positives and hard negatives from candidate blocks.
    X=[]; y=[]
    for pref in ("S2","S3"):
        refs=source_pools[pref]
        ref_by_id={r["id"]:r for r in refs}
        targets=[s1map[i] for i in sample]
        log(f"Blocking {len(targets):,} training targets against {len(refs):,} {pref} references...")
        cs=blocks(targets,refs,args.max_candidates)
        for t,cands in zip(targets,cs):
            truth=labels.get(t["id"],set())
            # All true pairs, even if blocking missed them, are retained in training.
            pos=[ref_by_id[x] for x in truth if x in ref_by_id and x.startswith(pref+"-")]
            for r in pos: X.append(features(t,r)); y.append(1)
            neg=[refs[j] for j in cands if refs[j]["id"] not in truth]
            rng.shuffle(neg)
            for r in neg[:max(5, min(30, 3*max(1,len(pos))))]: X.append(features(t,r)); y.append(0)
        log(f"  {pref} training pairs so far: {len(y):,} ({sum(y):,} matches)")
    if not X or not any(y) or all(y):
        raise ValueError("Training sample must contain both matching and non-matching pairs; increase --train-size or --reference-pool-size")
    log(f"Fitting logistic model on {len(X):,} labeled pairs...")
    model=fit(X,y); model["threshold"]=0.78
    # Tune precision-weighted F0.5 on a modest held-out S1 sample.
    vals=val[:min(len(val),args.validation_size)]
    if vals:
        vrows=[s1map[i] for i in vals]
        allp=[]
        for pref,refs in source_pools.items():
            cs=blocks(vrows,refs,args.max_candidates)
            for t,cands in zip(vrows,cs):
                for j in cands:
                    r=refs[j]; allp.append((t["id"],r["id"],float(probs(model,[features(t,r)])[0])))
        # Group once by S1. Avoid rescanning the complete candidate list for
        # every validation entity (quadratic work on a sizable validation set).
        grouped=defaultdict(list)
        for sid,rid,p in allp: grouped[sid].append((rid,p))
        best=(0.0,.78)
        # The upper range matters: F0.5 is precision-weighted, and full-corpus
        # calibration selected 0.96-0.99 on disjoint validation entities.
        for th in [.40,.50,.60,.65,.70,.75,.80,.85,.90,.93,.96,.97,.98,.985,.99]:
            scores=[]
            for sid in vals:
                truth=labels.get(sid,set()); pred={rid for rid,p in grouped[sid] if p>=th}; scores.append(f05(truth,pred))
            avg=sum(scores)/len(scores) if scores else 0
            if avg>best[0]: best=(avg,th)
        model["threshold"]=best[1]; model["validation_f05"]=best[0]; model["validation_entities"]=len(vals)
    else: model["validation_f05"]=None
    model["training_pairs"]=len(y); model["training_matches"]=int(sum(y))
    model["reference_pool_size"]=args.reference_pool_size; model["seed"]=args.seed
    model["training_entities"]=len(sample)
    Path(args.model).parent.mkdir(parents=True,exist_ok=True)
    Path(args.model).write_text(json.dumps(model,indent=2),encoding="utf-8")
    log(f"Trained on {len(X):,} labeled candidate pairs; threshold={model['threshold']:.2f}; held-out macro F0.5={model['validation_f05']}")

def _sidecar_key(kb, section):
    """Locate one key in a sidecar through its sorted 64-bit hash index."""
    hashes, keys, keyoff, keylen = section
    h = int.from_bytes(hashlib.blake2b(kb, digest_size=8).digest(), "big")
    lo = int(np.searchsorted(hashes, h, side="left"))
    hi = int(np.searchsorted(hashes, h, side="right"))
    for slot in range(lo, hi):
        start = int(keyoff[slot]); end = start + int(keylen[slot])
        if bytes(keys[start:end]) == kb: return slot
    return -1

def build_reference_sidecar(source_path, out_dir, log=print, chunk_lines=1500000):
    """Stream one reference source into on-disk postings that `candidate_for` can query.

    Peak memory is one sort chunk plus one key's postings, so the full reference
    index never has to be resident. Posting lists are capped exactly as
    `build_index` caps them, and rows are numbered in file order, so candidate
    selection is identical to the in-memory path. Writes refs.bin (length-prefixed
    pickles), keys.bin + hash/offset arrays, and post.bin (row IDs).
    """
    out=Path(out_dir); done=out/"complete.json"
    if done.exists():
        meta=json.loads(done.read_text(encoding="utf-8"))
        log(f"sidecar ready: {out.name} ({meta['rows']:,} rows, {meta['keys']:,} keys, cap {meta['cap']})")
        return meta
    out.mkdir(parents=True,exist_ok=True)
    tmp=out/"tmp"; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True)
    started=time.monotonic()
    chunk=[]; chunk_paths=[]; rows=0
    with open(out/"refs.bin","wb") as refs, open(source_path,encoding="utf-8",newline="") as f:
        offsets=array("Q"); pos=0
        for row in csv.DictReader(f,delimiter="\t"):
            rec=prep_one(row); blob=pickle.dumps(rec,protocol=4)
            refs.write(struct.pack("<I",len(blob))); refs.write(blob)
            offsets.append(pos); pos+=4+len(blob)
            rid=f"{rows:010d}".encode("ascii"); rows+=1
            for key in block_keys(rec):
                kb="|".join(key).encode("ascii")
                # A fixed-width hex hash and decimal row ID keep every record pure
                # ASCII, so no byte can be mistaken for the line separator, while a
                # plain byte sort still groups by key and orders postings by row.
                chunk.append(hashlib.blake2b(kb,digest_size=8).hexdigest().encode("ascii")+kb+b"\t"+rid)
            if len(chunk)>=chunk_lines:
                path=tmp/f"chunk{len(chunk_paths):03d}.bin"; chunk.sort()
                path.write_bytes(b"\n".join(chunk)+b"\n"); chunk_paths.append(path); chunk=[]
        if chunk:
            path=tmp/f"chunk{len(chunk_paths):03d}.bin"; chunk.sort()
            path.write_bytes(b"\n".join(chunk)+b"\n"); chunk_paths.append(path); chunk=[]
        with open(out/"refs.off","wb") as offs_file: offsets.tofile(offs_file)
    del offsets
    log(f"{out.name}: prepared {rows:,} references in {time.monotonic()-started:.0f}s")
    # Same caps build_index derives from the reference count.
    cap=max(500, min(1200, int(rows*0.00015))); other_cap=max(cap,2000)
    hashes=array("Q"); keyoff=array("Q"); keylen=array("I"); postoff=array("Q"); postcnt=array("I")
    keypos=0; total=0
    with open(out/"keys.bin","wb") as keys, open(out/"post.bin","wb") as post:
        state={"key":None,"hash":0,"ids":array("I")}
        def emit():
            nonlocal keypos, total
            key=state["key"]
            if key is None: return
            limit=cap if key[key.find(b"|")+1:key.find(b"|")+2] in (b"n",b"a") else other_cap
            ids=state["ids"]
            # A key over its cap is stored nowhere, so candidate_for drops it, which
            # is the same outcome as build_index leaving it out of the index.
            if len(ids)<=limit:
                keys.write(key)
                hashes.append(state["hash"]); keyoff.append(keypos); keylen.append(len(key)); keypos+=len(key)
                postoff.append(total); postcnt.append(len(ids)); total+=len(ids)
                post.write(ids.tobytes())
        for line in heapq.merge(*[iter_file(p) for p in chunk_paths]):
            # Layout: 16 hex hash chars, key bytes, one tab, 10-digit row ID.
            kb=line[16:-11]
            if kb!=state["key"]:
                emit()
                state={"key":kb,"hash":int(line[:16],16),"ids":array("I")}
            state["ids"].append(int(line[-10:]))
        emit()
    for name,vals,dtype in (("keys.hash",hashes,"Q"),("keys.off",keyoff,"Q"),("keys.len",keylen,"I"),("post.off",postoff,"Q"),("post.cnt",postcnt,"I")):
        np.asarray(vals,dtype=np.dtype(dtype).newbyteorder("<")).tofile(str(out/name))
    shutil.rmtree(tmp,ignore_errors=True)
    meta={"rows":rows,"keys":len(hashes),"postings":total,"cap":cap,"other_cap":other_cap,
          "source":Path(source_path).name,"seconds":round(time.monotonic()-started,1)}
    (out/"meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    done.write_text(json.dumps(meta),encoding="utf-8")
    log(f"{out.name}: {meta['keys']:,} indexed keys, {total:,} postings, cap {cap}, {meta['seconds']}s")
    return meta

def iter_file(path):
    """Yield records with exactly the separator byte the writer appended removed."""
    with open(path,"rb") as f:
        for line in f:
            if line.endswith(b"\n"): line=line[:-1]
            if line: yield line

def open_sidecar(out_dir):
    """Memory-map one source's sidecar; resident pages follow the keys actually queried."""
    out=Path(out_dir)
    meta=json.loads((out/"complete.json").read_text(encoding="utf-8"))
    section={"meta":meta,
             "hashes":np.memmap(out/"keys.hash",dtype="<u8",mode="r"),
             "keys":np.memmap(out/"keys.bin",dtype="u1",mode="r"),
             "keyoff":np.memmap(out/"keys.off",dtype="<u8",mode="r"),
             "keylen":np.memmap(out/"keys.len",dtype="<u4",mode="r"),
             "postoff":np.memmap(out/"post.off",dtype="<u8",mode="r"),
             "postcnt":np.memmap(out/"post.cnt",dtype="<u4",mode="r"),
             "posts":np.memmap(out/"post.bin",dtype="u1",mode="r"),
             "refoff":np.memmap(out/"refs.off",dtype="<u8",mode="r"),
             "refs_path":out/"refs.bin"}
    return section

def close_sidecar(section):
    """Release mapped files so a worker can exit or a directory can be removed."""
    for value in list(section.values()):
        if isinstance(value,np.memmap): value._mmap.close()
    section.clear()

def load_sidecar_postings(section, wanted):
    """Return {key: array('I')} for exactly the requested keys.

    All keys of a shard are located in one vectorized search over the sorted hash
    index, then verified byte for byte. Resolving them one at a time faulted a page
    per probe and made this the slowest part of a full-corpus run.
    """
    if not wanted: return {}
    encoded=[(key,"|".join(key).encode("ascii")) for key in wanted]
    probes=np.fromiter((int.from_bytes(hashlib.blake2b(kb,digest_size=8).digest(),"big") for _,kb in encoded),
                       dtype=np.uint64,count=len(encoded))
    table=section["hashes"]
    slots=np.searchsorted(table,probes,side="left")
    keys=section["keys"]; keyoff=section["keyoff"]; keylen=section["keylen"]
    postoff=section["postoff"]; postcnt=section["postcnt"]; posts=section["posts"]
    total_keys=len(table); inv={}
    for index,(key,kb) in enumerate(encoded):
        probe=int(probes[index]); slot=int(slots[index])
        while slot<total_keys and int(table[slot])==probe:
            start=int(keyoff[slot])
            if bytes(keys[start:start+int(keylen[slot])])==kb:
                offset=int(postoff[slot]); count=int(postcnt[slot])
                values=array("I"); values.frombytes(bytes(posts[offset*4:(offset+count)*4]))
                inv[key]=values
                break
            slot+=1
    return inv

def _score_shard(job):
    """Score one slice of Source 1 for one reference source, writing `s1\tcand\tmatched`.

    Candidate selection needs only the postings, so the references are then read in
    one forward sweep of the sidecar instead of one random read per candidate. On a
    memory-constrained machine that was the difference between minutes and hours:
    nothing stays cached across targets, so each random read was a disk seek.
    """
    sidecar_dir, target_path, start_byte, limit_rows, model_path, threshold, max_cands, out_path = job
    section=open_sidecar(sidecar_dir); meta=section["meta"]
    model=json.loads(Path(model_path).read_text(encoding="utf-8"))
    targets=[]
    with open(target_path,encoding="utf-8",newline="") as f:
        # Seeking past the header means the shard's first data row would be read
        # as the header, so the column names are passed in explicitly.
        header=f.readline().rstrip("\r\n").split("\t")
        f.seek(start_byte)
        for row in csv.DictReader(f,fieldnames=header,delimiter="\t"):
            targets.append(prep_one(row))
            if len(targets)>=limit_rows: break
    # Only this slice's keys are materialized, which is what keeps memory bounded.
    inv=load_sidecar_postings(section,{key for t in targets for key in block_keys(t)})
    cap=meta["cap"]
    picked=[candidate_for(t,None,inv,cap,max_cands) for t in targets]
    del inv
    starts=np.zeros(len(picked)+1,dtype=np.int64)
    np.cumsum([len(p) for p in picked],out=starts[1:])
    total=int(starts[-1])
    owner=np.empty(total,dtype=np.int32); position=np.empty(total,dtype=np.int32); ref_ids=np.empty(total,dtype=np.int32)
    cursor=0
    for index,pairs in enumerate(picked):
        if not pairs: continue
        stop=cursor+len(pairs)
        owner[cursor:stop]=index
        position[cursor:stop]=np.arange(len(pairs),dtype=np.int32)
        ref_ids[cursor:stop]=np.asarray(pairs,dtype=np.int32)
        cursor=stop
    order=np.argsort(ref_ids,kind="stable")
    ref_ids=ref_ids[order]; owner=owner[order]; position=position[order]
    slots=np.argsort(order)
    matrix=np.zeros((total,10),dtype=np.float64)
    ids=np.empty(total,dtype=object)
    refs_path=str(section["refs_path"]); refoff=section["refoff"]
    if total:
        with open(refs_path,"rb") as fh:
            bounds=np.flatnonzero(ref_ids[1:]!=ref_ids[:-1])+1
            group_start=np.r_[0,bounds]; group_stop=np.r_[bounds,total]
            for begin,end in zip(group_start.tolist(),group_stop.tolist()):
                fh.seek(int(refoff[int(ref_ids[begin])]))
                size=struct.unpack("<I",fh.read(4))[0]
                ref=pickle.loads(fh.read(size))
                ids[begin:end]=[ref["id"]]*(end-begin)
                for row in range(begin,end):
                    matrix[row]=features(targets[int(owner[row])],ref)
    probabilities=np.zeros(total,dtype=np.float64)
    for index in range(len(targets)):
        begin,stop=int(starts[index]),int(starts[index+1])
        if stop<=begin: continue
        # Restore candidate order so each target is scored exactly as the
        # in-memory path scores it.
        rows=slots[begin:stop]
        rows=rows[np.argsort(position[rows],kind="stable")]
        probabilities[begin:stop]=probs(model,matrix[rows])
    seen=candidates=matches=0
    with open(out_path,"w",encoding="utf-8",newline="") as out:
        w=csv.writer(out,delimiter="\t",lineterminator="\n")
        for index,t in enumerate(targets):
            begin,stop=int(starts[index]),int(starts[index+1])
            if stop<=begin:
                w.writerow([t["id"],"",""]); seen+=1; continue
            rows=slots[begin:stop]
            rows=rows[np.argsort(position[rows],kind="stable")]
            names=ids[rows]; scores=probabilities[begin:stop]
            kept=[name for name,score in zip(names,scores) if score>=threshold]
            w.writerow([t["id"],",".join(names),",".join(kept)])
            seen+=1; candidates+=len(names); matches+=len(kept)
    return {"rows":seen,"candidates":candidates,"matches":matches,"path":out_path}

def _shard_plan(path, shard_rows):
    """(byte offset, row count) for each Source 1 slice, so workers seek instead of scan."""
    plan=[]
    with open(path,"rb") as f:
        f.readline()
        while True:
            offset=f.tell(); rows=0
            while rows<shard_rows:
                if not f.readline(): break
                rows+=1
            if not rows: break
            plan.append((offset,rows))
            if rows<shard_rows: break
    return plan

def predict_streaming(args):
    """Full-corpus prediction with a bounded resident set.

    The reference index lives on disk and each worker materializes postings only
    for the keys its own Source 1 slice queries, so total memory stays flat as
    the corpora grow.
    """
    from multiprocessing import Pool
    started=time.monotonic()
    def log(message): print(f"[{time.monotonic()-started:7.1f}s] {message}",flush=True)
    model=json.loads(Path(args.model).read_text(encoding="utf-8"))
    d=Path(args.data); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cache=Path(args.cache) if args.cache else out/"sidecar"
    target_path=d/"test_source1.tsv"
    thr=args.threshold if args.threshold is not None else model["threshold"]
    sources=[("s2",d/"test_source2.tsv"),("s3",d/"test_source3.tsv")]
    plan=_shard_plan(target_path,args.shard_rows)
    shard_dir=out/"shards"; shard_dir.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for tag,source_path in sources:
        build_reference_sidecar(source_path,cache/tag,log=log,chunk_lines=args.sort_chunk)
        for index,(start_byte,rows) in enumerate(plan):
            path=shard_dir/f"{index:05d}.{tag}.tsv"
            # Completed shards are kept, so an interrupted run resumes where it stopped.
            if path.exists() and path.stat().st_size and path.read_bytes()[-1:]==b"\n":
                continue
            jobs.append((str(cache/tag),str(target_path),start_byte,rows,args.model,thr,args.max_candidates,str(path)))
    log(f"{len(plan)} shards x {len(sources)} sources; {len(jobs)} tasks pending, workers={args.workers}")
    if jobs:
        with Pool(args.workers) as pool:
            done=0
            for result in pool.imap_unordered(_score_shard,jobs,chunksize=1):
                done+=1
                log(f"shard done {done}/{len(jobs)}: {result['rows']:,} rows, {result['candidates']:,} candidates, {result['matches']:,} matches")
    match_path=out/"matching_results.tsv"; candidate_path=out/"candidate_pairs.tsv"
    total_candidates=total_matches=0; written=0
    with open(match_path,"w",encoding="utf-8",newline="") as mf, open(candidate_path,"w",encoding="utf-8",newline="") as cf:
        mw=csv.writer(mf,delimiter="\t",lineterminator="\n"); cw=csv.writer(cf,delimiter="\t",lineterminator="\n")
        mw.writerow(["source1_entity_id","matched_entity_ids"]); cw.writerow(["source1_entity_id","candidate_entity_ids"])
        for index in range(len(plan)):
            paths={tag:shard_dir/f"{index:05d}.{tag}.tsv" for tag,_ in sources}
            if not all(p.exists() for p in paths.values()): raise RuntimeError(f"shard {index} is incomplete; rerun predict")
            handles=[open(paths[tag],encoding="utf-8",newline="") for tag,_ in sources]
            try:
                readers=[csv.reader(h,delimiter="\t") for h in handles]
                for records in zip(*readers):
                    if len({r[0] for r in records})!=1: raise ValueError(f"shard {index}: Source 1 rows disagree across reference sources")
                    candidates=",".join(filter(None,(r[1] for r in records)))
                    matches=",".join(filter(None,(r[2] for r in records)))
                    mw.writerow([records[0][0],matches]); cw.writerow([records[0][0],candidates])
                    total_candidates+=len(candidates.split(",")) if candidates else 0
                    total_matches+=len(matches.split(",")) if matches else 0
                    written+=1
            finally:
                for h in handles: h.close()
    if not args.keep_shards: shutil.rmtree(shard_dir,ignore_errors=True)
    log(f"Wrote predictions for {written:,} S1 entities; candidates={total_candidates:,}, matches={total_matches:,}, threshold={thr:.2f}")
    return {"entities":written,"candidates":total_candidates,"matches":total_matches}

def predict(args):
    if getattr(args,"streaming_references",False): return predict_streaming(args)
    model=json.loads(Path(args.model).read_text(encoding="utf-8")); d=Path(args.data); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    target_path=d/"test_source1.tsv"
    with open(target_path,encoding="utf-8",newline="") as f: target_count=sum(1 for _ in csv.DictReader(f,delimiter="\t"))
    thr=args.threshold if args.threshold is not None else model["threshold"]
    temp_paths=[]
    # Score one source at a time and stream S1 rows. The reference index is the
    # largest resident structure; keeping all 1.7M normalized targets as well
    # caused paging on the challenge machine.
    for pref in ("S2","S3"):
        source_path=d/f"test_source{2 if pref=='S2' else 3}.tsv"
        refs=prep(load(source_path)); inv,cap=build_index(refs)
        tmp=out/f".{pref.lower()}_scored.tsv"; temp_paths.append(tmp)
        resume=repair_and_count_rows(tmp)
        if resume>target_count: raise ValueError(f"{tmp} has {resume} rows, more than the {target_count} test targets")
        mode="a" if resume else "w"
        with open(tmp,mode,encoding="utf-8",newline="") as f:
            w=csv.writer(f,delimiter="\t",lineterminator="\n")
            for i,t in enumerate(iter_prep(target_path),1):
                if i<=resume: continue
                cand=candidate_for(t,refs,inv,cap,args.max_candidates)
                rows=[refs[j] for j in cand]
                ps=probs(model,[features(t,r) for r in rows]) if rows else []
                matched=[r["id"] for r,p in zip(rows,ps) if p>=thr]
                w.writerow([t["id"],",".join(r["id"] for r in rows),",".join(matched)])
                if i%100000==0: print(f"{pref}: processed {i:,}/{target_count:,} Source 1 records",flush=True)
        del inv,refs
    # Merge the two source passes into the challenge's one-row-per-S1 files.
    match_path=out/"matching_results.tsv"; candidate_path=out/"candidate_pairs.tsv"
    with open(temp_paths[0],encoding="utf-8",newline="") as f2, open(temp_paths[1],encoding="utf-8",newline="") as f3, open(match_path,"w",encoding="utf-8",newline="") as mf, open(candidate_path,"w",encoding="utf-8",newline="") as cf:
        r2=csv.reader(f2,delimiter="\t"); r3=csv.reader(f3,delimiter="\t")
        mw=csv.writer(mf,delimiter="\t",lineterminator="\n"); cw=csv.writer(cf,delimiter="\t",lineterminator="\n")
        mw.writerow(["source1_entity_id","matched_entity_ids"]); cw.writerow(["source1_entity_id","candidate_entity_ids"])
        total_candidates=total_matches=0
        for a,b in zip(r2,r3):
            if a[0]!=b[0]: raise ValueError(f"Source output order mismatch: {a[0]} vs {b[0]}")
            candidates=",".join(filter(None,(a[1],b[1]))); matches=",".join(filter(None,(a[2],b[2])))
            mw.writerow([a[0],matches]); cw.writerow([a[0],candidates])
            total_candidates+=len(candidates.split(",")) if candidates else 0
            total_matches+=len(matches.split(",")) if matches else 0
    for p in temp_paths: p.unlink(missing_ok=True)
    print(f"Wrote predictions for {target_count:,} S1 entities; candidates={total_candidates:,}, matches={total_matches:,}, threshold={thr:.2f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("train"); a.add_argument("--data",default="student_resource/dataset"); a.add_argument("--model",default="student_resource/model/entity_matcher.json"); a.add_argument("--seed",type=int,default=42); a.add_argument("--train-size",type=int,default=3000); a.add_argument("--validation-size",type=int,default=100); a.add_argument("--reference-pool-size",type=int,default=200000); a.add_argument("--max-candidates",type=int,default=80); a.set_defaults(func=train)
    a=sub.add_parser("predict"); a.add_argument("--data",default="student_resource/dataset/test"); a.add_argument("--model",default="student_resource/model/entity_matcher.json"); a.add_argument("--output",default="student_resource/output"); a.add_argument("--threshold",type=float); a.add_argument("--max-candidates",type=int,default=80); a.add_argument("--streaming-references",action="store_true"); a.add_argument("--cache",default=None); a.add_argument("--workers",type=int,default=max(1,min(4,(os.cpu_count() or 2)//2))); a.add_argument("--shard-rows",type=int,default=50000); a.add_argument("--sort-chunk",type=int,default=1500000); a.add_argument("--keep-shards",action="store_true"); a.set_defaults(func=predict)
    args=ap.parse_args(); args.func(args)
