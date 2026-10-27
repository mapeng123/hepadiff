#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P-MDDM v4: 严格预注册排除版 (回应 JHP R1/R2, 投 CMH)
相对 v3 的改动 (超参不动, 保证可比):
  1) Tier1: 40 个 VCC 评估基因按基因名从全部扰动数据集剔除 (任何上下文未见)
  2) Tier2: gwps 中 seed=2026 随机 150 个非 Tier1 基因剔除 (非必需基因 discrimination/PDS 测试集)
  3) Tier3: GSE264667 HepG2+Jurkat 文件整体不进训练 (跨上下文零样本测试)
  4) 扰动类型嵌入: LOF (CRISPRi/KO) / GOF (CRISPRa, Norman) / compound (sci-Plex, McFarland)
  5) species 嵌入: human/mouse/other/cellline (语料 meta 需含 species 列; 缺失回退 NULL)
  6) 排除清单 manifest 存入检查点 (可审计)
"""
import time, os, json, glob, re, math, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42); np.random.seed(42)
DEV = "cuda"
BONE_MM = "/root/autodl-tmp/virtual-liver/processed/v4_mm_r1"
PERT_DIR = "/root/autodl-tmp/perturb_data"
CKPT = "/root/autodl-tmp/virtual-liver/checkpoints_pmdm_v5_donorho"
GENES_F = "/root/autodl-tmp/virtual-liver/processed/liver_genes_v4.txt"
EDGES_F = "/root/autodl-tmp/virtual-liver/processed/liver_bin_edges_v4.npy"
META_F = "meta_v6_r1.csv"          # v4 语料重建版 meta (含 species, condition_v3)
COND_COL = "condition_v3"          # 病因 x 病理双轴本体; 缺失时回退 condition_v2
os.makedirs(CKPT, exist_ok=True)

STEPS, BATCH = 60000, 96   # SF9 donor hold-out 减步对照
D_MODEL, N_HEAD, N_LAYER = 512, 8, 12
CHUNK, LR = 48, 2e-4
MASK_TOKEN = 51
P_DROP = 0.15
SAVE_EVERY = 5000
MIN_NNZ = 50
ZERO_FRAC = 0.1
NZ_W = 3.0
P_FULL, P_TRAJ = 0.15, 0.25
DE_W = 2.0
PERT_FRAC = 0.5
W_DELTA_HEAD = 0.5
W_DELTA_GEN = 0.2
P_FULLMASK_PERT = 0.3
P_PER_BATCH = 4
DE_TOPN = 200
GID_DIM = 256
N_TIER2 = 150
TIER2_SEED = 2026

# ---------- Tier1: 40 个 VCC 评估基因 (ST5 名单, 按基因名从所有扰动数据剔除) ----------
TIER1_40 = ["KRR1","FCF1","TSR2","DDX47","TIMM23B","EIF2B2","QARS","CHMP6","AATF","EIF2B5",
            "NOP16","HSPA9","SAMM50","MED19","RPL24","GPN3","EXOSC2","KDM1A","RPL23","HARS",
            "MED22","MED12","MED11","WDR82","HSD17B12","FARSA","GTF2H1","EXOSC8","NOL11","POLR1A",
            "ERCC2","SUPT5H","RARS2","TAF2","TAF8","ZC3H13","NKX6-1","EIF3D","PSMD7","DBR1"]

# ---------- 扰动类型映射 (按数据集 tag) ----------
PERT_TYPE = {"HepG2": "LOF", "Jurkat": "LOF", "K562-gwps": "LOF", "K562-essential": "LOF",
             "RPE1": "LOF", "CCLE": "compound", "Norman": "GOF",
             "sciplex2": "compound", "sciplex3": "compound", "teloHAEC": "LOF"}
PTYPE_VOCAB = {"LOF": 0, "GOF": 1, "compound": 2}
N_PTYPE = 4                       # 3 + NULL

# ---------- 器官语料 memmap ----------
print("加载器官语料 memmap ...", flush=True)
mm_data = np.load(f"{BONE_MM}/data.npy", mmap_mode='r')
mm_index = np.load(f"{BONE_MM}/indices.npy", mmap_mode='r')
mm_indptr = np.load(f"{BONE_MM}/indptr.npy", mmap_mode='r')
import ctypes, threading
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
for _p in [f"{BONE_MM}/data.npy", f"{BONE_MM}/indices.npy", f"{BONE_MM}/indptr.npy"]:
    _fd = os.open(_p, os.O_RDONLY); _libc.posix_fadvise(_fd, 0, 0, 1); os.close(_fd)
def _drop():
    while True:
        time.sleep(60)
        for _p in [f"{BONE_MM}/data.npy", f"{BONE_MM}/indices.npy", f"{BONE_MM}/indptr.npy"]:
            try:
                _fd = os.open(_p, os.O_RDONLY); _libc.posix_fadvise(_fd, 0, 0, 4); os.close(_fd)
            except OSError: pass
threading.Thread(target=_drop, daemon=True).start()

meta = pd.read_csv(f"{BONE_MM}/{META_F}", index_col=0, low_memory=False)
if COND_COL not in meta.columns:
    COND_COL = "condition_v2"
    print(f"!! {META_F} 无 condition_v3, 回退 condition_v2", flush=True)
meta['condition'] = meta[COND_COL].astype(str)
# v5: 标签修复 (condition_v4) + 非肝样本剔除
import pandas as _pd, numpy as _np
_v4 = _pd.read_csv(f"{BONE_MM}/condition_v4.csv")
assert len(_v4) == len(meta), (len(_v4), len(meta))
meta['condition'] = _v4['condition_v4'].astype(str).to_numpy()
_EXCL = _v4['exclude'].astype(bool).to_numpy() if 'exclude' in _v4.columns else _np.zeros(len(meta), bool)
print(f"v5 标签修复: condition_v4 生效, 非肝剔除 {int(_EXCL.sum())} 细胞", flush=True)
meta['celltype_v5'] = meta['celltype_v5'].astype(str)
HAS_SPECIES = 'species' in meta.columns
if HAS_SPECIES:
    meta['species'] = meta['species'].astype(str).str.lower()
else:
    print("!! meta 无 species 列, species 嵌入全部走 NULL (R2-1 修复需重建语料)", flush=True)
genes = [l.strip() for l in open(GENES_F)]
G = len(genes); gidx = {g: i for i, g in enumerate(genes)}
N_CHUNK = (G + CHUNK - 1) // CHUNK
G_PAD = N_CHUNK * CHUNK
edges = np.load(EDGES_F)

nnz = np.diff(mm_indptr)
keep_idx = np.where(nnz >= MIN_NNZ)[0]
keep_idx = keep_idx[~_EXCL[keep_idx]]
print(f"v5 剔除非肝后训练池: {len(keep_idx)}", flush=True)
del nnz
# ---------- SF9: donor-level hold-out (106 donor 整供体不进训练/验证) ----------
_sf9 = json.load(open(f"{BONE_MM}/sf9_holdout_donors.json"))
_ho_pairs = set(tuple(x.split("||", 1)) for x in _sf9["donors"])
_sp_arr = meta['species'].astype(str).str.lower().to_numpy() if HAS_SPECIES else np.array([""] * len(meta))
_sm_arr = meta['sample'].astype(str).to_numpy()
_ho_mask = np.array([(s, d) in _ho_pairs for s, d in zip(_sp_arr, _sm_arr)])
_n_before = len(keep_idx)
keep_idx = keep_idx[~_ho_mask[keep_idx]]
print(f"SF9 donor hold-out: 剔除 {_n_before - len(keep_idx)} 细胞 "
      f"({len(_ho_pairs)} donors), 训练池剩 {len(keep_idx)}", flush=True)
# ---------- 训练/验证划分: 按细胞类型分层留出 0.5% (稀有类型保底1个), 不参与训练 ----------
VAL_MOD = 197
_ct_arr = meta['celltype_v5'].to_numpy()
_order = np.argsort(_ct_arr[keep_idx], kind='stable')
_sorted_ki = keep_idx[_order]
_sorted_ct = _ct_arr[_sorted_ki]
_grp = np.searchsorted(_sorted_ct, _sorted_ct, side='left')  # 每个元素所属组的起始位置
_pos = np.arange(len(_sorted_ki)) - _grp                      # 组内序号
val_pool = _sorted_ki[(_pos % VAL_MOD) == 0]                  # 每组每197个取1个
_val_set = np.zeros(len(meta), bool)
_val_set[val_pool] = True
keep_idx_train = keep_idx[~_val_set[keep_idx]]
print(f"器官语料 {len(keep_idx)} 细胞 x {G} 基因 | cond列={COND_COL} | species={HAS_SPECIES} | "
      f"train {len(keep_idx_train)} val {len(val_pool)} (按细胞类型分层)", flush=True)
os.makedirs(CKPT, exist_ok=True)
np.savez(f"{CKPT}/val_split.npz", val_pool=val_pool, val_ct=_ct_arr[val_pool])

# ---------- 扰动数据 (Tier3: HepG2 文件不进训练) ----------
print("加载扰动数据 (Tier3 排除 HepG2) ...", flush=True)
import anndata as ad

def load_perturb_h5ad(path, obs_col='perturbation', ctrl_label='control', max_cells=400000, seed=0, per_pert=300):
    import gc
    cache = path + ".tokcache.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print("  %s: 缓存命中 %d 细胞" % (os.path.basename(path), z["tok"].shape[0]), flush=True)
        return z["tok"], z["pert"]
    a = ad.read_h5ad(path)
    if obs_col not in a.obs.columns:
        raise ValueError(f"{path} 无 {obs_col} 列: {list(a.obs.columns)[:10]}")
    common = [g for g in a.var.index if g in gidx]
    if len(common) < 1000:
        for cn in ('gene_name', 'gene_symbol'):
            if cn in a.var.columns:
                a.var_names = a.var[cn].astype(str).to_numpy()
                a.var_names_make_unique()
                common = [g for g in a.var.index if g in gidx]
                if len(common) >= 1000:
                    break
    if len(common) < 1000 and 'ensembl_id' in a.var.columns and os.path.exists(f'{PERT_DIR}/ensembl2symbol.tsv'):
        m = {}
        with open(f'{PERT_DIR}/ensembl2symbol.tsv') as fh:
            for line in fh:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 2:
                    m[parts[0]] = parts[1]
        a.var_names = np.array([m.get(str(e), '') for e in a.var['ensembl_id'].astype(str)])
        a.var_names_make_unique()
        common = [g for g in a.var.index if g in gidx]
    print(f"  {os.path.basename(path)}: 面板重合 {len(common)}/{a.n_vars} 基因", flush=True)
    a = a[:, common]
    if max_cells and a.n_obs > max_cells:
        pert_vals = a.obs[obs_col].astype(str).to_numpy()
        srs = pd.Series(np.arange(a.n_obs), index=pert_vals)
        keep = srs.groupby(level=0, sort=False).apply(
            lambda g: g.to_numpy() if (g.name == ctrl_label or len(g) <= per_pert)
            else g.sample(per_pert, random_state=seed).to_numpy())
        keep = np.concatenate([np.atleast_1d(x) for x in keep])
        if len(keep) > max_cells:
            keep = np.random.default_rng(seed).choice(keep, max_cells, replace=False)
        print(f"  {os.path.basename(path)}: sampled keep {len(keep)}/{a.n_obs} cells", flush=True)
        a = a[keep, :]
    col = np.array([gidx[g] for g in common])
    tok = np.zeros((a.n_obs, G), np.int8)
    BLK = 50000
    for i0 in range(0, a.n_obs, BLK):
        i1 = min(i0 + BLK, a.n_obs)
        Xb = a.X[i0:i1]
        if not isinstance(Xb, np.ndarray):
            Xb = Xb.toarray()
        Xb = np.log1p(Xb / Xb.sum(1, keepdims=True).clip(1) * 1e4).astype(np.float32)
        tok[i0:i1, col] = np.searchsorted(edges, Xb).astype(np.int8)
        del Xb
    pert = a.obs[obs_col].astype(str).to_numpy()
    pert[pert == ctrl_label] = 'control'
    del a
    gc.collect()
    print(f"  {os.path.basename(path)}: {tok.shape[0]} 细胞, {len(set(pert))-1} 扰动", flush=True)
    np.savez(cache, tok=tok, pert=pert)
    return tok, pert

PERT_FILES = [
    # Tier3: 整个 GSE264667 不进训练 — HepG2(肝源)+Jurkat 双 LOCO 独立验证
    # ("GSE264667_hepg2_raw_singlecell_01.h5ad", "gene", "non-targeting", "HepG2"),
    # ("GSE264667_jurkat_raw_singlecell_01.h5ad", "gene", "non-targeting", "Jurkat"),
    ("ReplogleWeissman2022_K562_gwps_400k.h5ad", "perturbation", "control", "K562-gwps"),
    ("ReplogleWeissman2022_K562_essential.h5ad", "perturbation", "control", "K562-essential"),
    ("ReplogleWeissman2022_rpe1.h5ad", "perturbation", "control", "RPE1"),
    ("McFarlandTsherniak2020.h5ad", "perturbation", "control", "CCLE"),
    ("NormanWeissman2019_filtered.h5ad", "perturbation", "control", "Norman"),
    ("SrivatsanTrapnell2020_sciplex2.h5ad", "perturbation", "control", "sciplex2"),
    ("SrivatsanTrapnell2020_sciplex3.h5ad", "perturbation", "control", "sciplex3"),
    # v4 新增: 内皮细胞 Perturb-seq (GSE210523, Schnitzler 2024) — 降低 K562 占比, 补充血管/内皮上下文
    ("Schnitzler2024_teloHAEC_perturbseq.h5ad", "perturbation", "control", "teloHAEC"),
]
# 低内存两遍法: 先读缓存 npy 头拿行数, 再逐文件载入-拷入-立即释放
import zipfile as _zf
def _tokcache_rows(path):
    cache = path + ".tokcache.npz"
    if os.path.exists(cache):
        with _zf.ZipFile(cache) as zf:
            with zf.open("tok.npy") as f:
                f.read(6); f.read(2)
                hlen = int.from_bytes(f.read(2), "little")
                return eval(f.read(hlen).decode("latin1"))["shape"][0]
    return None

PT = []
total = 0
for fn, oc, cl, tag in PERT_FILES:
    p = f"{PERT_DIR}/{fn}"
    if not os.path.exists(p):
        continue
    nr = _tokcache_rows(p)
    if nr is None:
        try:
            t, pp = load_perturb_h5ad(p, obs_col=oc, ctrl_label=cl)
            PT.append([t, pp, tag]); nr = t.shape[0]
        except Exception as e:
            print(f"  跳过 {fn}: {e}", flush=True)
            continue
    else:
        PT.append([p, oc, cl, tag])
        print(f"  {fn}: 缓存 {nr} 细胞 (延迟加载)", flush=True)
    total += nr
assert PT and total > 0, "无扰动数据"
PTOK = np.empty((total, G), np.int8)
PPERT = np.empty(total, dtype=object)
PCL = np.empty(total, dtype=object)
_off = 0
for _i, item in enumerate(PT):
    if isinstance(item[0], str):
        t, pp = load_perturb_h5ad(item[0], obs_col=item[1], ctrl_label=item[2])
        tag = item[3]
    else:
        t, pp, tag = item
    _n = t.shape[0]
    PTOK[_off:_off + _n] = t
    PPERT[_off:_off + _n] = pp
    PCL[_off:_off + _n] = tag
    _off += _n
    PT[_i] = None
    del t, pp
    gc.collect()
del PT
gc.collect()
print(f"扰动语料合计 {len(PPERT)} 细胞 (Tier3 排除后)", flush=True)

# ---------- Tier1+Tier2+Tier4+Tier5 排除 ----------
tier1_set = set(TIER1_40)
m_gwps_nc = (PCL == "K562-gwps") & (PPERT != 'control')
gwps_vc = pd.Series(PPERT[m_gwps_nc]).value_counts()
gwps_eligible = sorted([p for p, c in gwps_vc.items() if c >= 30 and p in gidx and p not in tier1_set])
rng_t2 = np.random.default_rng(TIER2_SEED)
TIER2 = sorted(rng_t2.choice(gwps_eligible, size=min(N_TIER2, len(gwps_eligible)), replace=False).tolist())
# Tier4 留出: K562-essential × RPE1 共享基因抽 100 个 (seed=2027), 全部排除出训练
# → 跨系一致性评估成为完全留出的泛化测试, 而非训练集内一致性
m_ess = (PCL == "K562-essential") & (PPERT != 'control')
m_rpe = (PCL == "RPE1") & (PPERT != 'control')
ess_vc = pd.Series(PPERT[m_ess]).value_counts()
rpe_vc = pd.Series(PPERT[m_rpe]).value_counts()
t4_pool = sorted((set(ess_vc[ess_vc >= 30].index) & set(rpe_vc[rpe_vc >= 30].index)
                  & set(gidx)) - tier1_set - set(TIER2))
rng_t4 = np.random.default_rng(2027)
TIER4_HO = sorted(rng_t4.choice(t4_pool, size=min(100, len(t4_pool)), replace=False).tolist()) \
    if len(t4_pool) else []
# Tier5 留出: 库外化合物演示的靶基因, 保证 target-annotation 通路演示为真零样本
TIER5_DEMO = sorted(g for g in ["THRB", "HMGCR", "NR1H4", "PPARA", "PPARG", "NR1H3",
                                "MTOR", "PIK3CA", "SREBF1", "SREBF2"] if g in gidx)
EXCLUDE = tier1_set | set(TIER2) | set(TIER4_HO) | set(TIER5_DEMO)
m_keep = np.array([p not in EXCLUDE for p in PPERT])
n_drop = int((~m_keep).sum())
# 低内存就地紧凑化: 分块前移, 避免布尔索引产生全量副本
_keep_pos = np.where(m_keep)[0]
_dst = 0
for _c0 in range(0, len(_keep_pos), 100000):
    _cs = _keep_pos[_c0:_c0 + 100000]
    PTOK[_dst:_dst + len(_cs)] = PTOK[_cs]
    _dst += len(_cs)
PTOK = PTOK[:_dst]
PCL = PCL[_keep_pos]
PPERT_EXCL = PPERT[~m_keep]; PPERT = PPERT[_keep_pos]
del m_keep, _keep_pos
gc.collect()
print(f"Tier1 排除 {len(tier1_set)} 基因 | Tier2 排除 {len(TIER2)} 基因 (seed={TIER2_SEED}) | "
      f"Tier4 排除 {len(TIER4_HO)} 跨系共享基因 | Tier5 排除 {len(TIER5_DEMO)} 演示靶基因 | "
      f"剔除细胞 {n_drop}", flush=True)
json.dump({"tier1_40": TIER1_40, "tier2_150": TIER2, "tier2_seed": TIER2_SEED,
           "tier3_excluded_files": ["GSE264667_hepg2_raw_singlecell_01.h5ad",
                                    "GSE264667_jurkat_raw_singlecell_01.h5ad"],
           "tier4_100": TIER4_HO, "tier5_demo_targets": TIER5_DEMO,
           "dropped_cells": n_drop,
           "sf9_donor_holdout": _sf9["donors"]},
          open(f"{CKPT}/holdout_manifest.json", "w"), indent=1)

# ---------- top-DE 表: 按细胞系 tag 分组 ----------
_lut_np = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)
def de_genes_grouped(tok, pert, pcl, topn=DE_TOPN):
    lut = _lut_np
    de = {}
    key = np.array([f"{c}||{p}" for c, p in zip(pcl, pert)])
    order = np.argsort(key, kind='stable')
    ks = key[order]
    uq, st = np.unique(ks, return_index=True)
    ends = np.r_[st[1:], len(order)]
    ctrl_mean = {}
    for u, s, e in zip(uq, st, ends):
        tag, p = u.split('||', 1)
        if p == 'control' and e - s >= 30:
            ctrl_mean[tag] = lut[tok[order[s:e]].astype(int)].mean(0)
    for u, s, e in zip(uq, st, ends):
        tag, p = u.split('||', 1)
        if p == 'control' or tag not in ctrl_mean or e - s < 30:
            continue
        Ep = lut[tok[order[s:e]].astype(int)].mean(0)
        d = Ep - ctrl_mean[tag]
        top = np.argsort(-np.abs(d))[:topn]
        de[(tag, p)] = (top, d[top].astype(np.float32))
    return de
DE = de_genes_grouped(PTOK, PPERT, PCL)
print(f"DE 表: {len(DE)} 个 (tag,扰动) 组合", flush=True)

# ---------- 词表 ----------
ct_vocab = {t: i for i, t in enumerate(sorted(set(meta.celltype_v5.astype(str).unique()) | set(PCL)))}
cond_vocab = {t: i for i, t in enumerate(sorted(set(meta.condition.unique()) - {''}))}
pert_list = sorted(set(PPERT) - {'control'})
pert_vocab = {p: i for i, p in enumerate(pert_list)}
N_CT, N_COND, N_PERT = len(ct_vocab) + 1, len(cond_vocab) + 1, len(pert_vocab) + 1
NULL_CT, NULL_COND, NULL_PERT = len(ct_vocab), len(cond_vocab), len(pert_vocab)

SPECIES_VOCAB = {"human": 0, "mouse": 1, "rat": 2, "cellline": 3}
N_SP = len(SPECIES_VOCAB) + 1
NULL_SP = len(SPECIES_VOCAB)
if HAS_SPECIES:
    sp_bone = meta.species.map(SPECIES_VOCAB).fillna(NULL_SP).astype(int).to_numpy()
else:
    sp_bone = np.full(len(meta), NULL_SP, dtype=int)

ct_bone = meta.celltype_v5.astype(str).map(ct_vocab).fillna(NULL_CT).astype(int).to_numpy()
cond_bone = meta.condition.map(cond_vocab).fillna(NULL_COND).astype(int).to_numpy()
pert_cl = pd.Series(PCL).map(ct_vocab).astype(int).to_numpy()
pert_pid = pd.Series(PPERT).map(pert_vocab).fillna(NULL_PERT).astype(int).to_numpy()
pert_ptype = np.array([PTYPE_VOCAB.get(PERT_TYPE.get(t, "LOF"), 0) for t in PCL], dtype=int)
pert_sp = np.full(len(PCL), SPECIES_VOCAB["cellline"], dtype=int)

pert_gene_idx = np.full(N_PERT, -1, dtype=np.int64)
n_gene_pert = 0
for p, i in pert_vocab.items():
    if p in gidx:
        pert_gene_idx[i] = gidx[p]
        n_gene_pert += 1
print(f"扰动中 {n_gene_pert}/{len(pert_vocab)} 为面板基因", flush=True)
pert_gid_all = pert_gene_idx[pert_pid]

_key = np.array([f"{c}||{p}" for c, p in zip(PCL, PPERT)])
_order = np.argsort(_key, kind='stable')
_ks = _key[_order]
_uq, _st = np.unique(_ks, return_index=True)
_bounds = {u: (s, e) for u, s, e in zip(_uq, _st, np.r_[_st[1:], len(_order)])}
def cells_of_tag(tag, p):
    s, e = _bounds[f"{tag}||{p}"]
    return _order[s:e]

ctrl_idx_by_tag = {}
for tag in pd.unique(PCL):
    ctrl_idx_by_tag[tag] = np.where((PCL == tag) & (PPERT == 'control'))[0]

DE_PERTS_BY_TAG = {}
for (tag, p) in DE.keys():
    DE_PERTS_BY_TAG.setdefault(tag, []).append(p)
TAGS = [t for t in DE_PERTS_BY_TAG if len(DE_PERTS_BY_TAG[t]) >= 1]
print(f"可分组采样 tag: {[(t, len(DE_PERTS_BY_TAG[t])) for t in TAGS]}", flush=True)

class PMDDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gene_emb = nn.Embedding(52, D_MODEL)
        self.compress = nn.Linear(CHUNK * D_MODEL, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, N_CHUNK + 5, D_MODEL) * 0.02)
        self.ct_emb = nn.Embedding(N_CT, D_MODEL)
        self.cond_emb = nn.Embedding(N_COND, D_MODEL)
        self.sp_emb = nn.Embedding(N_SP, D_MODEL)            # v4 新增: species
        self.ptype_emb = nn.Embedding(N_PTYPE, GID_DIM)      # v4 新增: LOF/GOF/compound
        self.gene_id_emb = nn.Embedding(G, GID_DIM)
        self.pert_gene_mlp = nn.Sequential(nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
        self.pert_free_emb = nn.Embedding(N_PERT, GID_DIM)
        self.pert_free_mlp = nn.Sequential(nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
        self.delta_head = nn.Sequential(nn.Linear(D_MODEL * 3, 512), nn.GELU(),
                                        nn.Linear(512, 512), nn.GELU(), nn.Linear(512, G))
        self.z0_proj = nn.Linear(D_MODEL, D_MODEL)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL * 4, batch_first=True,
                                       norm_first=True, activation="gelu"), N_LAYER)
        self.decompress = nn.Linear(D_MODEL, CHUNK * 52)
        print(f"参数量: {sum(p.numel() for p in self.parameters())/1e6:.1f} M | "
              f"ct={N_CT} cond={N_COND} pert={N_PERT} sp={N_SP}", flush=True)

    def z0_token(self, x0):
        e = self.gene_emb(x0.long()).view(x0.shape[0], N_CHUNK, CHUNK * D_MODEL)
        return self.z0_proj(self.compress(e).mean(1))

    def pert_vector(self, pert, pgid, ptype):
        v = torch.zeros(pert.shape[0], D_MODEL, device=pert.device)
        is_null = pert.eq(NULL_PERT)
        pt = self.ptype_emb(ptype.clamp(0, N_PTYPE - 1))
        gm = (pgid >= 0) & ~is_null
        if gm.any():
            v[gm] = self.pert_gene_mlp(self.gene_id_emb(pgid[gm]) + pt[gm])
        fm = (pgid < 0) & ~is_null
        if fm.any():
            v[fm] = self.pert_free_mlp(self.pert_free_emb(pert[fm]) + pt[fm])
        return v

    def forward(self, x, ct, cond, pert, z0, pgid, ptype, sp):
        B = x.shape[0]
        e = self.gene_emb(x.long()).view(B, N_CHUNK, CHUNK * D_MODEL)
        z = self.compress(e)
        pv = self.pert_vector(pert, pgid, ptype)
        head = torch.stack([self.ct_emb(ct), self.cond_emb(cond), pv, z0, self.sp_emb(sp)], 1)
        z = torch.cat([head, z], 1) + self.pos_emb
        z = self.encoder(z)[:, 5:]
        return self.decompress(z).view(B, G_PAD, 52)

model = PMDDM().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05)

_ck = sorted(glob.glob(f"{CKPT}/pmdm_v4_step*.pt"), key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
START_STEP = 0
if _ck:
    ck = torch.load(_ck[-1], map_location=DEV, weights_only=False)
    model.load_state_dict(ck["model"]); START_STEP = ck["step"]
    for _ in range(START_STEP): sched.step()
    print(f"恢复: {_ck[-1]} step {START_STEP}", flush=True)

def fetch_bone(bs):
    bi = keep_idx_train[np.random.randint(0, len(keep_idx_train), bs)]
    x = np.zeros((bs, G), np.int8)
    for i, r in enumerate(bi):
        s, e = int(mm_indptr[r]), int(mm_indptr[r+1])
        if e > s: x[i, mm_index[s:e].astype(np.int64)] = mm_data[s:e]
    return bi, x

def fetch_pert(bs):
    tag = TAGS[np.random.randint(len(TAGS))]
    dps = DE_PERTS_BY_TAG[tag]
    k = min(P_PER_BATCH, len(dps))
    ps = np.random.choice(dps, k, replace=len(dps) < k)
    per = bs // k
    idxs = []
    for p in ps:
        pool = cells_of_tag(tag, p)
        idxs.append(pool[np.random.randint(0, len(pool), per)])
    hi = np.concatenate(idxs)
    if len(hi) < bs:
        hi = np.concatenate([hi, idxs[0][np.random.randint(0, len(idxs[0]), bs - len(hi))]])
    hi = hi[:bs]
    x1 = PTOK[hi]
    ci = ctrl_idx_by_tag[tag]
    x0 = PTOK[np.random.choice(ci, bs, replace=len(ci) < bs)]
    return hi, x1, x0, tag

def mask_batch(x, force_full=False):
    B, NC = x.shape
    xm = x.copy()
    for i in range(B):
        nz = x[i] > 0
        if force_full:
            xm[i] = MASK_TOKEN
            continue
        u = np.random.rand()
        if u < P_FULL:
            m = np.ones(NC, bool)
        elif u < P_FULL + P_TRAJ:
            m = np.zeros(NC, bool); m[nz] = True
            keep = np.random.rand(NC) < 0.3; m &= ~keep
        else:
            t = np.random.rand()
            m = np.random.rand(NC) < t
            zm = ~nz & (np.random.rand(NC) < t * ZERO_FRAC)
            m |= zm
        xm[i, m] = MASK_TOKEN
    return torch.from_numpy(xm)

de_gene_sets = {}
de_gene_vals = {}
for (tag, p), (idx, vals) in DE.items():
    if p in pert_vocab:
        pid = pert_vocab[p]
        de_gene_sets[(tag, pid)] = torch.tensor(idx, dtype=torch.long)
        de_gene_vals[(tag, pid)] = (torch.tensor(idx, dtype=torch.long, device=DEV),
                                    torch.tensor(vals, dtype=torch.float32, device=DEV))
lut_t = torch.zeros(52, device=DEV); lut_t[:len(_lut_np)] = torch.tensor(_lut_np[:52], device=DEV)

# ---------- 固定验证批: 384 个留出细胞 + 固定掩码, 每1000步算 val_loss ----------
_vr = np.random.default_rng(2026).choice(val_pool, min(384, len(val_pool)), replace=False)
_vx = np.zeros((len(_vr), G), np.int8)
for _i, _r in enumerate(_vr):
    _s, _e = int(mm_indptr[_r]), int(mm_indptr[_r + 1])
    if _e > _s:
        _vx[_i, mm_index[_s:_e].astype(np.int64)] = mm_data[_s:_e]
VAL_X = torch.from_numpy(np.pad(_vx, ((0, 0), (0, G_PAD - G)))).to(DEV)
VAL_CT = torch.from_numpy(ct_bone[_vr]).to(DEV)
VAL_COND = torch.from_numpy(cond_bone[_vr]).to(DEV)
VAL_SP = torch.from_numpy(sp_bone[_vr]).to(DEV)
_mrng = np.random.default_rng(97)
VAL_MASK = torch.from_numpy(_mrng.random(VAL_X.shape) < 0.5).to(DEV)
VAL_XM = VAL_X.clone()
VAL_XM[VAL_MASK] = MASK_TOKEN
del _vx

print(f"P-MDDM v4 冷启动: {STEPS} 步 batch={BATCH} | Tier1=40 Tier2={len(TIER2)} Tier3=HepG2 | "
      f"ptype+species 嵌入", flush=True)
t0 = time.time(); losses, losses_p, losses_dh, losses_dg = [], [], [], []
for step in range(START_STEP + 1, STEPS + 1):
    is_pert = np.random.rand() < PERT_FRAC and len(PTOK) > 0
    force_full = False
    if is_pert:
        hi, x1, x0, tag = fetch_pert(BATCH)
        ct = torch.from_numpy(pert_cl[hi]).to(DEV)
        cond = torch.full((BATCH,), NULL_COND, device=DEV)
        pert = torch.from_numpy(pert_pid[hi]).to(DEV)
        pgid = torch.from_numpy(pert_gid_all[hi]).to(DEV)
        ptype = torch.from_numpy(pert_ptype[hi]).to(DEV)
        sp = torch.from_numpy(pert_sp[hi]).to(DEV)
        x0t = torch.from_numpy(np.pad(x0, ((0, 0), (0, G_PAD - G)))).to(DEV)
        with torch.no_grad():
            z0 = model.z0_token(x0t)
        xt = np.pad(x1, ((0, 0), (0, G_PAD - G)))
        force_full = np.random.rand() < P_FULLMASK_PERT
        xm = mask_batch(xt, force_full=force_full).to(DEV)
        target = torch.from_numpy(xt).to(DEV)
        ctrl_ref = lut_t[torch.from_numpy(x0).long().to(DEV)].mean(0)
    else:
        bi, x = fetch_bone(BATCH)
        ct = torch.from_numpy(ct_bone[bi]).to(DEV)
        cd = torch.from_numpy(cond_bone[bi]).to(DEV)
        cond = torch.where(torch.rand(BATCH, device=DEV) < P_DROP, torch.full_like(cd, NULL_COND), cd)
        pert = torch.full((BATCH,), NULL_PERT, device=DEV)
        pgid = torch.full((BATCH,), -1, dtype=torch.int64, device=DEV)
        ptype = torch.full((BATCH,), 3, dtype=torch.int64, device=DEV)   # NULL ptype
        sp = torch.from_numpy(sp_bone[bi]).to(DEV)
        z0 = torch.zeros(BATCH, D_MODEL, device=DEV)
        xp = np.pad(x, ((0, 0), (0, G_PAD - G)))
        xm = mask_batch(xp).to(DEV)
        target = torch.from_numpy(xp).to(DEV)
    logits = model(xm, ct, cond, pert, z0, pgid, ptype, sp)
    tgt = target.long()
    ce = F.cross_entropy(logits.view(-1, 52), tgt.view(-1), reduction='none').view(BATCH, G_PAD)
    masked = xm.eq(MASK_TOKEN)
    w = torch.ones_like(ce)
    w = torch.where(tgt > 0, torch.tensor(NZ_W, device=DEV), w)
    if is_pert:
        for b in range(BATCH):
            key = (tag, int(pert[b].item()))
            if key in de_gene_sets:
                w[b, de_gene_sets[key].to(DEV)] *= DE_W
    l = (ce * w * masked).sum() / (w * masked).sum().clamp(1)
    if is_pert:
        pv = model.pert_vector(pert, pgid, ptype)
        dh = model.delta_head(torch.cat([pv, model.ct_emb(ct), z0], -1))
        dl, nd = 0.0, 0
        for pid in torch.unique(pert).tolist():
            if pid == NULL_PERT:
                continue
            key = (tag, pid)
            if key not in de_gene_vals:
                continue
            m = pert == pid
            idx_t, val_t = de_gene_vals[key]
            pred_d = dh[m].mean(0)[idx_t]
            dl = dl + (1 - F.cosine_similarity(pred_d, val_t, dim=0, eps=1e-6)) \
                    + 0.1 * F.mse_loss(pred_d, val_t)
            nd += 1
        if nd > 0:
            ldh = dl / nd
            l = l + W_DELTA_HEAD * ldh
            losses_dh.append(float(ldh.detach()))
        if force_full:
            probs = F.softmax(logits.view(BATCH, G_PAD, 52).float(), -1)
            expr_hat = (probs @ lut_t)[:, :G]
            dg, ng = 0.0, 0
            for pid in torch.unique(pert).tolist():
                if pid == NULL_PERT:
                    continue
                key = (tag, pid)
                if key not in de_gene_vals:
                    continue
                m = pert == pid
                idx_t, val_t = de_gene_vals[key]
                pred_d = expr_hat[m].mean(0)[idx_t] - ctrl_ref[idx_t]
                dg = dg + (1 - F.cosine_similarity(pred_d, val_t, dim=0, eps=1e-6))
                ng += 1
            if ng > 0:
                ldg = dg / ng
                l = l + W_DELTA_GEN * ldg
                losses_dg.append(float(ldg.detach()))
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    (losses_p if is_pert else losses).append(l.item())
    if step % 1000 == 0:
        try:
            model.eval()
            with torch.no_grad():
                _vls = []
                for _i0 in range(0, VAL_X.shape[0], 128):
                    _sl = slice(_i0, min(_i0 + 128, VAL_X.shape[0]))
                    _b = _sl.stop - _sl.start
                    _vl = model(VAL_XM[_sl], VAL_CT[_sl], VAL_COND[_sl],
                                torch.full((_b,), NULL_PERT, device=DEV),
                                torch.zeros(_b, D_MODEL, device=DEV),
                                torch.full((_b,), -1, dtype=torch.int64, device=DEV),
                                torch.full((_b,), 3, dtype=torch.int64, device=DEV),
                                VAL_SP[_sl])
                    _vce = F.cross_entropy(_vl.view(-1, 52), VAL_X[_sl].long().view(-1),
                                           reduction='none').view(_b, G_PAD)
                    _vls.append(float((_vce * VAL_MASK[_sl]).sum() / VAL_MASK[_sl].sum().clamp(1)))
                _vloss = float(np.mean(_vls))
            model.train()
        except Exception as _e:
            _vloss = float('nan')
            print(f"  val 跳过: {_e}", flush=True)
            model.train()
        el = (time.time() - t0) / 60
        print(f"step {step:6d} | organ {np.mean(losses[-500:]) if losses else float('nan'):.4f} | "
              f"pert {np.mean(losses_p[-500:]) if losses_p else float('nan'):.4f} | "
              f"dhead {np.mean(losses_dh[-200:]) if losses_dh else float('nan'):.4f} | "
              f"dgen {np.mean(losses_dg[-200:]) if losses_dg else float('nan'):.4f} | "
              f"val {_vloss:.4f} | "
              f"{el:.0f} min | ETA {(STEPS-step)/max(step-START_STEP,1)*el:.0f} min", flush=True)
    if step % SAVE_EVERY == 0:
        torch.save({"model": model.state_dict(), "step": step, "ct_vocab": ct_vocab,
                    "cond_vocab": cond_vocab, "pert_vocab": pert_vocab,
                    "species_vocab": SPECIES_VOCAB, "ptype_vocab": PTYPE_VOCAB,
                    "holdout": {"tier1_40": TIER1_40, "tier2": TIER2, "tier3": "HepG2"}},
                   f"{CKPT}/pmdm_v4_step{step}.pt")
        print(f"  保存 step {step}", flush=True)
print("PMDDM_V4_DONE", flush=True)

