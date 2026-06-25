import sys, numpy as np, pandas as pd, openpyxl
sys.path.insert(0, '/home/user/users/shubhom/closed_loop_stim/patterns_5k')
from utils import read_pattern_json, preprocess_pattern_stimulations_df

ROOT = '/home/user/users/shubhom/closed_loop_stim/vis_stim'
d1dir = f'{ROOT}/data/icms_150_6_2_26'
d2dir = f'{ROOT}/data/icms_150_6_3_26'
OUTDIR = f'{ROOT}/data/exports'
import os; os.makedirs(OUTDIR, exist_ok=True)

# --- timing constants (match notebook) ---
SAMPLE_RATE   = 30_000
SAMPLES_PER_MS = SAMPLE_RATE // 1000
STIM_MS        = 500          # visual window
VIS_ORIENT_TIME = 600        # electrical window
BIN_MS         = 10
BIN_SAMPLES    = BIN_MS * SAMPLES_PER_MS

# ============ load spikes ============
def load_spikes(dd):
    a = np.load(f'{dd}/All_Shank_Spk_Vecs.npy', allow_pickle=False)
    df = pd.DataFrame(a).rename(columns={'sample_index':'timestamp','unit_index':'neuron_id'})
    return df.drop(columns=['segment_index'])
spk1, spk2 = load_spikes(d1dir), load_spikes(d2dir)

# ============ registration -> common units ============
wb = openpyxl.load_workbook(f'{d2dir}/unit_registration_results.xlsx')
ws = wb[wb.sheetnames[0]]
rows = list(ws.iter_rows(min_row=2, values_only=True))
matched = [(int(a),int(b)) for a,b in rows if a is not None and b is not None]
common = pd.DataFrame(matched, columns=['day1_unit','day2_unit'])
common.insert(0,'common_idx', np.arange(len(common)))
N_COMMON = len(common)
day1_to_common = dict(zip(common['day1_unit'], common['common_idx']))
day2_to_common = dict(zip(common['day2_unit'], common['common_idx']))
print(f'N_COMMON = {N_COMMON}')

# ============ visual trials ============
def visual_trials(dd):
    ts = np.load(f'{dd}/VStim_Onset_TS.npy') * 30
    lab = np.load(f'{dd}/VStim_Labels.npy')
    return pd.DataFrame({'timestamp': ts, 'orientation': lab})
vt1, vt2 = visual_trials(d1dir), visual_trials(d2dir)

# ============ pattern (electrical) ============
pat1,_ = preprocess_pattern_stimulations_df(read_pattern_json(f'{d1dir}/Combined_Pattern_Registrations.pkl'), align_to_stim=True)
pat2,_ = preprocess_pattern_stimulations_df(read_pattern_json(f'{d2dir}/Combined_Pattern_Registrations.pkl'), align_to_stim=True)
xref = pd.read_csv(f'{d1dir}/generated_patterns_day2_6_3_26.csv'); xref['pattern_name']=xref['pattern_name'].astype(int)
xref_meta = xref.set_index('pattern_name')[['target_orientation_deg','source_budget','decomp_method','ori_idx']]
xref_names = set(xref['pattern_name'])

# ---- binned-response helper: (n_trials, N_COMMON, n_bins), trials ordered by onset appearance ----
def binned_by_trial(spikes_df, unit_to_common, onsets, n_bins, win_len):
    """For each onset (in given order), bin spikes of common units over [t0,t0+win_len)."""
    # restrict to common units, remap neuron_id -> common_idx
    s = spikes_df[spikes_df['neuron_id'].isin(unit_to_common)].copy()
    s['cidx'] = s['neuron_id'].map(unit_to_common).astype(int)
    ts = s['timestamp'].values; cidx = s['cidx'].values
    order = np.argsort(ts); ts = ts[order]; cidx = cidx[order]
    out = np.zeros((len(onsets), N_COMMON, n_bins), dtype=np.int32)
    for i, t0 in enumerate(onsets):
        lo = np.searchsorted(ts, t0, 'left'); hi = np.searchsorted(ts, t0+win_len, 'left')
        if hi > lo:
            rel = (ts[lo:hi] - t0) // BIN_SAMPLES
            ci  = cidx[lo:hi]
            m = (rel >= 0) & (rel < n_bins)
            np.add.at(out, (i, ci[m], rel[m].astype(int)), 1)
    return out

# ===================== VISUAL exports =====================
def export_visual(tag, vt, spikes_df, unit_to_common):
    vt = vt.reset_index(drop=True)             # appearance order = file order
    onsets = vt['timestamp'].values.astype(np.int64)
    n_bins = STIM_MS // BIN_MS
    resp = binned_by_trial(spikes_df, unit_to_common, onsets, n_bins, STIM_MS*SAMPLES_PER_MS)
    # trial-within-orientation index, in appearance order
    trial_in_ori = vt.groupby('orientation').cumcount().values
    fn = f'{OUTDIR}/visual_{tag}_registered.npz'
    np.savez_compressed(
        fn,
        responses=resp,                                   # (n_trials, N_COMMON, n_bins)
        orientation=vt['orientation'].values,             # (n_trials,)
        trial_in_orientation=trial_in_ori,                # (n_trials,)
        onset_timestamp=onsets,                           # (n_trials,)
        common_idx=common['common_idx'].values,
        day1_unit=common['day1_unit'].values,
        day2_unit=common['day2_unit'].values,
        bin_ms=BIN_MS, window_ms=STIM_MS, sample_rate=SAMPLE_RATE)
    return fn, resp, vt, trial_in_ori

# ===================== ELECTRICAL exports =====================
def export_electrical(tag, pat, spikes_df, unit_to_common):
    deliv = (pat[['pattern_name','pattern_flag_start_timestamp','trial','is_oracle']]
             .drop_duplicates(subset=['pattern_name','pattern_flag_start_timestamp']))
    deliv['pattern_name'] = deliv['pattern_name'].astype(int)
    # appearance order = onset timestamp ascending
    deliv = deliv.sort_values('pattern_flag_start_timestamp').reset_index(drop=True)
    onsets = deliv['pattern_flag_start_timestamp'].values.astype(np.int64)
    n_bins = VIS_ORIENT_TIME // BIN_MS
    resp = binned_by_trial(spikes_df, unit_to_common, onsets, n_bins, VIS_ORIENT_TIME*SAMPLES_PER_MS)

    pn = deliv['pattern_name'].values
    is_gen = np.array([p in xref_names for p in pn])
    is_orc = deliv['is_oracle'].values.astype(bool)
    condition = np.where(is_gen, 'generated', np.where(is_orc, 'oracle', 'random'))
    # generated metadata (NaN where not generated)
    tgt_ori = np.full(len(pn), np.nan); budget = np.full(len(pn), np.nan)
    ori_idx = np.full(len(pn), np.nan); decomp = np.array(['']*len(pn), dtype=object)
    for i,p in enumerate(pn):
        if p in xref_meta.index:
            r = xref_meta.loc[p]
            tgt_ori[i]=r['target_orientation_deg']; budget[i]=r['source_budget']
            ori_idx[i]=r['ori_idx']; decomp[i]=r['decomp_method']
    fn = f'{OUTDIR}/electrical_{tag}_registered.npz'
    np.savez_compressed(
        fn,
        responses=resp,                                   # (n_trials, N_COMMON, n_bins)
        pattern_name=pn,
        condition=condition.astype(str),                  # oracle/random/generated
        trial=deliv['trial'].values,                      # within-pattern trial index
        target_orientation_deg=tgt_ori,                   # generated only, else NaN
        source_budget=budget, ori_idx=ori_idx,
        decomp_method=decomp.astype(str),
        onset_timestamp=onsets,
        common_idx=common['common_idx'].values,
        day1_unit=common['day1_unit'].values,
        day2_unit=common['day2_unit'].values,
        bin_ms=BIN_MS, window_ms=VIS_ORIENT_TIME, sample_rate=SAMPLE_RATE)
    return fn, resp, deliv, condition, tgt_ori

# ---------- run ----------
vfn1, vr1, vt1o, vti1 = export_visual('day1', vt1, spk1, day1_to_common)
vfn2, vr2, vt2o, vti2 = export_visual('day2', vt2, spk2, day2_to_common)
efn1, er1, ed1, ec1, et1 = export_electrical('day1', pat1, spk1, day1_to_common)
efn2, er2, ed2, ec2, et2 = export_electrical('day2', pat2, spk2, day2_to_common)

print('\n=== files written ===')
for fn,r in [(vfn1,vr1),(vfn2,vr2),(efn1,er1),(efn2,er2)]:
    print(f'  {fn}  responses={r.shape}')

# =================== SANITY CHECKS ===================
print('\n=== SANITY CHECKS ===')
ok = True
def chk(cond, msg):
    global ok
    print(('  [OK] ' if cond else '  [FAIL] ')+msg); ok &= bool(cond)

# 1. unit axis = N_COMMON for all
for r in [vr1,vr2,er1,er2]:
    chk(r.shape[1]==N_COMMON, f'unit axis == {N_COMMON}')

# 2. visual: trial count matches onsets, orientations match labels, trial_in_ori correct
chk(vr1.shape[0]==len(vt1o), f'visual day1 trials {vr1.shape[0]} == onsets {len(vt1o)}')
chk(vr2.shape[0]==len(vt2o), f'visual day2 trials {vr2.shape[0]} == onsets {len(vt2o)}')
# trial_in_orientation: per orientation should be 0..count-1 contiguous
for tag,vt,vti in [('day1',vt1o,vti1),('day2',vt2o,vti2)]:
    g = pd.DataFrame({'o':vt['orientation'].values,'t':vti})
    bad = g.groupby('o')['t'].apply(lambda s: not np.array_equal(np.sort(s.values), np.arange(len(s))))
    chk(not bad.any(), f'visual {tag} trial_in_orientation is 0..n-1 per orientation')

# 3. electrical: condition labels coverage + generated metadata correctness
for tag,ed,ec,et in [('day1',ed1,ec1,et1),('day2',ed2,ec2,et2)]:
    vals,counts = np.unique(ec, return_counts=True)
    print(f'    {tag} condition counts: {dict(zip(vals.tolist(),counts.tolist()))}')
    # every 'generated' row must have a target orientation; non-generated must be NaN
    gen_mask = ec=='generated'
    chk(np.all(~np.isnan(et[gen_mask])) if gen_mask.any() else True,
        f'electrical {tag}: all generated rows have target_orientation_deg')
    chk(np.all(np.isnan(et[~gen_mask])) if (~gen_mask).any() else True,
        f'electrical {tag}: non-generated rows have NaN target_orientation_deg')

# 4. day2 generated target orientation matches xref exactly (spot check via reload)
z = np.load(efn2, allow_pickle=True)
genm = z['condition']=='generated'
mism = 0
for p,t in zip(z['pattern_name'][genm], z['target_orientation_deg'][genm]):
    if not np.isclose(xref_meta.loc[int(p),'target_orientation_deg'], t): mism+=1
chk(mism==0, f'electrical day2: generated target_orientation matches xref (mismatches={mism})')

# 5. spike-count conservation: binned sum == raw count in window for a sampled trial
def raw_count(spikes_df, unit_to_common, t0, win):
    s = spikes_df[(spikes_df['timestamp']>=t0)&(spikes_df['timestamp']<t0+win)&(spikes_df['neuron_id'].isin(unit_to_common))]
    return len(s)
import random; random.seed(0)
for tag,r,onsets,spd,u2c,win in [
    ('vis_d1',vr1,vt1o['timestamp'].values,spk1,day1_to_common,STIM_MS*SAMPLES_PER_MS),
    ('elec_d2',er2,ed2['pattern_flag_start_timestamp'].values,spk2,day2_to_common,VIS_ORIENT_TIME*SAMPLES_PER_MS)]:
    idxs = random.sample(range(len(onsets)), min(5,len(onsets)))
    good=True
    for i in idxs:
        if int(r[i].sum()) != raw_count(spd,u2c,int(onsets[i]),win): good=False
    chk(good, f'{tag}: binned-sum == raw common-unit count for 5 sampled trials')

print('\nALL SANITY CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
