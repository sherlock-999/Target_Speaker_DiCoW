import csv, json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from local_text_norm.english import EnglishTextNormalizer

norm = EnglishTextNormalizer(standardize_numbers=False, standardize_numbers_rev=True, remove_fillers=True)

def clean(s):
    s = norm(str(s))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def build_rows(csv_path):
    ref_multi=[]; hyp_multi=[]; ref_wer=[]; hyp_wer=[]
    with open(csv_path, newline='', encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            mix=row['mix_id']
            spk=row.get('target_side','spk')
            ref=clean(row.get('ref_text',''))
            hyp=clean(row.get('hyp_text',''))
            ref_multi.append({'session_id':mix,'speaker':spk,'start_time':0.0,'end_time':0.0,'words':ref,'segment_index':0})
            hyp_multi.append({'session_id':mix,'speaker':spk,'start_time':0.0,'end_time':0.0,'words':hyp,'segment_index':0})
            ref_wer.append({'session_id':mix,'speaker':'single','start_time':0.0,'end_time':0.0,'words':ref,'segment_index':0})
            hyp_wer.append({'session_id':mix,'speaker':'single','start_time':0.0,'end_time':0.0,'words':hyp,'segment_index':0})
    return ref_multi,hyp_multi,ref_wer,hyp_wer

def agg(rows):
    from collections import defaultdict
    d=defaultdict(list)
    for x in rows: d[(x['session_id'],x['speaker'])].append(x)
    out=[]
    for (sid,spk),xs in d.items():
        txt=' '.join(i['words'] for i in xs).strip()
        out.append({'session_id':sid,'speaker':spk,'start_time':0,'end_time':0,'words':txt})
    return out

def run_meeteval(ref,hyp,mode,out):
    p=Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    cmd=[sys.executable,'-m','meeteval.wer',mode,'-r',str(ref),'-h',str(hyp),'--average-out',str(p)]
    subprocess.run(cmd,check=True)
    return json.loads(p.read_text())

base=Path('output/libri2mix_solospeech_whisper')
outbase=Path('output/libri2mix_solospeech_whisper/meeteval_norm')
summary={}
for name,file in [('non_streaming',base/'nonstreaming_predictions.csv'),('streaming',base/'streaming_predictions.csv')]:
    ref_multi,hyp_multi,ref_wer,hyp_wer = build_rows(file)
    d=outbase/name
    d.mkdir(parents=True, exist_ok=True)
    ref_wer_agg=agg(ref_wer); hyp_wer_agg=agg(hyp_wer)
    ref_multi_agg=agg(ref_multi); hyp_multi_agg=agg(hyp_multi)
    (d/'reference_wer_agg.seglst.json').write_text(json.dumps(ref_wer_agg),encoding='utf-8')
    (d/'hypothesis_wer_agg.seglst.json').write_text(json.dumps(hyp_wer_agg),encoding='utf-8')
    (d/'reference_multi_agg.seglst.json').write_text(json.dumps(ref_multi_agg),encoding='utf-8')
    (d/'hypothesis_multi_agg.seglst.json').write_text(json.dumps(hyp_multi_agg),encoding='utf-8')
    wer=run_meeteval(d/'reference_wer_agg.seglst.json',d/'hypothesis_wer_agg.seglst.json','wer',d/'wer_average.norm.json')
    cp=run_meeteval(d/'reference_multi_agg.seglst.json',d/'hypothesis_multi_agg.seglst.json','cpwer',d/'cpwer_average.norm.json')
    summary[name]={'wer':wer['error_rate'],'cpwer':cp['error_rate'],'wer_len':wer['length'],'cpwer_len':cp['length']}

(outbase/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
