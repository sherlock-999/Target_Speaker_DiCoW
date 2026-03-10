import json
from pathlib import Path
from collections import defaultdict
from score_testset_tse_from_existing_dicow import (
    build_dataset, load_hypothesis_map, map_nsf_session, map_l2m_session, read_jsonl_rows
)

testset_root = Path('/home3/adnan/DICOW/mt-asr-data-prep/testset_tse')
out_root = Path('output/testset_tse_dicow_from_existing')
out_root.mkdir(parents=True, exist_ok=True)

nsf_hyp = load_hypothesis_map(Path('output/notsofar/mtg_sc160_dicow_offline/hypothesis_multi.jsonl'))
l2m_hyp = defaultdict(list)
for shard in sorted(Path('output/libri2mix_dicow_sharded').glob('shard_*/hypothesis_multi.jsonl')):
    for row in read_jsonl_rows(shard):
        l2m_hyp[row['session_id']].append(row)

summaries = {
    'nsf': build_dataset('nsf', testset_root / 'nsf', nsf_hyp, map_nsf_session, out_root / 'nsf'),
    'l2m': build_dataset('l2m', testset_root / 'l2m', l2m_hyp, map_l2m_session, out_root / 'l2m'),
}
print(json.dumps(summaries, indent=2))
