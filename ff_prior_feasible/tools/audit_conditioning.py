"""Read-only TRAIN/VAL representation audit; never trains or reads TEST arrays."""
from collections import Counter
from ff_prior.storage import read_json,atomic_json,file_digest,digest
from ff_prior.contracts import public_condition,CONDITION_KEYS
from ff_prior_feasible.experiment import ROOT,PREVIOUS


def main():
    index=read_json(PREVIOUS/'training_index.json')['records']; counts={}; input_files={}
    for split in ('TRAIN','VAL'):
        entries=[e for e in index if e['split']==split]; rows=[]
        for e in entries:
            p=ROOT/e['record']; assert file_digest(p)==e['record_sha256']
            input_files[e['record']]=e['record_sha256']; rows.append(read_json(p))
        scenes=Counter(bool(r.get('scene',{}).get('objects',[])) for r in rows)
        counts[split]={'rows':len(rows),'family_count':len({e['family_id'] for e in entries}),
            'obstacle_scene_rows':scenes[True],'empty_scene_rows':scenes[False],
            'unique_initial_features':len({digest(r['condition']['initial_features']) for r in rows}),
            'unique_conditions':len({digest(r['condition']) for r in rows}),
            'condition_dimensions':sorted({len(public_condition(r['condition'])) for r in rows})}
    result={'status':'OBSERVED','counts':counts,'public_condition_keys':sorted(CONDITION_KEYS),
        'learned_condition_includes_obstacle_geometry':False,
        'geometry_is_available_to_runtime_collision_guidance':True,
        'training_uses_full_initial_state_runtime_audit':True,
        'scope':'stationary origins with fixed base pose; q17 alone does not encode arbitrary base pose or momentum',
        'interpretation':'few obstacle TRAIN labels and no learned obstacle input limit learned collision-specific proposal discrimination; causal effect not measured',
        'input_record_hashes':input_files,'TEST_arrays_read':False,'extra_model_queries':0}
    atomic_json(ROOT/'ff_prior_feasible/evidence/conditioning_audit.json',result)
    print({k:v for k,v in result.items() if k!='input_record_hashes'})


if __name__=='__main__': main()
