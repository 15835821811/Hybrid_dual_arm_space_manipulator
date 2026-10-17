"""TRAIN-only mixture of uniform rows and uniform initial q17 origins."""
from collections import Counter
import copy
import numpy as np
from ff_prior.storage import digest


def population(conditions, ids, splits):
    slots=np.flatnonzero(np.asarray(splits)=='TRAIN')
    keys=[digest(conditions[i]['initial_features']) for i in slots]
    counts=Counter(keys); n=len(slots); k=len(counts)
    if n==0:raise ValueError('empty TRAIN')
    weights=np.asarray([.5+.5*n/(k*counts[key]) for key in keys])
    proof={'TRAIN_only':True,'rows':n,'origins':k,'origin_counts':dict(counts),
           'ids_hash':digest([ids[i] for i in slots]),'origin_keys_hash':digest(keys),
           'weights_hash':digest(weights),'mixture':[.5,.5],
           'definition':'half uniform row; half uniform q17 origin then uniform row in origin'}
    return slots,keys,weights,proof


class CoverageSampler:
    """Independent checkpointed RNG; exhaustive queues at both mixture levels."""
    def __init__(self,keys,seed,state=None):
        self.keys=list(keys); self.origins=sorted(set(keys)); self.members={}
        for i,key in enumerate(keys):self.members.setdefault(key,[]).append(i)
        self.rng=np.random.default_rng(seed)
        self.row_queue=[];self.origin_queue=[];self.within={k:[] for k in self.origins}
        self.row_counts=np.zeros(len(keys),dtype=np.int64)
        self.origin_row_counts=np.zeros(len(keys),dtype=np.int64)
        self.origin_counts=np.zeros(len(self.origins),dtype=np.int64)
        if state is not None:
            if state['keys_hash']!=digest(self.keys):raise ValueError('sampling population changed')
            self.rng.bit_generator.state=copy.deepcopy(state['rng'])
            self.row_queue=list(state['row_queue']); self.origin_queue=list(state['origin_queue'])
            self.within=copy.deepcopy(state['within'])
            for name in ('row_counts','origin_row_counts','origin_counts'):
                setattr(self,name,np.asarray(state[name],dtype=np.int64).copy())

    def take(self,batch):
        if batch%2 or batch<=0:raise ValueError('positive even batch required')
        half=batch//2
        while len(self.row_queue)<half:self.row_queue.extend(self.rng.permutation(len(self.keys)).tolist())
        first=self.row_queue[:half];del self.row_queue[:half]
        while len(self.origin_queue)<half:self.origin_queue.extend(self.rng.permutation(len(self.origins)).tolist())
        os=self.origin_queue[:half];del self.origin_queue[:half]
        second=[]
        for oi in os:
            key=self.origins[oi]
            if not self.within[key]:self.within[key]=self.rng.permutation(self.members[key]).tolist()
            second.append(self.within[key].pop())
        np.add.at(self.row_counts,first,1);np.add.at(self.origin_row_counts,second,1)
        np.add.at(self.origin_counts,os,1)
        return first+second

    def state(self):
        return {'keys_hash':digest(self.keys),'rng':copy.deepcopy(self.rng.bit_generator.state),
                'row_queue':list(self.row_queue),'origin_queue':list(self.origin_queue),'within':copy.deepcopy(self.within),
                'row_counts':self.row_counts.copy(),'origin_row_counts':self.origin_row_counts.copy(),
                'origin_counts':self.origin_counts.copy()}

    def report(self,ids):
        return {'stream_draw_counts':{'uniform_rows':int(sum(self.row_counts)),
                                     'uniform_origins':int(sum(self.origin_row_counts))},
                'uniform_row_draw_counts':dict(zip(ids,self.row_counts.tolist())),
                'origin_stream_row_draw_counts':dict(zip(ids,self.origin_row_counts.tolist())),
                'origin_stream_draw_counts':dict(zip(self.origins,self.origin_counts.tolist())),
                'sampler_state_hash':digest(self.state())}
