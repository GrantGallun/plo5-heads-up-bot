import numpy as np
import strategy
from game import AgentConfig,Action,Game,make_deal

def fresh():
    memory={};strategy.new_half(AgentConfig(),memory);return memory

def test_size_evidence_separates_small_bluffs_from_large_value():
    m=fresh();st=3;cls=strategy.BET
    assert 'hist_size' in m, 'size-conditioned range evidence is missing'
    m['hist'][st,cls,0]=30;m['hist'][st,cls,-1]=30
    m['hist_size'][st,cls,0,0]=30;m['hist_size'][st,cls,1,-1]=30
    sim=dict(n=2,pcts=[np.array([.1,.99]) for _ in range(4)])
    small=strategy._weights(sim,{st:cls},m,{st:0})
    large=strategy._weights(sim,{st:cls},m,{st:1})
    # _weights returns densities, so compare integrated bucket masses here.
    assert small[0]*strategy.WIDTH[0]>small[1]*strategy.WIDTH[-1]
    assert large[1]*strategy.WIDTH[-1]>large[0]*strategy.WIDTH[0]

def test_without_size_evidence_the_existing_likelihood_is_preserved():
    m=fresh();assert 'hist_size' in m
    sim=dict(n=3,pcts=[np.array([.1,.6,.99]) for _ in range(4)])
    pooled=strategy._weights(sim,{3:strategy.BET},m)
    sized=strategy._weights(sim,{3:strategy.BET},m,{3:0})
    np.testing.assert_allclose(pooled,sized,rtol=0,atol=1e-14)

def test_history_records_aggressive_size_and_passive_action_preserves_it():
    m=fresh();g=Game(make_deal('size-history',0),0)
    g.apply(Action.RAISE_HALF)
    obs=g.observation(1);h=strategy._new_hand_state(obs);m['hand']=h
    strategy._replay(obs,m)
    assert h.get('sizes',{}).get(0)==0
    before=m['freq'].copy();strategy._replay(obs,m)
    np.testing.assert_array_equal(before,m['freq'])

def test_new_half_resets_conditional_evidence():
    m=fresh();assert 'hist_size' in m
    m['hist_size'][:]=7
    strategy.new_half(AgentConfig(),m)
    assert not m['hist_size'].any()
