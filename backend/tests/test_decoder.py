import pandas as pd
import pytest
from backend.app.decoding.hmm_adapter import decode
from backend.app.core.errors import AppError

def c(arc,p):return {'arc_id':arc,'probability':p}

def test_directed_backtracking_beats_independent_argmax():
    ar=pd.DataFrame({'u':[0,1,2,9],'v':[1,2,3,10]})
    g=[[c(0,.9),c(3,.1)],[c(1,.4),c(3,.6)],[c(2,.9),c(3,.1)]]
    r=decode(g,ar,[1,2,3])
    assert r['classifier_arcs']==[0,3,2]
    assert r['hmm_arcs']==[0,1,2]
    assert r['diagnostics']['route_continuity']==1
    assert not r['diagnostics']['transition_breaks']

@pytest.mark.parametrize('times,reason',[([1,100],'GPS_GAP'),([1,1],'NONINCREASING_TIMESTAMP'),([1,2],'NO_DIRECTED_CONNECTION')])
def test_breaks_are_explicit(times,reason):
    # Opposite arcs are forbidden as siblings; only actual directed adjacency is allowed.
    ar=pd.DataFrame({'u':[0,5],'v':[1,6]})
    r=decode([[c(0,.9)],[c(1,.9)]],ar,times)
    assert r['hmm_arcs']==[0,1]
    assert r['diagnostics']['transition_breaks']==[{'observation_index':1,'reason':reason}]
    assert r['diagnostics']['route_continuity']==0

def test_empty_or_out_of_range_groups_fail():
    ar=pd.DataFrame({'u':[0],'v':[1]})
    for g,t in [([],[]),([[c(0,.9)],[]],[1,2]),([[c(2,.9)]],[1])]:
        with pytest.raises(AppError):decode(g,ar,t)

def test_same_arc_and_single_point():
    ar=pd.DataFrame({'u':[0],'v':[1]})
    assert decode([[c(0,.5)]]*3,ar,[1,2,3])['hmm_arcs']==[0,0,0]
