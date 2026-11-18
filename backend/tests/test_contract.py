import numpy as np
import pandas as pd
import pytest
from backend.app.core.errors import AppError
from backend.app.models.contract import FEATURES, validate_features,group_scores

@pytest.mark.parametrize('change',['reorder','missing','extra','dtype','nan','inf'])
def test_strict_feature_validation(synthetic_features,change):
    x=synthetic_features.copy()
    if change=='reorder': x=x[list(reversed(FEATURES))]
    if change=='missing': x=x.drop(columns=FEATURES[0])
    if change=='extra': x['label']=1
    if change=='dtype': x[FEATURES[0]]=x[FEATURES[0]].astype('float32')
    if change=='nan': x.loc[0,FEATURES[0]]=np.nan
    if change=='inf': x.loc[0,FEATURES[0]]=np.inf
    with pytest.raises(AppError): validate_features(x)

def test_only_documented_nulls(synthetic_features):
    x=synthetic_features.copy();x.loc[0,'bearing_diff_deg']=np.nan
    with pytest.raises(AppError):validate_features(x)
    validate_features(x,allow_missing=True)

def test_grouping_and_ranking_preserve_rows(synthetic_features):
    m=pd.DataFrame({'point_idx':np.repeat(np.arange(3),5),'candidate_edge_id':np.tile(np.arange(5),3)})
    p=np.linspace(.01,.99,15)
    g=group_scores(synthetic_features,m,p,3,5)
    assert [v[0]['arc_id'] for v in g]==[4,4,4]
    assert [v[0]['row_index'] for v in g]==[4,9,14]
    assert g[1][0]['probability']==p[9]

@pytest.mark.parametrize('change',['short','2d','nan','inf','range','missing_group','arc','duplicate','fractional'])
def test_bad_scores_or_mappings(synthetic_features,change):
    m=pd.DataFrame({'point_idx':np.repeat(np.arange(3),5),'candidate_edge_id':np.tile(np.arange(5),3)})
    p=np.full(15,.5)
    if change=='short':p=p[:-1]
    if change=='2d':p=np.ones((15,2))
    if change=='nan':p[0]=np.nan
    if change=='inf':p[0]=np.inf
    if change=='range':p[0]=1.1
    if change=='missing_group':m.loc[m.point_idx==1,'point_idx']=2
    if change=='arc':m.loc[0,'candidate_edge_id']=5
    if change=='duplicate':m.loc[0,'candidate_edge_id']=1
    if change=='fractional':m['candidate_edge_id']=m.candidate_edge_id.astype(float)
    with pytest.raises(AppError):group_scores(synthetic_features,m,p,3,5)
