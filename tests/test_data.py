import numpy as np
from aiit_cnn_mmha.data import DEFAULT_FEATURES,FuzzyCMeans,make_synthetic,prepare

def test_hourly_windows_use_future_labels():
    d=make_synthetic(n=180)
    x=prepare(d,lookback=12)
    assert x.x.shape==(168,12,9)
    assert x.label_indices[0]==12
    assert x.train.sum()+x.val.sum()+x.test.sum()==len(x.y)
    assert x.train.any() and x.val.any() and x.test.any()

def test_scaler_is_fitted_only_on_training_segment():
    df=make_synthetic(n=180)
    df.loc[165:,'ambient_temperature']=10000
    d=prepare(df,lookback=12)
    assert d.feature_scaler.data_max_[DEFAULT_FEATURES.index('ambient_temperature')]<100
    assert d.x[d.test,:,DEFAULT_FEATURES.index('ambient_temperature')].max()>1

def test_memberships_are_normalized():
    x=np.random.default_rng(0).normal(size=(35,4))
    f=FuzzyCMeans().fit(x)
    u=f.membership(x)
    assert np.allclose(u.sum(axis=1),1)
    assert (u>=0).all()

def test_missing_target_is_rejected():
    import pytest
    d=make_synthetic(n=180)
    d.loc[2,'pv_power']=np.nan
    with pytest.raises(ValueError,match='target contains missing'):
        prepare(d)

def test_nonpositive_lookback_is_rejected():
    import pytest
    with pytest.raises(ValueError,match='lookback must be >= 1'):
        prepare(make_synthetic(n=180),lookback=0)

def test_irregular_time_axis_is_rejected():
    import pytest
    from aiit_cnn_mmha.data import load_frame
    from pathlib import Path
    import tempfile
    d=make_synthetic(n=180)
    d=d.drop(index=80).reset_index(drop=True)
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'tmp.csv';d.to_csv(p,index=False)
        with pytest.raises(ValueError,match='one-hour intervals'):
            load_frame(p,DEFAULT_FEATURES)
