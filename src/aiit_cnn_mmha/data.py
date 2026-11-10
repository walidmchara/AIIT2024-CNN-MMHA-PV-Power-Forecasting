"""Causal hourly forecast alignment, train-only scaling and FCM weather regimes."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DEFAULT_FEATURES = [
    'global_irradiance', 'solar_height', 'ambient_temperature',
    'wind_speed', 'relative_humidity', 'precipitation',
    'cloud_cover', 'atmospheric_pressure', 'wind_direction',
]
TARGET = 'pv_power'


@dataclass
class ForecastDataset:
    x: np.ndarray
    y: np.ndarray
    label_indices: np.ndarray
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    regime: np.ndarray
    pearson: dict[str,float]
    spearman: dict[str,float]
    features: list[str]
    timestamps: np.ndarray


class FuzzyCMeans:
    def __init__(self, clusters: int=4, fuzziness: float=2.0, random_state: int=42, max_iter: int=150):
        if clusters < 2 or fuzziness<=1: raise ValueError('invalid fuzzy c-means parameters')
        self.clusters,self.m,self.seed,self.max_iter=clusters,fuzziness,random_state,max_iter

    def membership(self, a: np.ndarray) -> np.ndarray:
        d=np.linalg.norm(a[:,None,:]-self.centers_[None,:,:],axis=2)
        exact = d <= 1e-10
        w = np.maximum(d,1e-10)**(-2/(self.m-1))
        u=w/w.sum(axis=1,keepdims=True)
        rows=exact.any(axis=1)
        if rows.any():
            equal=exact[rows].astype(float)
            u[rows] = equal/equal.sum(axis=1,keepdims=True)
        return u

    def fit(self, a: np.ndarray) -> 'FuzzyCMeans':
        if len(a)<self.clusters: raise ValueError('insufficient training records for FCM')
        g=np.random.default_rng(self.seed)
        self.centers_ = a[g.choice(len(a),self.clusters,replace=False)].astype(float).copy()
        for _ in range(self.max_iter):
            u=self.membership(a); w=u**self.m
            centers=(w.T@a)/(w.sum(axis=0)[:,None]+1e-12)
            if np.max(np.abs(self.centers_-centers)) < 1e-6:
                self.centers_=centers;break
            self.centers_=centers
        return self


def load_frame(path: str|Path, features: list[str], time_col: str='timestamp',target: str=TARGET) -> pd.DataFrame:
    frame=pd.read_csv(path)
    required=[*features,target]
    missing=[c for c in required if c not in frame.columns]
    if missing: raise ValueError('missing required columns: '+', '.join(missing))
    if time_col in frame:
        frame[time_col] = pd.to_datetime(frame[time_col], errors='raise')
        if frame[time_col].duplicated().any(): raise ValueError('duplicate timestamps: resolve conflicting observations first')
        frame=frame.sort_values(time_col,kind='stable').reset_index(drop=True)
        gaps=frame[time_col].diff().iloc[1:]
        if not gaps.eq(pd.Timedelta(hours=1)).all():
            raise ValueError('timestamps are not regularly spaced at one-hour intervals; reconcile missing/duplicate hours before forecasting')
    else:
        frame=frame.reset_index(drop=True)
    for c in required: frame[c]=pd.to_numeric(frame[c],errors='coerce')
    return frame


def prepare(frame: pd.DataFrame, features: list[str]|None=None, target: str=TARGET,
            lookback: int=24, seed: int=42, time_col: str='timestamp') -> ForecastDataset:
    features=features or DEFAULT_FEATURES
    if any(c not in frame for c in [*features,target]): raise ValueError('dataset missing configured features/target')
    if len(frame)<max(100,lookback+50): raise ValueError('dataset too short for meaningful holdouts')
    # Chronological 64/16/20 split: 80/20 development/test, 20% of development reserved for validation.
    dev_end=int(len(frame)*.8);train_end=int(dev_end*.8)
    # Causal forward-fill is applied independently after sorting; missing training values use TRAIN medians.
    numeric=frame[[*features,target]].apply(pd.to_numeric,errors='coerce')
    if numeric[target].isna().any():
        raise ValueError('PV target contains missing/non-numeric measurements: resolve target validity before training')
    # Avoid using any future values for filling at the start of the series.
    med=numeric.iloc[:train_end].median().fillna(0)
    numeric=numeric.ffill().fillna(med)
    if numeric.isna().any().any(): raise ValueError('unresolved numeric missing values')
    xf=MinMaxScaler().fit(numeric.iloc[:train_end][features])
    yf=MinMaxScaler().fit(numeric.iloc[:train_end][[target]])
    X=xf.transform(numeric[features]).astype(np.float32)
    Y=yf.transform(numeric[[target]]).ravel().astype(np.float32)
    # Four fuzzy regimes used for descriptive per-regime test reporting; never fitted to held-out data.
    fcm=FuzzyCMeans(4,random_state=seed).fit(X[:train_end])
    regime=fcm.membership(X).argmax(axis=1).astype(int)
    x=np.stack([X[end-lookback+1:end+1] for end in range(lookback-1,len(frame)-1)])
    label_indices=np.arange(lookback,len(frame),dtype=int)
    labels=Y[label_indices]
    train=label_indices<train_end
    val=(label_indices>=train_end)&(label_indices<dev_end)
    test=label_indices>=dev_end
    if any(np.count_nonzero(a)<1 for a in [train,val,test]): raise ValueError('empty train/val/test windows')
    # Pearson and Spearman coefficients are calculated using the training records only.
    pc=numeric.iloc[:train_end][features].corrwith(numeric.iloc[:train_end][target],method='pearson').fillna(0).to_dict()
    sc=numeric.iloc[:train_end][features].corrwith(numeric.iloc[:train_end][target],method='spearman').fillna(0).to_dict()
    times = frame[time_col].astype(str).to_numpy() if time_col in frame else np.arange(len(frame)).astype(str)
    return ForecastDataset(x,labels,label_indices,train,val,test,xf,yf,regime,pc,sc,features,times)


def make_synthetic(n: int=160, seed: int=17) -> pd.DataFrame:
    """Standalone smoke-test signal; NEVER used as empirical PV research data."""
    g=np.random.default_rng(seed);h=np.arange(n);sun=np.maximum(0,np.sin((h%24)*np.pi/12))
    frame=pd.DataFrame({
      'timestamp':pd.date_range('2021-01-01',periods=n,freq='h'),
      'global_irradiance':(sun*850+g.normal(0,15,n)).clip(0),
      'solar_height':(sun*70),
      'ambient_temperature':22+4*sun+g.normal(0,.5,n),
      'wind_speed':np.abs(g.normal(3,.7,n)),
      'relative_humidity':(65-18*sun+g.normal(0,3,n)).clip(0,100),
      'precipitation':g.choice([0,0,0,.2],size=n),
      'cloud_cover':np.clip(g.beta(2,5,size=n)*100,0,100),
      'atmospheric_pressure':1014+g.normal(0,2,n),
      'wind_direction':g.uniform(0,360,size=n),
    })
    frame['pv_power']=np.maximum(0,frame['global_irradiance']*.001*np.maximum(0,(1-frame['cloud_cover']*.005))+g.normal(0,.015,n))
    return frame
