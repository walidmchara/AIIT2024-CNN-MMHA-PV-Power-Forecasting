"""One-hour-ahead PV-power forecasting with CNN–MMHA.

The synthetic smoke-test path verifies software behavior only; model accuracy in the
published paper has not been reproduced here without the original Borj Cedria data.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from aiit_cnn_mmha import CNNMaskedAttention, DEFAULT_FEATURES, prepare, load_frame
from aiit_cnn_mmha.data import make_synthetic
from aiit_cnn_mmha.baselines import make_baseline


def metrics(y:np.ndarray, pred:np.ndarray, min_abs:float=0.01)->dict:
    y=np.asarray(y,dtype=float).ravel();pred=np.asarray(pred,dtype=float).ravel()
    err=pred-y;rss=np.square(err).sum();var=np.square(y-y.mean()).sum()
    valid=np.abs(y)>min_abs
    return {'count':int(len(y)),'MAE':float(np.abs(err).mean()),
            'RMSE':float(np.sqrt(np.square(err).mean())),
            'R2':float(1-rss/var) if var>1e-10 else None,
            'MAPE_pct_above_threshold':float(100*np.mean(np.abs(err[valid]/y[valid]))) if valid.any() else None,
            'mape_threshold_in_target_units':float(min_abs),'mape_evaluated_records':int(valid.sum())}


def main():
    parser=argparse.ArgumentParser(description='CNN–MMHA PV power forecasting')
    parser.add_argument('--csv',help='Hourly PV measurements, columns documented in data/SCHEMA.md')
    parser.add_argument('--smoke',action='store_true',help='Synthetic pipeline sanity check only')
    parser.add_argument('--config',default='configs/paper.json')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--lookback',type=int)
    parser.add_argument('--out',default='outputs/experiment')
    parser.add_argument('--time-col',default='timestamp')
    parser.add_argument('--target',default='pv_power')
    parser.add_argument('--features',help='Comma-separated order of nine weather predictors')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--architecture',choices=['cnn_mmha','mlp','lstm','bilstm','cnn_lstm','cnn_bilstm'],default='cnn_mmha')
    args=parser.parse_args()
    if bool(args.csv)==bool(args.smoke):parser.error('select exactly one of --csv or --smoke')
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    g=cfg['model'];features=args.features.split(',') if args.features else cfg['input']['features']
    if len(features)!=9:parser.error('9 predictor columns are required by the 9×49 projection')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    epochs=args.epochs if args.epochs is not None else (1 if args.smoke else cfg['training']['epochs'])
    if epochs<1:parser.error('epochs must be >= 1')
    lookback=args.lookback or cfg['input']['lookback_hours']
    if args.smoke:
        df=make_synthetic(n=160,seed=args.seed)
        print('Running a synthetic smoke test; its metrics are not publication results.')
    else:df=load_frame(args.csv,features,time_col=args.time_col,target=args.target)
    ds=prepare(df,features=features,target=args.target,time_col=args.time_col,
               lookback=lookback,seed=args.seed)
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    model=(CNNMaskedAttention(n_features=len(features),spatial_dim=g['spatial_output'],
          conv_channels=tuple(g['conv_channels']),model_dim=g['model_dim'],heads=g['attention_heads'],ff_hidden=g['ff_hidden'])
          if args.architecture=='cnn_mmha' else make_baseline(args.architecture,features=len(features))).to(device)
    def loader(mask,shuffle=False):
        x=torch.from_numpy(ds.x[mask]);y=torch.from_numpy(ds.y[mask]);
        return DataLoader(TensorDataset(x,y),batch_size=cfg['training']['batch_size'],shuffle=shuffle)
    train=loader(ds.train,True);val=loader(ds.val);test=loader(ds.test)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['learning_rate'])
    loss_fn=torch.nn.MSELoss()
    best=math.inf;state=None;history=[];patience=cfg['training']['early_stopping_patience'];stale=0
    for epoch in range(1,epochs+1):
        model.train();train_total=0.0;num=0
        for x,y in train:
            x=x.to(device);y=y.to(device)
            opt.zero_grad(set_to_none=True)
            pred=model(x);loss=loss_fn(pred,y)
            if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step();train_total+=float(loss.detach())*len(x);num+=len(x)
        model.eval();v_loss=0.0;v_num=0
        with torch.no_grad():
            for x,y in val:
                pred=model(x.to(device));v_loss+=float(loss_fn(pred,y.to(device)))*len(x);v_num+=len(x)
        v_loss/=max(1,v_num);history.append({'epoch':epoch,'train_MSE_scaled':train_total/num,'val_MSE_scaled':v_loss})
        print(f'epoch {epoch}/{epochs}: train={train_total/num:.6f} val={v_loss:.6f}')
        if v_loss<best:
            best=v_loss;state={k:vv.detach().cpu().clone() for k,vv in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=patience:break
    model.load_state_dict(state)
    model.eval();scaled=[]
    with torch.no_grad():
        for x,_ in test:scaled.append(model(x.to(device)).cpu().numpy())
    pred=ds.target_scaler.inverse_transform(np.concatenate(scaled)[:,None]).ravel()
    actual=ds.target_scaler.inverse_transform(ds.y[ds.test,None]).ravel()
    idx=ds.label_indices[ds.test]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':state,'config':cfg,'features':features,'lookback':lookback,'architecture':args.architecture},out/'checkpoint.pt')
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    pd.DataFrame({'timestamp':ds.timestamps[idx],'actual_PV':actual,'prediction_PV':pred,'regime_id':ds.regime[idx]}).to_csv(out/'test_predictions.csv',index=False)
    report={'status':'synthetic functional test' if args.smoke else 'evaluation on supplied measured CSV',
       'paper_doi':'10.1109/AIIT63112.2025.11082930',
       'architecture':args.architecture,
       'input_records':len(df),'split':{'train':int(ds.train.sum()),'validation':int(ds.val.sum()),'test':int(ds.test.sum())},
       'lookback_hours':lookback,'forecast_steps_ahead':1,
       'all_test':metrics(actual,pred,min_abs=cfg['evaluation']['mape_min_power']),
       'per_fcm_regime':{str(k):metrics(actual[ds.regime[idx]==k],pred[ds.regime[idx]==k],min_abs=cfg['evaluation']['mape_min_power']) for k in range(4) if np.any(ds.regime[idx]==k)},
       'train_pearson_correlations':ds.pearson,'train_spearman_correlations':ds.spearman,
       'validation_notice':'These are independently computed results, not reproduced published figures.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['all_test'],indent=2))
    print('Results:',out)
    return report

if __name__=='__main__':main()
