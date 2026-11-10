"""Render model-prediction comparisons from a completed run.

Usage: python scripts/plot_forecasts.py outputs/experiment
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('run_directory');args=p.parse_args()
    out=Path(args.run_directory);df=pd.read_csv(out/'test_predictions.csv')
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(df['actual_PV'].to_numpy(),label='Measured PV power',lw=1.4)
    ax.plot(df['prediction_PV'].to_numpy(),label='Predicted PV power',lw=1.2)
    ax.set_xlabel('Test sample');ax.set_ylabel('PV power (source units)');ax.legend();fig.tight_layout()
    dest=out/'test_predictions.png';fig.savefig(dest,dpi=160);plt.close(fig)
    print(dest)

if __name__=='__main__':main()
