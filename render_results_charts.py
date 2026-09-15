"""Website charts from the supplied final paper tables (trial means only)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('public/results')
OUT.mkdir(exist_ok=True)
datasets = [
 ('belief-ablation', ['Teleoperation','Pos. belief','Ori. belief','PATH w/o adapt. trans.','PATH w/o ori.','PATH w/o pos.','PATH'],
  [[39.90,35.90,37.30,29.10,34.30,31.30,26.00], [4.328,2.168,2.760,1.018,2.419,1.555,.805], [3.970,4.290,4.810,3.730,4.390,3.910,3.070], [37.490,37.240,39.120,31.240,36.260,33.750,26.510]]),
 ('reference-generation', ['Alpha Blend','Alpha Blend + LPF','PATH'],
  [[32.65,30.35,22.70],[.521,.424,.109],[4.925,4.338,3.432],[39.062,34.654,27.764]]),
 ('teleoperation-comparison', ['SONIC','TWIST2','XR-Teleop','PATH'],
  [[64.18,56.42,55.04,42.82],[15.751,13.946,3.838,1.289],[11.089,8.151,6.062,5.658],[43.518,33.057,34.267,27.228]])
]
titles = ['Completion time (s)', 'Normalized jerk (×10¹⁰)', 'Translation path (m)', 'Rotation path (rad)']
for name, methods, metrics in datasets:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8 if len(methods)>4 else 6.5), layout='constrained')
    fig.set_facecolor('white')
    for ax, values, title in zip(axes.flat, metrics, titles):
        colors=['#169b69' if m=='PATH' else '#9baabd' for m in methods]
        ax.barh(methods, values, color=colors, height=.58)
        ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.24)
        ax.set_title(title+'  ↓', loc='left', fontsize=13, weight='bold', pad=14)
        ax.tick_params(axis='both', length=0, labelsize=10)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='x',color='#edf1f5')
        for i,value in enumerate(values):
            ax.text(value+max(values)*.025,i, f'{value:.2f}' if title.startswith('Completion') else f'{value:.3f}',va='center',fontsize=10,weight='bold' if methods[i]=='PATH' else 'normal',color='#126947' if methods[i]=='PATH' else '#334155')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=150,bbox_inches='tight')
    plt.close(fig)
    print(name)
