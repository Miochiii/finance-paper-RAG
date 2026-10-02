"""Render V11 aggregate figure with the legend outside annotation space."""
import argparse
from pathlib import Path
from rag_fresh_change_experiment import load


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    labels={'live_normal':'原始证据','live_reverse':'顺序反转','live_dedup':'重叠去重','live_distractor':'相关干扰在前'}
    rows=load(output/'analysis_summary.json')['metrics']
    fig,ax=plt.subplots(figsize=(8,4.8))
    x=list(range(len(rows)))
    ax.bar(x,[r['complete'] for r in rows],label='完整',color='#298777')
    ax.bar(x,[r['bad'] for r in rows],bottom=[r['complete'] for r in rows],label='错误或遗漏',color='#d58a35')
    ax.bar(x,[r['quality_u'] for r in rows],bottom=[r['complete']+r['bad'] for r in rows],label='未确定',color='#9e9e9e')
    for i,r in enumerate(rows):
        ax.text(i,r['n']+.3,f"误报 {r['complete_false_flags']}/{r['complete']}",ha='center',fontsize=10)
    ax.set_xticks(x,[labels[r['condition']] for r in rows])
    ax.set_ylim(0,max(r['n'] for r in rows)+3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
    ax.set_ylabel('本轮新回答数')
    ax.set_title('V11：保留原始事实的正常证据压力实验',pad=12)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False,fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.86)
    for ext in ('png','svg'):
        fig.savefig(output/('normal_stress_quality.'+ext),dpi=180)
    plt.close(fig)
    print('V11 figure layout repaired',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    render(ap.parse_args().output)
