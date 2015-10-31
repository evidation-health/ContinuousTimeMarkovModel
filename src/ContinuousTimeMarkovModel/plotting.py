from pymc3 import traceplot
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def groundTruthTraceplot(truths,trace,var,sigma=0.01,scale=None,jitter=0.01,ymax=None,show=True,**kwargs):
    ax = traceplot(trace=trace,vars=[var],**kwargs)
#Reset color cycle to match colors properly
    ax[0][0].set_color_cycle(None)
    #ax[0][0].set_line_style('dashed')
    truths = truths.flatten()
    xmin = min(truths.min(),trace[var].min())
    xmax = max(truths.max(),trace[var].max())
    x = np.linspace(xmin,xmax,int(1000*(xmax-xmin)))
    if scale is None:
        scale = len(truths)
    for truth in truths:
        randj = (np.random.rand()-0.5)*jitter
        ax[0][0].plot(x,mlab.normpdf(x,truth+randj,sigma),'--')
        #ax[0][0].plot(x,mlab.normpdf(x,truth,sigma)+(np.random.rand()-0.5)*jitter,'--')
        #ax[0][0].plot(x,gaussian(x,truth,sigma))
    if ymax is not None:
        ax[0][0].set_ylim((0,ymax))
    if show:
        plt.show()
    
