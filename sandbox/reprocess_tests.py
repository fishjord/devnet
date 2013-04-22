#!/usr/bin/python

import numpy
import sys
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import os

params = {'backend': 'ps',
          'font.family' : 'cmr10',
          'font.serif' : 'Computer Modern Roman',
           'axes.labelsize': 8,
           'text.fontsize': 8,
           'legend.fontsize': 8,
           'xtick.labelsize': 6,
           'ytick.labelsize': 6,
           'text.usetex': True,
           'axes.linewidth' : 0.75
           }
matplotlib.rcParams.update(params)

rthreshes = [".05", ".1", ".15", ".2", ".25", ".3", ".35", ".4", ".45", ".5", ".6", ".7", ".8", ".9"]
max_bad_ratios = [".05", ".1", ".15", ".2", ".25", ".3", ".35", ".4", ".45", ".5", ".6", ".7", ".8", ".9"]

#for_paper.dn    0.2     0.3     80      5       0       88      80      0.462427745665  173
sys.stdout.write("#epoch\tdn\trthresh\tmax_below_thresh_ratio\ttp\ttn\tfp\tfn\tnum_correct\taccuracy\ttotal\n")
sys.stdout.flush()

line_types = ["o", "D", "^", "*"]
epochs = range(50)
plot_epochs = [0, 5, 25, 27] #len(line_types))

best_error = (1.0, None, None, None, None, None, None, None)
best = (0, 0, 0, None, None, None, None, None, None, None)
for i in range(len(epochs)):
    epoch = epochs[i]
    tprs = [0.0]
    spcs = [0.0]

    for rthresh in rthreshes:
        for max_bad_ratio in max_bad_ratios:
            f = "raw_results/epoch_%s_rthresh_%s_max_bad_%s_results.txt" % (epoch, rthresh, max_bad_ratio)
            if not os.path.exists(f):
                continue
            infile = open(f)

            tp = 0
            fp = 0
            tn = 0
            fn = 0
            corr = 0

            for line in infile:
                if line[0] == "#":
                    continue
                
                lexemes = line.strip().split()
                if len(lexemes) != 4:
                    continue

                name = lexemes[1]
                label = lexemes[2]

                if label == name:
                    corr += 1

                if label == "rejected":
                    if name == "rejected":
                        tn += 1
                    else:
                        fn += 1
                elif name == "rejected":
                    fp += 1
                else:
                    tp += 1

            if tp+tn+fp+fn != 1638:
                continue

            print "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (epoch + 1, rthresh, max_bad_ratio, tp, tn, fp, fn, corr, corr / float(tp + tn + fp + fn), tp + tn + fp + fn)
            tpr = tp / float(tp + fn)
            spc = tn / float(tn + fp)
            error = 1 - (corr / float(tp + tn + fp + fn))

            
            tprs.append(tpr)
            spcs.append(spc)

            if tpr + spc > best[0]:
                best = (tpr + spc, tpr, spc, tp, fp, tn, fn, epoch, rthresh, max_bad_ratio)

            if error < best_error[0]:
                best_error = (error, tp, fp, tn, fn, epoch, rthresh, max_bad_ratio)

    tprs = numpy.array(tprs)
    spcs = 1 - numpy.array(spcs)

    order = numpy.argsort(spcs)

    spcs = spcs[order]
    tprs = tprs[order]

    if epoch in plot_epochs:
        plt.figure(0)
        plt.plot(spcs, tprs, linestyle="-", marker=line_types[plot_epochs.index(i)], markevery=10, markersize=2.5, linewidth=.25, label="Epoch %s" % (epoch + 1))

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(2.8,2.8)
fig.subplots_adjust(bottom=0.18, left=.18)

plt.figure(0)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4, prop={'size':6})
plt.savefig("roc.png", dpi=226)
plt.savefig("roc.eps")

print "Best:", best
print "Best error:",best_error
