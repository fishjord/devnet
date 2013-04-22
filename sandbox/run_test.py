#!/usr/bin/python

import sys
import subprocess

epochs = range(25, 50)
rthreshes = [".05", ".1", ".15", ".2", ".25", ".3", ".35", ".4", ".45", ".5", ".6", ".7", ".8", ".9"]
max_bad_ratios = [".05", ".1", ".15", ".2", ".25", ".3", ".35", ".4", ".45", ".5", ".6", ".7", ".8", ".9"]

#for_paper.dn    0.2     0.3     80      5       0       88      80      0.462427745665  173
sys.stdout.write("#epoch\tdn\trthresh\tmax_below_thresh_ratio\ttp\ttn\tfp\tfn\tnum_correct\taccuracy\ttotal\n")
sys.stdout.flush()

for epoch in epochs:
    for rthresh in rthreshes:
        for max_bad_ratio in max_bad_ratios:
            cmd = ["../../main.py", "-r", rthresh, "-m", max_bad_ratio, "-f", "../../test_images/testing/testing_list.txt", "-d", "5_yneuron_epoch_%s.dn" % epoch, "-o", "epoch_%s_rthresh_%s_max_bad_%s" % (epoch, rthresh, max_bad_ratio)]
            sys.stdout.write("%s\t" % epoch)
            sys.stdout.flush()
            subprocess.check_call(cmd, stdout=open("/dev/null", "w"), stderr=sys.stdout)
            sys.stdout.flush()
