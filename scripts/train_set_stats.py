#!/usr/bin/python

import os
import sys
import random
import numpy

if len(sys.argv) != 1:
    print >>sys.stderr, "USAGE: make_train_test.py"
    sys.exit(1)

input_images = {}
for line in sys.stdin:
    if "background" in line:
        continue

    line = os.path.split(line.strip())[-1]
    person = line
    if "-" in person:
        person = person.split("-")[0]

    if person not in input_images:
        input_images[person] = []
    
    input_images[person].append(line)

people = sorted(input_images.keys())
views_for_person = [len(input_images[x]) for x in people]
num_views = sum(views_for_person)

print "Number of people people: %s" % len(people)
print "Number of views: %s" % num_views
print "Average views per person: %s, stdev: %s" % (numpy.average(views_for_person), numpy.std(views_for_person))

print
print "Person\tViews"

view_histo = {}

for i in range(len(people)):
    views = views_for_person[i]
    print "%s\t%s" % (people[i], views)

    view_histo[views] = view_histo.get(views, 0) + 1

print
print "# views\tCount"

for i in sorted(view_histo.keys()):
    print "%s\t%s" % (i, view_histo[i])
