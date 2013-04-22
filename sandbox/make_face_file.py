#!/usr/bin/python

import os
import sys
import random


first_only = False
if len(sys.argv) > 1 and sys.argv[1] == "--first-only":
    first_file = 2
else:
    first_file = 1

if len(sys.argv) - first_file < 1:
    print >>sys.stderr, "USAGE: make_train_test.py [--first-only] <face file...>"
    sys.exit(1)

def write_file(imgs, out_file):
    out = open(out_file, "w")
    people = sorted(imgs.keys())
    img_counts = [len(imgs[person]) for person in people]
    tot = sum(img_counts)

    out.write("%s\n" % len(img_counts))
    out.write("%s\n" % tot)
    for cnt in img_counts:
        out.write("%s\n" % cnt)

    for person in people:
        for img in imgs[person]:
            out.write("%s\n" % img)

    out.close()

input_images = {}
for line in sys.argv[first_file:]:
    if "background" in line:
        continue

    line = os.path.split(line.strip())[-1]
    person = line
    if "-" in person:
        person = person.split("-")[0]

    if person not in input_images:
        input_images[person] = []
    else if first_only:
        continue
    
    input_images[person].append(line)

write_file(input_images, "list.txt")

print "Input people: %s" % len(input_images)
