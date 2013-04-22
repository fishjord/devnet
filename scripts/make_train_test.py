#!/usr/bin/python

import os
import sys
import random

if len(sys.argv) != 3:
    print >>sys.stderr, "USAGE: make_train_test.py <train_ratio> <unknown_ratio>"
    sys.exit(1)

train_image_ratio = float(sys.argv[1])
unknown_ratio = float(sys.argv[2])

if train_image_ratio > 1 or unknown_ratio > 1 or train_image_ratio < 0 or unknown_ratio < 0:
    print >>sys.stderr, "All inputs must be in the range (0, 1)"
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

people = list(input_images.keys())
random.shuffle(people)

num_unknown = int(len(people) * unknown_ratio)

unknown_people = people[:num_unknown]
people = people[num_unknown:]

training = {}
testing = {}

for person in people:
    imgs = input_images[person]
    random.shuffle(imgs)

    if len(imgs) < 2:
        print >>sys.stderr, "Not enough images for person %s" % person
        continue

    num_training = int(len(imgs) * train_image_ratio)
    num_training = max(1, num_training)
    training[person] = input_images[person][:num_training]
    testing[person] = input_images[person][num_training:]

for person in unknown_people:
    testing[person] = input_images[person]

write_file(training, "training_list.txt")
write_file(testing, "testing_list.txt")

print "Input people: %s" % len(input_images)
print "People in training set: %s" % len(people)
print "People left out of training: %s" % len(unknown_people)

print "Num images in training: %s" % (sum([len(training[x]) for x in people]))
print "Num images in testing: %s" % (sum([len(testing[x]) for x in people]))
print "Num images in unknown: %s" % (sum([len(input_images[x]) for x in unknown_people]))
