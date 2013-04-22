#!/usr/bin/python

import math
import os
import numpy

expected_width = 64
expected_height = 88

t1 = 10     #samples to use normal average for (amensiac range 1)
t2 = 1000   #samples to use linear scale to c * learning rate
c = 2       #learnnig rate in second interval
r = 10000   #learning rate

#NE constants
sigma_n0 = 3
beta_s = 0.8
beta_b = 1.2

"""
Function to compute the NE factor
Taken from week 13 day 2 slides
"""
def ae_function(sigma, real_dev = None):
    if real_dev == None:
        real_dev = sigma

    r = sigma / numpy.average(real_dev)

    for i in range(len(r)):
        t = r[i]
        if t < beta_s:
            r[i] = 1.0
        elif t < beta_b:
            r[i] = (beta_b - r[i]) / (beta_b - beta_s)
        else:
            r[i] = 0.0

    return r

"""
Sort the input array with a stable sorting algorithm (insertion sort)
"""
def stable_argsort(arr):
    ret = range(len(arr))

    for i in range(1, len(arr)):
        j = i
        item = arr[ret[j]]

        while j > 0 and arr[ret[j - 1]] < item:
            ret[j] = ret[j - 1]
            j -= 1

        ret[j] = i

    return ret

"""
Rescales the input vector so every value is between new_low and new_high (default 0-1)
"""
def rescale(pattern, new_low = 0, new_high = 1):
    l = min(pattern)
    u = max(pattern)

    m = float(u - l)

    #This would give a small divide by zero error
    if m == 0:
        raise Exception("min and max are the same")

    ret = (pattern - l) * (new_high - new_low) / float(u - l)

    return ret

"""
activation function

just returns the pre-response value, but could be modified to sigmoidal or any other response function
"""
def activation_function(x):
    return x

"""
Amnesic mean function, taken from lecture notes
"""
def amnesic(t):
    if t <= t1:
        return 0
    elif t <= t2:
        return c * (t - t1) / (t2 - t1)
    else:
        return c + (t - t2) / r

"""
Exepects an array of neuron-like object for input neurons
i.e. something with a .response attribute (class, namedtuple)
"""
def preresponse(input_neurons, weights):
    s = numpy.dot(input_neurons, weights)
    normX = numpy.linalg.norm(input_neurons, ord=2)
    normW = numpy.linalg.norm(weights, ord=2)

    if normX == 0 or normW == 0:
        #print("s: %s, normX: %s, normW: %s" % (s, normX, normW))
        #print("input neurons: %s" % input_neurons)
        #print("weights: %s" % weights)
        raise Exception("A vector norm was zero!")

    #normX = math.sqrt(normX)
    #normW = math.sqrt(normW)

    return s / (normX * normW)

"""
write a pgm image of the vector in data to the given file name

data is rescaled to 0-255
"""
def write_image(fname, h, w, data):
    out = open(fname, "wb")
    l = min(data)
    u = max(data)
    m = (u - l)

    towrite = numpy.array(((data - l) / (float(m))) * 255, numpy.uint8)

    out.write("P5\n%s  %s  255\n" % (w, h))
    towrite.tofile(out)
    out.close()

"""
generator for the training/testing list file format
"""
def read_faces(fname):
    base_path = os.path.split(fname)[0]

    stream = open(fname)
    try:
        people = int(stream.readline().strip())
        samples = int(stream.readline().strip())
        images_per_person = [int(stream.readline().strip()) for i in range(people)]

        if len(images_per_person) != people:
            raise IOError("Expected %s samples per person lines, not %s" % people, len(images_per_person))

        read_cnt = 0
        for person in range(people):
            for img in range(images_per_person[person]):
                read_cnt += 1

                ret = stream.readline()
                if ret == "":
                    raise EOFError

                yield (person, img, os.path.join(base_path, ret.strip()))

    except Exception as e:
        raise IOError("Error while reading list file: %s" % e)

"""
read the raw grey-scale image from fname
"""
def read_face(fname):
    ret = numpy.fromfile(open(fname, "rb"), numpy.uint8, expected_width * expected_height)
    return ret

"""
helper function to parse information out of the file name
"""
def split_img_name(fname):
    n = os.path.split(fname)[-1]
    if "-" in n:
        return n.split("-")[0]
    else:
        return n.split(".")[0][:-2]
