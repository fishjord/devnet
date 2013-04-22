#!/usr/bin/python

import sys
import math
import numpy
import dn_utils
from dn_utils import preresponse, activation_function, rescale, amnesic

#Tell numpy to show us more of the arrays
numpy.set_printoptions(threshold=100, linewidth=10000)

"""
Neuron class, holds weights and learning rates for each individual neuron and handles updating said values and computing responses to given inputs
"""
class Neuron:
    """
    Constructor to setup a neuron

    input_sizes = array-like with the dimensionality of each input (expected 1d)
    response_weights = array-like weights of each respective layer input to the final preresponse
    name = the name of the neuron (mostly used for debugging to identify individual neurons)
    """
    def __init__(self, input_sizes, response_weights, name):
        self.name = name

        #We're dealing with floats, so use sorta-equal instead of exact equality
        if math.fabs(1 - sum(response_weights)) > .0001:
            raise Exception("sum of input weights %s doesn't equal 1" % response_weights)

        #Store the relative weights of the various pre-responses to the final neuronal response
        self.response_weights = response_weights

        #Make sure the number of layer connections equals the number of pre-response weightings
        if len(response_weights) != len(input_sizes):
            raise Exception("Number of response weights doesn't equal number of input counts")

        #storage for our weight vectors and variance vecors
        self.weights = []
        self.sigmas = []
        for i in range(len(input_sizes)):
            #Don't want it to be quite zero, norm would be zero and cause bad things
            self.weights.append(numpy.array([0.0001] * input_sizes[i]))
            #Init sigma to the value from the slide on week 13 day 2
            self.sigmas.append(numpy.array([1.0 / math.sqrt(12)] * input_sizes[i]))

        #initalize age and learning rate
        self.age = 1.0

        self.omega1 = 0
        self.omega2 = 1
    
    """
    Called to UPDATE our input weights AFTER we've fired (pass in the response from fire())
    """
    def update_weights(self, inputs, response):
        #print "\t+++++Updating weights for neuron %s, age %s, response %s (learning rates: %s, %s)" % (self.name, self.age, response, self.omega1, self.omega2)

        #Compute the response to each set of inputs
        for i in range(len(inputs)):
            #special case, if the weight for this input is 0 don't calculate it
            if self.response_weights[i] == 0:
                continue

            #print "\t\tUpdating input %s weights" % i
            #print "\t\t\tLearning from pattern %s " % inputs[i]
            #print "\t\t\told weights           %s" % self.weights[i]
            p = inputs[i]
            #sanity check
            if len(p) != len(self.weights[i]):
                raise Exception("Number of bottom connections (%s) doesn't equal number of weights (%s)" % (len(p), len(self.vb)))
            
            #only update the variance after the critical age, n0
            if self.age > dn_utils.sigma_n0:
                if self.name == "Y_0":
                    print "Updating sigma from %s" % self.sigmas[i]
                self.sigmas[i] = self.omega1 * self.sigmas[i] + self.omega2 * numpy.absolute(self.weights[i] - p)
                if self.name == "Y_0":
                    print "Updating sigma to %s" % self.sigmas[i]
            
            #update the weight vector
            self.weights[i] = self.omega1 * self.weights[i] + self.omega2 * response * p
            if self.name == "Y_0":
                print "Updating weight to %s" % self.weights[i]
            #self.weights[i] = dn_utils.ae_function(self.sigmas[i]) * self.weights[i] #synapse retraction

            #print "\t\t\tnew weights           %s" % self.weights[i]

        #Finally update our learning rate
        self.age += 1
        self.omega1 = (self.age - 1 - amnesic(self.age)) / self.age
        self.omega2 = (1 + amnesic(self.age)) / self.age

        #print "\t-----New stats for neuron %s, %s %s %s" % (self.name, self.age, self.omega1, self.omega2)

    """
    Compute this input's deviation from the expected value given this neuron fires
    """
    def compute_deviation(self, inputs, rthresh=.3):
        deviations = []
        tmp = []
        t2 = []
        for i in range(len(inputs)):
            if self.response_weights[i] == 0:
                continue

            this_sigma = numpy.absolute(self.weights[i] - inputs[i])
            ae = dn_utils.ae_function(this_sigma, self.sigmas[i])
            t2.append(this_sigma)
            tmp.append(numpy.average(this_sigma))
            deviations.append(ae * self.response_weights[i])

        print "Inputs: %s" % inputs[0]
        print "Simga: %s" % self.sigmas[0]

        num_zero_rs = 0.0
        for r in deviations[0]:
            if r < rthresh:
                num_zero_rs += 1

        print
        print "This sigma: %s" % t2
        print "This sigma avg var: %s" % tmp
        print "R vector: %s" % deviations
        print "R average: %s" % numpy.average(deviations)
        print "R zeros: %s" %  num_zero_rs
        print
        print

        return num_zero_rs / len(deviations[0]) #numpy.average(deviations)

    """
    Computes this neuron's over all pre-response value from the given set of inputs
    """
    def fire(self, inputs):
        pr = 0  #over all pre-response value
        #loop over every input vector
        for i in range(len(inputs)):
            response_weight = self.response_weights[i]
            #Special case, don't waste effort computing a response that won't be used
            if response_weight == 0:
                continue

            p = inputs[i]
            v = self.weights[i]
            #Sanity check
            if len(p) != len(v):
                raise Exception("Number of bottom connections (%s) doesn't equal number of weights (%s)" % (len(p), len(v)))

            #compute the ith preresponse (will thrown an exception if the input vector is all zeros)
            pr_i = response_weight * preresponse(p, v)

            pr += pr_i

            #print("Neuron %s firing with input %s (weight= %s, pr_i= %s, pr= %s)" % (self.name, i, response_weight, pr_i, pr))
            #print("\tpattern= %s" % p)
            #print("\tweights= %s" % v)

        #Well, we don't want to make sure that this'll aaaalways fire
        #maybe there is a real one that matched, so we'll just say 90 instead of 100
        if self.age == 1:
            return .99

        return activation_function(pr)  #Use our activation function, could be any number of functions (ie sigmoidal), right now it simply returns the input parameter

"""
Layer class
Holds a set of neurons that make up the layer, manages:
 processing inputs
 top-k competition
 updating weights
"""
class Layer:
    """
    Constructor
    c = number of neurons in the layer
    k = top k neurons that can fire
    input_sizes = number of input layers and their sizes (array-like)
    input_weights = relative weights of each input response to each neuron's final pre-response
    name = layer name, debugging
    """
    def __init__(self, c, k, input_sizes, input_weights, name):
        self.k = k
        self.name = name
        #initalize our current response to random garbage
        self.response = numpy.random.rand(c)
        #initalize our next response to random garbage (we use two vectors and swap them so that we can do updates in any order)
        self.next_response = numpy.random.rand(c)
        #hold each layer we're connected to
        self.connections = []

        #setup all the neurons
        self.neurons = []
        for i in range(c):
            self.neurons.append(Neuron(input_sizes, input_weights, "%s_%s" % (self.name, i)))

    """
    Supervise the layer and optionally update the weights of the neurons based on the supervised response

    supervision = array-like vector of supervised responses, must have the same dimensionality as the layer
    update = whether or not to update the weight vectors of the individual neurons
    """
    def supervise(self, supervision, update = True):
        if len(supervision) != len(self.neurons):
            raise Exception("Not enough supervision for number of neurons")

        #Get the response vectors from all of our input layers
        inputs = [layer.response for layer in self.connections]

        #print "\tLayer %s responses: %s" % (self.name, supervision)

        #As a sanity check, keep track of the number of neurons that fire because of supervision and make sure it isn't more than k
        num_updated = 0
        for i in range(len(supervision)):
            self.next_response[i] = supervision[i]
            #Assume a supervision of 0 means don't fire
            if supervision[i] > 0:
                if update:
                    #print("updating neuron %s in layer %s after supervision %s" % (i, self.name, supervision[i]))
                    self.neurons[i].update_weights(inputs, self.next_response[i])
                num_updated += 1

        if self.k > 0 and num_updated > self.k:
            raise Exception("Supervision caused more neurons to fire [%s] than top k [%s]" % (num_updated, self.k))

    """
    Connect this layer with the input layer 'layer'
    """
    def connect(self, layer):
        self.connections.append(layer)

    """
    swap the self.response with self.next_response (called at the end of an update cycle)
    """
    def swap_responses(self):
        tmp = self.response
        self.response = self.next_response
        self.next_response = tmp

    """
    Compute the response from this layer using the connected layer responses as input
    """
    def get_response(self, update = False, rthresh = .3, max_low_conf_ratio = .4):
        #our inputs are the outputs from the layers we're connected to
        inputs = [layer.response for layer in self.connections]

        #For each neuron compute the response
        for i in range(len(self.neurons)):
            try:
                self.next_response[i] = self.neurons[i].fire(inputs)
            except:
                """This can happen if either the weight or input vector is zero
                the input vector CAN be zero now since we're supressing firing
                in the case that the input is outside of the expected variation
                """
                if not update:
                    self.next_response[i] = 0.0
                else:
                    #If we're suppose to update...well...we can't do anything, so just pass the error up
                    raise

        #A simple stable (implemented as insertion) sort
        #Need to make sure that the same neuron is picked every time
        top_neurons = dn_utils.stable_argsort(self.next_response)
        #print "\tLayer %s responses: %s" % (self.name, self.next_response)
        #print "\t%s" % top_neurons

        if self.k > 0:
            #Zero out the k to c neurons (the surpressed neurons by lateral inhibition)
            for i in range(self.k, len(self.neurons)):
                idx = top_neurons[i]
                self.next_response[idx] = 0.0

            #Rescale the top k neuron responses
            for i in range(self.k):
                idx = top_neurons[i]
                neuron = self.neurons[idx]

                self.next_response[idx] = (1.0 / (i + 1)) * self.next_response[idx]
                if update:
                    #print("updating neuron %s in layer %s" % (idx, self.name))
                    #Fire zee neuron!
                    self.neurons[idx].update_weights(inputs, self.next_response[idx])
                else:
                    print "%s, %s, %s" % (self.name, neuron.name, neuron.age)
                    dev = neuron.compute_deviation(inputs, rthresh)
                    print dev
                    if self.name == "Y" and dev > max_low_conf_ratio:
                        self.next_response[idx] = 0

"""
Development Network class
"""
class DN:
    def __init__(self, n, input_size, bg_img, bg_label, labels):
        #Number of neurons in the hidden label
        c = n * n

        num_labels = len(labels)
        #setup the layers
        self.x = Layer(input_size, -1, [c], [1.0], "X")
        self.y = Layer(c, 1, [input_size, num_labels], [1.0, 0.0], "Y")
        self.z = Layer(num_labels, 1, [c], [1.0], "Z")
        self.labels = labels

        print >>sys.stderr, "Connecting %s neurons in x to %s neurons in y" % (len(self.x.neurons), len(self.y.neurons))
        
        #setup layer connections
        self.x.connect(self.y)

        self.y.connect(self.x)
        self.y.connect(self.z)

        self.z.connect(self.y)

        if len(set(labels)) != len(labels):
            raise Exception("Some labels are duplicated")

        self.bg_img = bg_img
        print self.bg_img
        self.bg_label = bg_label
        self.last_label = bg_label


        self.rthresh = 0.0
        self.max_low_conf_ratio = 1.0

        #start with 2 background images
        self._present(self.bg_img, self.bg_label, update=False) # Don't update on the first one, the responses in the layers are garbage
        self._present(self.bg_img, self.bg_label, update=True)

    """
    Make the network perform (ie, don't update weights)
    """
    def perform(self, pattern):
        rets = []

        #continue the pattern of image image image bg bg
        rets.append(self._present(pattern, update=False))
        rets.append(self._present(pattern, update=False))
        rets.append(self._present(pattern, update=False))
        rets.append(self._present(self.bg_img, update=False))
        rets.append(self._present(self.bg_img, update=False))

        #print "*********Responses after performing %s" % rets
        #print

        return rets[-1]

    """
    Learn and perform from the input pattern
    """
    def learn(self, pattern, label):
        rets = []

        print "========Learning: %s" % (label)
        rets.append(self._present(pattern, self.bg_label, update=True))
        rets.append(self._present(pattern, update=True))
        rets.append(self._present(pattern, label, update=True))
        rets.append(self._present(self.bg_img, update=True))
        rets.append(self._present(self.bg_img, label, update=True))

        print "========Responses after learning %s" % rets
        print 

        return rets[-1]

    #handle the actual DN computation
    def _present(self, pattern, label = None, update=False):
        #print "\tPresented %s: %s (last label: %s)" % (label, pattern, self.last_label)
        self.last_label = label

        #rescale the input pattern so that every dimension is between 0 and 1
        pattern = rescale(pattern, 0, 1)

        #the x layer is always supervised
        self.x.supervise(pattern, update=False)
        #print "\tInput pattern:             %s" % pattern

        #Y layer is never supervised
        self.y.get_response(update, self.rthresh, self.max_low_conf_ratio)
        
        #Decide if we're suppose to supervise Z
        if label != None:
            supervision = []
            #Since they gave us a label, we have to convert it to a vector to supervise z with
            for l in self.labels:
                if l == label:
                    print("Supervising %s, %s" % (label, len(supervision)))
                    supervision.append(1.0)
                else:
                    supervision.append(0.0)
            #Then supervise...
            self.z.supervise(supervision, update)
        else:
            #Otherwise we just ask it for it's response
            self.z.get_response(update)

        #We're done with this update, so swap all the layer responses to get ready for the next
        self.x.swap_responses()
        self.y.swap_responses()
        self.z.swap_responses()

        #Aaaaand finally figure out which labels corrospond to the firing Z neurons
        rets = []
        for i in range(len(self.z.neurons)):
            if self.z.response[i] > 0:
                rets.append((self.z.response[i], self.labels[i]))

        #print 

        #Return the labels in order of highest response to lowest response
        return [x[1] for x in sorted(rets, key=lambda x: x[0])]
