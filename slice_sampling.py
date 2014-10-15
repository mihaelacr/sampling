"""Implementation of multi variate slice sampling. For details please see
the chapter on sampling in Bayesian reasoning and Machine learning.
"""

import numpy as np

"""
distribution: callable, a function proportional to the probability distribution we want to sample
steps: the steps used to find out the slice corresponding to a y sample

TODO: add args to distribution
"""
def sliceSample(distribution, steps, nrSamples, nrBurnSamples, init):
  assert len(init) == len(steps)

  currentSample = init
  samples = []

  # Get the samples that we need
  for sampleNr in xrange(nrSamples + nrBurnSamples):
    currentSample = getOneSample(distribution, currentSample, steps)
    print currentSample

    # Do not record the burn samples, we just want to heat up the markov chain
    # with them
    if sampleNr > nrBurnSamples:
      samples += [currentSample]

  return samples

def getOneSample(distribution, currentSample, steps):
  nrDimensions = len(currentSample)
  # Sample the auxiliary variable y
  y = np.random.uniform(0, distribution(currentSample))
  print "y", y

  sliceLeftBoundary = np.array(currentSample)
  sliceRightBoundary = np.array(currentSample)

  # For each dimension do the sampling step
  for dim in xrange(nrDimensions):
    # We have to find the slice for this dimension
    # We start by defining the low and right boundaries

    # The step allowed for this dimension (specified by caller)
    dimensionStep = steps[dim]

    r = np.random.uniform(low=0.0, high=1.0)
    print "r", r
    sliceLeftBoundary[dim]  = currentSample[dim] - r * dimensionStep
    sliceRightBoundary[dim] = currentSample[dim] + (1.0 - r) * dimensionStep

    print "sliceLeftBoundary", sliceLeftBoundary
    print "sliceRightBoundary", sliceRightBoundary

    # Extend the slice as long as we can
    while distribution(sliceLeftBoundary) > y:
      sliceLeftBoundary[dim] = sliceLeftBoundary[dim] - dimensionStep

    while distribution(sliceRightBoundary) > y:
      sliceRightBoundary[dim] = sliceRightBoundary[dim] + dimensionStep

    sample = np.array(currentSample)
    print "currentSample", currentSample

    print "sliceLeftBoundary", sliceLeftBoundary
    print "sliceRightBoundary", sliceRightBoundary


    # Now try to sample from our estimate of the slice
    # if the value is good, end the loop, we have this dimension for our final sample
    # if not,  reduce the slice
    while True:
      # get a uniform sample for the current dimension
      sampleComponent = np.random.uniform(low=0, high=sliceRightBoundary[dim] - sliceLeftBoundary[dim])  + sliceLeftBoundary[dim]
      print "sampleComponent", sampleComponent
      sample[dim] = sampleComponent
      print "sample", sample
      print "distribution(sample)", distribution(sample)
      print "currentSample", currentSample
      # if the sample is not one what we want to keep
      # update the slices
      if distribution(sample) < y:
        # Decide if we shall reduce the slice from the left or right
        if sample[dim] > currentSample[dim]:
          # Reduce from the right
          print "in if"
          sliceRightBoundary[dim] = sample[dim]

        elif sample[dim] < currentSample[dim]:
          print "in elif"
          # Reduce from the left
          sliceLeftBoundary[dim] = sample[dim]
        else:
          raise Exception("We have reached the already accepted sample and rejected it.")
      else:
        # we have established the value on this dimension for our sample
        currentSample = np.array(sample)
        # check that the sample has only changed in this dimension
        assert equalInAllIndicesButOne(currentSample, sample, dim)
        break

  # Once you are done looping trough the dimensions, return the new sample
  return currentSample


def equalInAllIndicesButOne(array1, array2, index):
  for i in xrange(len(array1)):
    if i != index and array2[i] != array2[i]:
      return False

  return True

def main():
  def pdf(x, mean=0, std=1):
    return 1. / (np.sqrt(2 * np.pi) * std) * np.exp( - (x - mean)** 2 / (2 * std **2))

  samples = sliceSample(pdf, np.array([0.01]), 500, 100, np.array([0.0]))
  print "mean"
  print np.array(samples).mean()

  # http://stackoverflow.com/questions/20011122/fitting-a-normal-distribution-to-1d-data
  # do this tomorrow

if __name__ == '__main__':
  main()