"""Implementation of multi variate slice sampling. For details please see
the chapter on sampling in Bayesian reasoning and Machine learning.

For an overview of how to extend slice sampling to the multivariate case, look at
http://en.wikipedia.org/wiki/Slice_sampling#Multivariate_Methods
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


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
    print "currentSample", currentSample

    # Do not record the burn samples, we just want to heat up the markov chain
    # with them
    if sampleNr > nrBurnSamples:
      samples += [currentSample]

  return samples

def getOneSample(distribution, currentSample, steps):

  def getDistWithAddedValueInDimension(sample, dimension, value):
    res = sample.copy()
    res[dimension] += value
    return distribution(res)


  nrDimensions = len(currentSample)
  # Sample the auxiliary variable y
  y = np.random.uniform(0, distribution(currentSample))

  # For each dimension do the sampling step
  for dim in xrange(nrDimensions):
    # We have to find the slice for this dimension
    # We start by defining the low and right boundaries

    # The step allowed for this dimension (specified by caller)
    dimensionStep = steps[dim]

    r = np.random.uniform(low=0.0, high=1.0)

    sliceLeftBoundary  = - r * dimensionStep
    sliceRightBoundary = + (1.0 - r) * dimensionStep

    # Extend the slice as long as we can
    while getDistWithAddedValueInDimension(currentSample, dim, sliceLeftBoundary) > y:
      sliceLeftBoundary = sliceLeftBoundary - dimensionStep

    while getDistWithAddedValueInDimension(currentSample, dim, sliceRightBoundary) > y:
      sliceRightBoundary = sliceRightBoundary + dimensionStep

    # Now try to sample from our estimate of the slice
    # if the value is good, end the loop, we have this dimension for our final sample
    # if not,  reduce the slice
    while True:
      # get a uniform sample for the current dimension
      sampleComponent = np.random.uniform(low=0, high=sliceRightBoundary - sliceLeftBoundary)  + sliceLeftBoundary

      sample = currentSample.copy()
      sample[dim] = sampleComponent + currentSample[dim]

      # if the sample is not one what we want to keep
      # update the slices
      if distribution(sample) < y:
        # Decide if we shall reduce the slice from the left or right
        if sampleComponent > 0:
          # Reduce from the right
          sliceRightBoundary = 0

        elif sampleComponent < 0:
          # Reduce from the left
          sliceLeftBoundary = 0
        else:
          raise Exception("We have reached the already accepted sample and rejected it.")
      else:
        # we have established the value on this dimension for our sample
        # print "sampleComponent", sampleComponent
        currentSample = sample
        break

  # Once you are done looping trough the dimensions, return the new sample
  return currentSample.copy()


def testUnivariateGaussian(mean, std):
  probDist = lambda x: norm.pdf(x, loc=mean, scale=std)
  samples = sliceSample(probDist, np.array([0.01]), 500, 100, np.array([0.0]))

  samples = np.array(samples)
  print samples

  # Fit the normal distribution
  estimatedMu, estimatedStd = norm.fit(samples)

  print "actual mean\n", mean
  print "estimated mean\n", estimatedMu
  print "estimated covariance\n", estimatedStd
  print "actual covariance\n", std

  samples = np.array(samples)
  # Plot the histogram.
  plt.hist(samples, bins=25, normed=True, color='g')

  # Plot the PDF.
  xmin, xmax = plt.xlim()
  x = np.linspace(xmin, xmax, 100)
  p = norm.pdf(x, estimatedMu, estimatedStd)
  plt.plot(x, p, 'k', linewidth=2)
  title = "Fit results: mu = %.2f,  std = %.2f" % (estimatedMu, estimatedStd)
  plt.title(title)

  plt.show()

def testMultiVariateGaussian(mean, cov):
  probDist = lambda x: multivariate_normal.pdf(x, mean=mean, cov=cov)
  samples = sliceSample(probDist, np.array([0.05] * len(mean)), 5000, 500, np.array([0.0] * len(mean)))
  samples = np.array(samples)

  fig = plt.figure()
  plt.plot(samples[:, 0], samples[:, 1], 'r:', label=u'samples')
  plt.show()

  print "actual mean", mean
  print "estimated mean", samples.mean(axis=0)
  print "estimated covariance", np.cov(samples.T, bias=1)
  print "actual covariance", cov


def main():
  testUnivariateGaussian(0.0, 1.0)
  # testUnivariateGaussian(0.0, 0.5)
  # testUnivariateGaussian(-1.0, 0.5)
  # testUnivariateGaussian(-2.0, 1.0)
  # testUnivariateGaussian(-2.0, 2.0)

  testMultiVariateGaussian(np.array([0.0, 0.0]), np.identity(2))

if __name__ == '__main__':
  main()