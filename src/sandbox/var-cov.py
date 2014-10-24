
import numpy as np

a = np.array([[1., 5., .6, 7., .1, .4], [5., 7., .8, 7., .9, 1],
              [1.1, 5.2, 3.6, 7.5, 7.1, 1.4]])

print np.cov(a)
print np.var(a)
print np.mean(a)

