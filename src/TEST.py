

from sklearn import cross_validation

bs = cross_validation.Bootstrap(9, train_size=0.66)

print list(bs)