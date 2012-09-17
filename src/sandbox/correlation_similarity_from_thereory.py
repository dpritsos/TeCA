
import scipy.spatial.distance as ssd
import numpy as np

a = np.array([[0,1,1,1,1,1,1,1]])
b = np.array([[1,0,0,0,0,0,0,0]])

a= a[0]
b = b[0]

a_ = np.where(a > 0, 0, 1)
b_ = np.where(b > 0, 0, 1)

print a, a_

s11 = np.dot(a,b)
s00 = np.dot(a_,b_)

print s00, s11

s01 = np.dot(a_,b)
s10 = np.dot(a,b_)

print s01, s10

print "SUBST", (s11*s00 - s01*s10)
print "SQRT", np.sqrt((s10+s11)*(s01+s00)*(s11+s01)*(s00+s10))

corr = (s11*s00 - s01*s10) / np.sqrt((s10+s11)*(s01+s00)*(s11+s01)*(s00+s10))
print "CORR SIM", corr

print "CORR DIF", ssd.correlation(a, b)

