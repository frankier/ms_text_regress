import sys

import numpy

mat = numpy.load(sys.argv[1], allow_pickle=True)
print(mat)
prop = mat / mat.sum(axis=-1)[:, numpy.newaxis]
print(prop)

prop = prop[0:2, 0:2]
print(prop)

mutual_information = numpy.sum(
    prop * numpy.log(prop / (prop.sum(axis=0) * prop.sum(axis=1)))
)
print(mutual_information)
