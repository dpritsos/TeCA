#!/usr/bin/env python

import sys
import numpy as np

from teca.analytics.metrix import openness, openness2


# openness(tn, ts, tg)
# print openness(12, 88, 88)

for tn in range(7, 0, -1):
    # print openness(tn, 8, tn)
    print openness2(tn, 8)
