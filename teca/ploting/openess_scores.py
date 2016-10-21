#!/usr/bin/env python

import sys
import numpy as np
sys.path.append('../../teca')
from analytics.metrix import openness

for tn in range(7, 0, -1):
    print openness(tn, 8, tn)
