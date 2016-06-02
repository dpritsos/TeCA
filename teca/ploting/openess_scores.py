#!/usr/bin/env python

import sys
import numpy as np
sys.path.append('../../teca')
from analytics.metrix import openness

for tn in range(1, 20):
    print openness(tn, 20, tn)
