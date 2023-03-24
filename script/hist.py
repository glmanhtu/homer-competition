import csv

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('MACOSX')

scales = []
with open('scales.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        scales.append(float(row['scale']))
scales = np.array(scales)

counts, bins = np.histogram(scales, bins=100)
plt.stairs(counts, bins, fill=True)
plt.show()