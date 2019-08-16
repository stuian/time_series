import numpy as np

a = np.array([3,4,1,7,2])

print(sorted(enumerate(a),key=lambda x: x[1])[:2])
