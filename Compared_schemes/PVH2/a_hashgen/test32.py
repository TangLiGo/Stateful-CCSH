import numpy as np
import math
forged_info = [[1, 210], [120, 162], [233, 313], [61, 122], [1, 190], [211, 262], [90, 117], [133, 160], [169, 359],
                   [104, 149]]# 954
#forged_info=np.array(forged_info)
for i in range(len(forged_info)):
    forged_info[i][0]=int(forged_info[i][0]/5)
    forged_info[i][1] = math.ceil(forged_info[i][1] / 5)
print("forged_info=",forged_info)