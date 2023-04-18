import math
a=[[320,240],[704,480],[1280,720],[1920,1080],[3840,2160]]
b=[]
for i in range(len(a)):
    b.append(math.ceil(a[i][0]/16)*math.ceil(math.log2((a[i][1]/16)-1))+math.ceil(a[i][1]/16)*math.ceil(math.log2((a[i][0]/16)-1)))
print(b,math.ceil(math.log2(1080/16-1)))
