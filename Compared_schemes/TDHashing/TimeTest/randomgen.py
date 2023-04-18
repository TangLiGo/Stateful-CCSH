import random
random.seed(10)
x=[]
for i in range(20):
    x.append(int(random.random()*1000))
print("Random number with seed 10: ", x)