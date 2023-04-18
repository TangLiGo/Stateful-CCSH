import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, perm

tp=0.05
m=50
a=0.85
t_min=30
n=8100#1320

def ccsh_ave_p(N,N_c):

    return t_min*N/n+(min(a*t_min*N/n,N_c)-t_min*N_c/n)*(n-N)/(n-N_c)
def ccsh_prob_min(N,N_c):
    b=2
    return (1 - math.pow(b,m*(m-1)/2)*math.pow((1 - t_min / n), N * m) * math.pow(1 - (t_min ) / (n - N_c), (N - N_c) * (m * (m - 1) / 2)))
def ccsh_prob(N,N_c):
    b=1/N_c
    return 1-math.pow(b,m*(m-1)/2)*math.pow(comb(n-t_min,N)/comb(n,N),m)*math.pow(comb(n-t_min-N_c,N-N_c)/comb(n-N_c,N-N_c),m*(m-1)/2)
def ccsh_ave_p_all(N,N_c,i):
    if i==2:
        return tp*N
    else:
        return tp*N+(min(a*ccsh_ave_p_all(N,N_c,i-1),N_c)-tp*N_c)*(n-N)/(n-N_c)
def ccsh_ave_p_sum(N,N_c,i):
    if i==2:

        return tp*N,tp*N
    else:
        last,last_sum=ccsh_ave_p_sum(N,N_c,i-1)
     #   print("p is ",i-1,last)
        cur=tp*N+(min(a*last,N_c)-tp*N_c)*(n-N)/(n-N_c)
        return cur,cur+last_sum
p_info=[]
def ccsh_p_change(N,N_c,i):
    if i==2:

        return tp*N
    else:
        last=ccsh_p_change(N,N_c,i-1)
        p_info.append(last)
      #  print("p" , last)
        return tp*N+(min(a*last,N_c)-tp*N_c)*(n-N)/(n-N_c)
def gsh_ave_p_sum(N,i):

    return i*t_min*N/n
def merged_p(x):
    return 1-math.pow(1-tp,m*x)+tp*x
def pa_ccsh(x1,x2,i):
    return (1-math.pow(a,i-2))/(1-a)*tp*x1+math.pow(a,i-2)*tp*x2
def rsh_getN(P,m):

    return 1/m*math.log(1-P,1-t_min/n)
def rsh(N):
    return 1-math.pow((1-tp),N*m)
def ccsh_r(N,N_c,p):

    return (1-math.pow((1-t_min/n),N*m)*math.pow(1-(t_min-N_c)/(n-N_c),(N-N_c)*(m*(m-1)/2)))
def testD(N,N_c):
   print(tp*n*(N-N_c)/(n-N_c-a*(n-N)))
   print("max", tp *  (N - N_c) / (1-a+a*N/n))
   print("min",tp*N)
   z= tp * N + ( N_c - tp * N_c) * (n - N) / (n - N_c)
   print("z",z)
def test_large_Nc(N,N_c):
    return tp*N+(1-tp)*N_c*(n-N)/(n-N_c)
plt.figure()
for Nc_temp in range(1,5):
    ccsh_p_change(20, Nc_temp, m)

    plt.plot(p_info)
    p_info = []
plt.figure()
#plt.plot([(ccsh_ave_p_sum(20,i,m))[1]/m for i in [0,1,2,3,4,5]],'rx')
plt.plot([(ccsh_ave_p_sum(20,i,m))[1]/m for i in [0,1,2,3,4,5]], color='red', marker='o',markerfacecolor='none')
plt.savefig('figs/v1_m2_t1.png')
#plt.plot([gsh_ave_p_sum(20,m)/m for i in [0,1,2,3,4,5]])
plt.show()
Ns=range(5,100,5)
plt.figure()
plt.plot(Ns,[rsh(i) for i in Ns],'rx')
plt.show()
plt.figure()
plt.plot([test_large_Nc(30,i) for i in [0,1,2,3,4,5,6,7,8,9,10]],'rx')
#plt.plot([gsh_ave_p_sum(20,m)/m for i in [0,1,2,3,4,5]])
plt.show()
testD(20,1)
testD(20,4)
for Nc_temp in range(1,5):
    ccsh_p_change(20, Nc_temp, m)
    plt.figure()
    plt.plot(p_info)
    p_info = []




plt.figure()
plt.plot([ccsh_ave_p(20,i) for i in [0,0.5,1.,1.5,2,2.5,3,3.5,4]],'rx')
plt.show()
plt.figure()
plt.plot([ccsh_prob(20,i) for i in [1.,1.5,2,2.5,3,3.5,4]],'rx')
plt.show()
N=np.arange(5,100,1)
p_ccsh=[ccsh_r(N[i],1,2) for i in range(len(N))]
p_rsh=[rsh(N[i]) for i in range(len(N))]
p_ccsh_Nc=[ccsh_r(15,i,4) for i in range(5)]
print(p_ccsh_Nc)
print([ccsh_r(N[i],0.25,2) for i in range(len(N))])
print([ccsh_r(N[i],0.5,2) for i in range(len(N))])
print([ccsh_r(N[i],1,2) for i in range(len(N))])
print([ccsh_r(N[i],2,2) for i in range(len(N))])
print([ccsh_r(N[i],3,2) for i in range(len(N))])
print(p_rsh)
plt.figure()
plt.plot(p_ccsh_Nc,'rx')

plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(N,p_ccsh,'rx')
plt.subplot(2,1,2)
plt.plot(N,p_rsh,'rx')
plt.show()

x=np.arange(0.95,1,0.00005)
#print(math.log((1-x),0.7))
y=[1/m*math.log((1-x[i]),(1-tp)) for i in range(len(x))]
print(y)
plt.figure()
plt.plot(x,y,'rx')
#plt.show()

N=np.arange(15,100,1)
y2=[1-math.pow(1-tp,m*N[i]) for i in range(len(N))]
plt.figure()
plt.plot(N,y2,'rx')
#plt.show()

Nc=3
y3=[p_ccsh(N[i],N[i]+Nc,2) for i in range(len(N))]
plt.figure()
plt.plot(N,y3,'rx')

Nc2=np.arange(1,5,1)
y4=[p_ccsh(20-Nc2[i],20,5) for i in range(len(Nc2))]
y5=[tp*20 for i in range(len(Nc2))]
plt.figure()
plt.plot(Nc2,y4,'rx')
plt.plot(Nc2,y5,'bx')
plt.show()