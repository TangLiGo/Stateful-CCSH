import math
import argparse
def d2b(n):
    b=''
    while True:
        s=n//2
        y=n%2
        b=b+str(y)
        if s==0:
            break
        n=s
    while len(b)<32:
        b=b+str(0)
    return b[::-1]

def operation(data,n,m):
    label = 1
    if data<0:
        label=-1
    b=d2b(abs(data))


   # print(b)
    b_core=b[32-n-m:32-n]
    b_core2=b[:31-n+m]+b[32-n:]
    b_new=b_core+b_core2
    print(b)
    print(b_new)
    return label*int(b_new[:16],2)
def getOutput(li,n,m):
    out=[]
    for data in li:
        out.append(operation(data,n,m))
    return max(out,key=out.count)

input_datas=[-16,-1,-273,-22]
n=4
m=4
getOutput(input_datas,n,m)

print(getOutput(input_datas,4,4))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.description='Please enter X n m :'
    parser.add_argument("-X","--inputX",help="1<=X<=2000 ",type=int,dest="X")
    parser.add_argument("-n", "--inputn", help="0<=n<=31 ", type=int, dest="n")
    parser.add_argument("-m", "--inputm", help="0<=m<=31 ", type=int, dest="m")
    args=parser.parse_args()
    X=args.X
    n=args.n
    m=args.m
    parser2 = argparse.ArgumentParser()
    parser2.description = 'Please enter 32 bits values:'
    parser2.add_argument("-l", "--list", help="Enter values ", type=int, dest="l")
    args2=parser2.parse_args()
    datas=args2.l
    print(datas)
   # print(X,n,m,getOutput())