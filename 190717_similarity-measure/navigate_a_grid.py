import sys

result = []
for i in range(1):
    mn=list(map(int,sys.stdin.readline().strip().split()))
    m=mn[0]
    n=mn[1]
    seq=[]
    if m != 0 and n != 0:
        for line in range(m):
            values=list(map(int,sys.stdin.readline().strip().split()))
            seq.append(values)
        print(seq)
    else:
        pass