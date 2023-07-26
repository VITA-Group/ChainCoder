import sys
input = sys.stdin.readline

n=int(input())
lr=[list(map(int,input().split())) for i in range(n)]

L=[lr[i][0] for i in range(n)]
R=[lr[i][1] for i in range(n)]
L.sort()
R.sort()

ANS=0

for i in range(n):
    ANS+=max(L[i],R[i])

print(ANS+n)


# 🟨 🟨 🟨 🟨 

def mainA():
    n = int(input())
    s = input()
    cnt = 0
    for i in s:
        if i == '8':
            cnt += 1
    print(min(cnt, n // 11))

def mainB():
    def get(n):
        ret = 0
        while n > 0:
            ret += n % 10
            n //= 10
        return ret
    
    n = int(input())
    if n <= 9:
        print(n)
        return
    t = 9
    while n > t:
        t = t * 10 + 9
    t //= 10
    print(get(t) + get(n - t))


def mainD():
    n, A, B, ans = int(input()), [], [], 0
    for i in range(n):
        a, b = list(map(int, input().split()))
        A.append(a)
        B.append(b)
    A.sort()
    B.sort()
    for i in range(n):
        ans += max(A[i], B[i])
    print(ans + n)

mainD()


# 🟨 🟨 🟨 🟨 

n = int(input())
ps,qs = [], []
for _ in range(n):
    p,q = list(map(int, input().split()))
    ps.append(p)
    qs.append(q)

ps.sort(reverse=True)
qs.sort(reverse=True)
res = n
for i in range(n):
    res += max(ps[i], qs[i])
print(res)


# 🟨 🟨 🟨 🟨 

a = [[], []] 
_ = ([[a[i].append(int(x)) for i, x in enumerate(input().split())] for _ in range(int(input()))], [x.sort() for x in a], print(len(a[0]) + sum(max(x, y) for x, y in zip(*a))))


# 🟨 🟨 🟨 🟨 

_ = (lambda a : ([[a[i].append(int(x)) for i, x in enumerate(input().split())] for _ in range(int(input()))], [x.sort() for x in a], print(len(a[0]) + sum(max(x, y) for x, y in zip(*a)))))([[], []])


# 🟨 🟨 🟨 🟨 

print(sum([max(*x)+1 for x in zip(*list(map(sorted,list(zip(*[list(map(int,input().split())) for _ in range(int(input()))])))))]))


# 🟨 🟨 🟨 🟨 

ans=0
n=int(input())
a=[]
b=[]
for i in range(n):
    x,y=list(map(int,input().split()))
    a.append(x)
    b.append(y)
a.sort()
b.sort()
#print(a)
for i in range(0,n):
    ans+=max(a[i],b[i])+1
print(ans)
        


# 🟨 🟨 🟨 🟨 

n = int(input())
a = []
b = []
for _ in range(n) :
    x, y = list(map(int, input().split()))
    a.append(x)
    b.append(y)
a.sort()
b.sort()
print(n + sum(map(max, a, b)))


# 🟨 🟨 🟨 🟨 

import heapq
n=int(input())
fa=[i for i in range(n)]
ls=[]
rs=[]
for i in range(n):
    l,r=[int(x) for x in input().split()]
    ls.append((l,i))
    rs.append((r,i))
ls.sort()
rs.sort()
ans=n
for i in range(n):
    ans+=max(ls[i][0],rs[i][0])
# heapq.heapify(ls)
# heapq.heapify(rs)
#
# ans=n
# if n==1:
#     print(max(ls[0][0],rs[0][0])+1)
#     quit()
# for i in range(n):
#     ll=heapq.heappop(ls)
#     if fa[rs[0][1]]!=fa[ll[1]]:
#         rr=heapq.heappop(rs)
#         fa[ll[1]]=rr[1]
#     else:
#         tem=heapq.heappop(rs)
#         rr=heapq.heappop(rs)
#         fa[ll[1]]=rr[1]
#         heapq.heappush(rs,tem)
#     ans+=max(ll[0],rr[0])
print(ans)

# 🟨 🟨 🟨 🟨 

n = int(input())
a = []
b = []
for i in range(n):
  l, r = [int(_) for _ in input().strip().split()]
  a.append(l)
  b.append(r)

a.sort()
b.sort()

print(n + sum([max(a[_],b[_]) for _ in range(n)]))


# 🟨 🟨 🟨 🟨 

n=int(input())

a=[]
b=[]
for i in range(n):
	inp=[int(x) for x in input().split(" ")]
	a.append(inp[0])
	b.append(inp[1])

a.sort()
b.sort()

ans=0

for i in range(n):
	ans=ans+max(a[i],b[i])+1

print(ans)

# 🟨 🟨 🟨 🟨 


# -*- coding: utf-8 -*-
# @Date    : 2018-10-02 08:00:37
# @Author  : raj lath (oorja.halt@gmail.com)
# @Link    : link
# @Version : 1.0.0

from sys import stdin

max_val=int(10e12)
min_val=int(-10e12)

def read_int()     : return int(stdin.readline())
def read_ints()    : return [int(x) for x in stdin.readline().split()]
def read_str()     : return input()
def read_strs()    : return [x for x in stdin.readline().split()]


nb_guest = read_int()
left, rite = [], []
for _ in range(nb_guest):
    a, b = read_ints()
    left.append(a)
    rite.append(b)
left = sorted(left)
rite = sorted(rite)
answ = nb_guest
for i in range(nb_guest):
    answ  += max(left[i] , rite[i])
print(answ)




# 🟨 🟨 🟨 🟨 

#!/usr/bin/env python3
import sys

def rint():
    return map(int, sys.stdin.readline().split())
#lines = stdin.readlines()

n = int(input())
r = [0]*n
l = [0]*n

for i in range(n):
    r[i], l[i] = rint()

r.sort()
l.sort()

ans = 0

for i in range(n):
    ans += max(r[i], l[i])

print(ans + n)

# 🟨 🟨 🟨 🟨 

n=int(input())
l=[]
r=[]
for i in range(n):
    a,b=map(int,input().split())
    l.append(a)
    r.append(b)
r=sorted(r)
l=sorted(l)
ss=n
for i in range(n):
    ss+=max(l[i],r[i])
print(ss)

# 🟨 🟨 🟨 🟨 

n = int(input())
l = [0 for i in range(n)]
r = [0 for i in range(n)]

for i in range(n):
    [l[i], r[i]] = map(int, input().split())

l.sort()
r.sort()

res = n
for i in range(n):
    res += max(l[i], r[i])
print(res)

# 🟨 🟨 🟨 🟨 

n = int(input())

l = []
r = []

for i in range(0, n):
	x, y = map(int, input().split())
	l.append(x)
	r.append(y)

l.sort()
r.sort()

res = n

for i in range(0, n):
	res += max(l[i], r[i])

print(res)

# 🟨 🟨 🟨 🟨 

n = int(input())

l = []
r = []

for i in range(n):
    numbers_in_line = [int(num) for num in input().split()]
    l_new, r_new = numbers_in_line
    l.append(l_new)
    r.append(r_new)

l.sort()
r.sort()

maxes = [max(lv, rv) for lv, rv in zip(l, r)]

print(n + sum(maxes))


# 🟨 🟨 🟨 🟨 

n = int(input())
leftSpaces = []
rightSpaces = []
for i in range(n):
	left,right = list(map(int,input().split(" ")))
	leftSpaces += [left]
	rightSpaces += [right]
leftSpaces.sort()
rightSpaces.sort()
chairs = 0
for i in range(n):
	chairs += 1
	chairs += max(leftSpaces[i],rightSpaces[i])
print(chairs)




# 🟨 🟨 🟨 🟨 

n = int(input())
leftSpaces = []
rightSpaces = []
for i in range(n):
	left,right = map(int,input().split(" "))
	leftSpaces += [left]
	rightSpaces += [right]
leftSpaces.sort()
rightSpaces.sort()
chairs = 0
for i in range(n):
	chairs += 1
	chairs += max(leftSpaces[i],rightSpaces[i])
print(chairs)

# 🟨 🟨 🟨 🟨 

n = int(input())
a = []
b = []
for _ in range(n) :
    x, y = map(int, input().split())
    a.append(x)
    b.append(y)
a.sort()
b.sort()
print(n + sum(map(max, a, b)))

# 🟨 🟨 🟨 🟨 

n = int(input())
l = []
r = []
for _ in range(n):
    x, y = map(int, input().split())
    l.append(x)
    r.append(y)
print(n+sum(map(max, sorted(l), sorted(r))))

# 🟨 🟨 🟨 🟨 

n = int(input())
left = []
right = []

for i in range(n):
    l, r = input().split()
    left.append(int(l))
    right.append(int(r))

left = sorted(left)
right = sorted(right)

res = n

for i in range(n):
    res += max(left[i], right[i])

print(res)


# 🟨 🟨 🟨 🟨 

def get_input_list():
	return list(map(int, input().split()))
n = int(input())
l = []
r = []
for _ in range(n):
	li, ri = get_input_list()
	l.append(li)
	r.append(ri)
l.sort()
r.sort()
res = n
for i in range(n):
	res += max(l[i],r[i])
print(res)

# 🟨 🟨 🟨 🟨 

## Problem @ http://codeforces.com/problemset/problem/1060/D
## #greedy #math
n = int(input())
left = []
right = []
for i in range(n):
    a,b = map(int, input().split())
    left.append(a)
    right.append(b)
print(n + sum(map(max, sorted(left), sorted(right))))
