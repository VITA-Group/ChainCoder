def put():
  return list(map(int, input().split()))

def dfs(x,flag=1):
    s,vis,ans   = [x],[0]*n,['+']*m
    vis[x]= 1
    while s:
        i = s.pop()
        for j,k in graph[i]:
            if vis[j]==0:
                if k*flag<0:
                    ans[abs(k)-1]='-'
                elif k*flag>0:
                    ans[abs(k)-1]='+'
                if flag==1 or k==0:
                    s.append(j)
                    vis[j]=1
    return ''.join(ans), sum(vis)

n,m,s = put()
graph = [[] for i in range(n)]
k=1
for _ in range(m):
    z,x,y = put()
    x,y = x-1,y-1
    if z==1:
        graph[x].append((y, 0))
    else:
        graph[x].append((y, k))
        graph[y].append((x,-k))
        k+=1
m = k-1
x,y = dfs(s-1, 1)
print(y)
print(x)
x,y = dfs(s-1,-1)
print(y)
print(x)


# 🟨 🟨 🟨 🟨 

import sys
input = sys.stdin.readline
def put():
  return list(map(int, input().split()))

def dfs(x,flag=1):
    s,vis,ans   = [x],[0]*n,['+']*m
    vis[x]= 1
    while s:
        i = s.pop()
        for j,k in graph[i]:
            if vis[j]==0:
                if k*flag<0:
                    ans[abs(k)-1]='-'
                elif k*flag>0:
                    ans[abs(k)-1]='+'
                if flag==1 or k==0:
                    s.append(j)
                    vis[j]=1
    return ''.join(ans), sum(vis)

n,m,s = put()
graph = [[] for i in range(n)]
k=1
for _ in range(m):
    z,x,y = put()
    x,y = x-1,y-1
    if z==1:
        graph[x].append((y, 0))
    else:
        graph[x].append((y, k))
        graph[y].append((x,-k))
        k+=1
m = k-1
x,y = dfs(s-1, 1)
print(y)
print(x)
x,y = dfs(s-1,-1)
print(y)
print(x)


# 🟨 🟨 🟨 🟨 

import sys
input = sys.stdin.readline

def put():
  return list(map(int, input().split()))




def dfs0(x):
  s = [x]
  vis = [0] * n
  ans = ['+'] * m
  vis[x] = 1
  while s:
    i = s.pop()
    for j, k in graph[i]:
      if (vis[j] == 0):
        if (k < 0):
          ans[-k - 1] = '-'
        elif (k > 0):
          ans[k - 1] = '+'
        
          
        s.append(j)
        vis[j] = 1

  return ''.join(ans), sum(vis)

def dfs1(x):
  s = [x]
  vis = [0] * n
  ans = ['+'] * m
  vis[x] = 1
  while s:
    i = s.pop()
    for j, k in graph[i]:
      if (vis[j] == 0):
        if (k < 0):
          ans[-k - 1] = '+'
        elif (k > 0):
          ans[k - 1] = '-'
        if (k == 0):
          s.append(j)
          vis[j] = 1

  return ''.join(ans), sum(vis)

        

  




n,m,s = put()
graph = [[] for i in range(n)]

k = 1

for _ in range(m):
  z,x,y = put()
  x,y = x - 1, y - 1
  if (z == 1):
    graph[x].append((y, 0))
  else:
    graph[x].append((y, k))
    graph[y].append((x, -k))
    k += 1

m = k - 1
x, y = dfs0(s - 1)
print(y)
print(x)
x, y = dfs1(s - 1)
print(y)
print(x)


