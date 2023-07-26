
import sys
input = sys.stdin.readline
var_0 = int(input())
var_1 = [list(map(int, input().split())) for var_2 in range(var_0)]
var_3 = [var_1[var_2][0] for var_2 in range(var_0)]
var_4 = [var_1[var_2][1] for var_2 in range(var_0)]
var_3.sort()
var_4.sort()
var_5 = 0
for var_2 in range(var_0):
    var_5 += max(var_3[var_2], var_4[var_2])
print((var_5 + var_0))



# 🟨 🟨 🟨 🟨 



def func_0():
    n = int(input())
    var_0 = input()
    var_1 = 0
    for var_2 in var_0:
        if (var_2 == '8'):
            var_1 += 1
    print(min(var_1, (n // 11)))

def func_1():

    def get(n):
        var_3 = 0
        while (n > 0):
            var_3 += (n % 10)
            n //= 10
        return var_3
    n = int(input())
    if (n <= 9):
        print(n)
        return
    var_4 = 9
    while (n > var_4):
        var_4 = ((var_4 * 10) + 9)
    var_4 //= 10
    print((get(var_4) + get((n - var_4))))

def func_2():
    (n, var_5, var_6, var_7) = (int(input()), [], [], 0)
    for var_2 in range(n):
        (var_8, var_9) = list(map(int, input().split()))
        var_5.append(var_8)
        var_6.append(var_9)
    var_5.sort()
    var_6.sort()
    for var_2 in range(n):
        var_7 += max(var_5[var_2], var_6[var_2])
    print((var_7 + n))
func_2()



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
(var_1, var_2) = ([], [])
for var_3 in range(var_0):
    (var_4, var_5) = list(map(int, input().split()))
    var_1.append(var_4)
    var_2.append(var_5)
var_1.sort(reverse=True)
var_2.sort(reverse=True)
var_6 = var_0
for var_7 in range(var_0):
    var_6 += max(var_1[var_7], var_2[var_7])
print(var_6)



# 🟨 🟨 🟨 🟨 


var_0 = [[], []]
var_1 = ([[var_0[var_2].append(int(var_3)) for (var_2, var_3) in enumerate(input().split())] for var_1 in range(int(input()))], [var_3.sort() for var_3 in var_0], print((len(var_0[0]) + sum((max(var_3, var_4) for (var_3, var_4) in zip(*var_0))))))



# 🟨 🟨 🟨 🟨 


var_0 = (lambda var_in_0: ([[var_in_0[var_1].append(int(var_2)) for (var_1, var_2) in enumerate(input().split())] for var_0 in range(int(input()))], [var_2.sort() for var_2 in var_in_0], print((len(var_in_0[0]) + sum((max(var_2, var_3) for (var_2, var_3) in zip(*var_in_0)))))))([[], []])



# 🟨 🟨 🟨 🟨 


print(sum([(max(*var_0) + 1) for var_0 in zip(*list(map(sorted, list(zip(*[list(map(int, input().split())) for var_1 in range(int(input()))])))))]))



# 🟨 🟨 🟨 🟨 


var_0 = 0
var_1 = int(input())
var_2 = []
var_3 = []
for var_4 in range(var_1):
    (var_5, var_6) = list(map(int, input().split()))
    var_2.append(var_5)
    var_3.append(var_6)
var_2.sort()
var_3.sort()
for var_4 in range(0, var_1):
    var_0 += (max(var_2[var_4], var_3[var_4]) + 1)
print(var_0)



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    (var_4, var_5) = list(map(int, input().split()))
    var_1.append(var_4)
    var_2.append(var_5)
var_1.sort()
var_2.sort()
print((var_0 + sum(map(max, var_1, var_2))))



# 🟨 🟨 🟨 🟨 


import heapq
var_0 = int(input())
var_1 = [var_2 for var_2 in range(var_0)]
var_3 = []
var_4 = []
for var_2 in range(var_0):
    (var_5, var_6) = [int(var_7) for var_7 in input().split()]
    var_3.append((var_5, var_2))
    var_4.append((var_6, var_2))
var_3.sort()
var_4.sort()
var_8 = var_0
for var_2 in range(var_0):
    var_8 += max(var_3[var_2][0], var_4[var_2][0])
print(var_8)



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    (var_4, var_5) = [int(var_6) for var_6 in input().strip().split()]
    var_1.append(var_4)
    var_2.append(var_5)
var_1.sort()
var_2.sort()
print((var_0 + sum([max(var_1[var_6], var_2[var_6]) for var_6 in range(var_0)])))



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    var_4 = [int(var_5) for var_5 in input().split(' ')]
    var_1.append(var_4[0])
    var_2.append(var_4[1])
var_1.sort()
var_2.sort()
var_6 = 0
for var_3 in range(var_0):
    var_6 = ((var_6 + max(var_1[var_3], var_2[var_3])) + 1)
print(var_6)



# 🟨 🟨 🟨 🟨 


from sys import stdin
var_0 = int(10000.0)
var_1 = int((- 10000.0))

def func_0():
    return int(stdin.readline())

def func_1():
    return [int(var_2) for var_2 in stdin.readline().split()]

def func_2():
    return input()

def func_3():
    return [var_2 for var_2 in stdin.readline().split()]
var_3 = func_0()
(var_4, var_5) = ([], [])
for var_6 in range(var_3):
    (var_7, var_8) = func_1()
    var_4.append(var_7)
    var_5.append(var_8)
var_4 = sorted(var_4)
var_5 = sorted(var_5)
var_9 = var_3
for var_10 in range(var_3):
    var_9 += max(var_4[var_10], var_5[var_10])
print(var_9)



# 🟨 🟨 🟨 🟨 


import sys

def func_0():
    return map(int, sys.stdin.readline().split())
var_0 = int(input())
var_1 = ([0] * var_0)
var_2 = ([0] * var_0)
for var_3 in range(var_0):
    (var_1[var_3], var_2[var_3]) = func_0()
var_1.sort()
var_2.sort()
var_4 = 0
for var_3 in range(var_0):
    var_4 += max(var_1[var_3], var_2[var_3])
print((var_4 + var_0))



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    (var_4, var_5) = map(int, input().split())
    var_1.append(var_4)
    var_2.append(var_5)
var_2 = sorted(var_2)
var_1 = sorted(var_1)
var_6 = var_0
for var_3 in range(var_0):
    var_6 += max(var_1[var_3], var_2[var_3])
print(var_6)



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = [0 for var_2 in range(var_0)]
var_3 = [0 for var_2 in range(var_0)]
for var_2 in range(var_0):
    [var_1[var_2], var_3[var_2]] = map(int, input().split())
var_1.sort()
var_3.sort()
var_4 = var_0
for var_2 in range(var_0):
    var_4 += max(var_1[var_2], var_3[var_2])
print(var_4)



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(0, var_0):
    (var_4, var_5) = map(int, input().split())
    var_1.append(var_4)
    var_2.append(var_5)
var_1.sort()
var_2.sort()
var_6 = var_0
for var_3 in range(0, var_0):
    var_6 += max(var_1[var_3], var_2[var_3])
print(var_6)



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    var_4 = [int(var_5) for var_5 in input().split()]
    (var_6, var_7) = var_4
    var_1.append(var_6)
    var_2.append(var_7)
var_1.sort()
var_2.sort()
var_8 = [max(var_9, var_10) for (var_9, var_10) in zip(var_1, var_2)]
print((var_0 + sum(var_8)))



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    (var_4, var_5) = list(map(int, input().split(' ')))
    var_1 += [var_4]
    var_2 += [var_5]
var_1.sort()
var_2.sort()
var_6 = 0
for var_3 in range(var_0):
    var_6 += 1
    var_6 += max(var_1[var_3], var_2[var_3])
print(var_6)



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    (var_4, var_5) = map(int, input().split(' '))
    var_1 += [var_4]
    var_2 += [var_5]
var_1.sort()
var_2.sort()
var_6 = 0
for var_3 in range(var_0):
    var_6 += 1
    var_6 += max(var_1[var_3], var_2[var_3])
print(var_6)



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    (var_4, var_5) = map(int, input().split())
    var_1.append(var_4)
    var_2.append(var_5)
var_1.sort()
var_2.sort()
print((var_0 + sum(map(max, var_1, var_2))))



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    (var_4, var_5) = map(int, input().split())
    var_1.append(var_4)
    var_2.append(var_5)
print((var_0 + sum(map(max, sorted(var_1), sorted(var_2)))))



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    (var_4, var_5) = input().split()
    var_1.append(int(var_4))
    var_2.append(int(var_5))
var_1 = sorted(var_1)
var_2 = sorted(var_2)
var_6 = var_0
for var_3 in range(var_0):
    var_6 += max(var_1[var_3], var_2[var_3])
print(var_6)



# 🟨 🟨 🟨 🟨 



def func_0():
    return list(map(int, input().split()))
var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    (var_4, var_5) = func_0()
    var_1.append(var_4)
    var_2.append(var_5)
var_1.sort()
var_2.sort()
var_6 = var_0
for var_7 in range(var_0):
    var_6 += max(var_1[var_7], var_2[var_7])
print(var_6)



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
var_1 = []
var_2 = []
for var_3 in range(var_0):
    (var_4, var_5) = map(int, input().split())
    var_1.append(var_4)
    var_2.append(var_5)
print((var_0 + sum(map(max, sorted(var_1), sorted(var_2)))))


