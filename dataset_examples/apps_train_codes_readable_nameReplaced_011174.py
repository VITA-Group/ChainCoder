
import sys
input = sys.stdin.readline
var_0 = (lambda : list(map(int, input().split())))
for var_1 in range(int(input())):
    (var_2, var_3) = var_0()
    var_4 = 0
    var_5 = 0
    for var_6 in var_0():
        if (var_3 == var_6):
            var_4 = 1
        var_5 = max(var_5, var_6)
    if var_4:
        print(1)
    else:
        print(max(2, (((var_3 - 1) // var_5) + 1)))



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
for var_1 in range(var_0):
    (var_2, var_3) = list(map(int, input().split()))
    var_4 = [int(var_5) for var_5 in input().split()]
    var_4.sort()
    if (var_4[(- 1)] > var_3):
        if (var_4.count(var_3) == 0):
            print(2)
        else:
            print(1)
    elif ((var_3 % var_4[(- 1)]) == 0):
        print((var_3 // var_4[(- 1)]))
    else:
        print(((var_3 // var_4[(- 1)]) + 1))



# 🟨 🟨 🟨 🟨 


for var_0 in range(int(input())):
    (var_1, var_2) = list(map(int, input().split()))
    var_3 = list(map(int, input().split()))
    var_4 = (((var_2 + max(var_3)) - 1) // max(var_3))
    if ((var_4 == 1) and (var_2 not in var_3)):
        var_4 = 2
    print(var_4)



# 🟨 🟨 🟨 🟨 



def func_0():
    (var_0, var_1) = list(map(int, input().split()))
    var_2 = list(map(int, input().split()))
    var_2.sort()
    for var_3 in range(var_0):
        if (var_2[var_3] == var_1):
            print('1')
            return
    for var_3 in range(var_0):
        if (var_2[var_3] >= var_1):
            print('2')
            return
    print(int((((var_1 - 1) / var_2[(var_0 - 1)]) + 1)))
for var_4 in range(int(input())):
    func_0()



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
for var_1 in range(var_0):
    (var_2, var_3) = map(int, input().split())
    var_4 = list(map(int, input().split()))
    print(min((((((var_3 + var_5) - 1) // var_5) if (var_5 <= var_3) else 2) for var_5 in var_4)))



# 🟨 🟨 🟨 🟨 



def func_0():
    (var_0, var_1) = list(map(int, input().split()))
    var_2 = set(map(int, input().split()))
    var_3 = max(var_2)
    var_4 = (((var_1 + var_3) - 1) // var_3)
    if ((var_4 == 1) and (var_1 not in var_2)):
        var_4 = 2
    print(var_4)
var_5 = int(input())
for var_6 in range(var_5):
    func_0()



# 🟨 🟨 🟨 🟨 


var_0 = int(input())
for var_1 in range(var_0):
    (var_2, var_3) = map(int, input().split())
    var_4 = list(map(int, input().split()))
    var_5 = 10000.0
    for var_6 in var_4:
        var_7 = 0
        if ((var_3 % var_6) != 0):
            var_7 += 1
        var_7 += (var_3 // var_6)
        if (((var_3 // var_6) == 0) and ((var_3 % var_6) != 0)):
            var_7 += 1
        var_5 = min(var_5, var_7)
    print(int(var_5))


