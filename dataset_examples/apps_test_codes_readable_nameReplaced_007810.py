

def func_0():
    return list(map(int, input().split()))

def func_1(var_in_0, var_in_1=1):
    (var_0, var_1, var_2) = ([var_in_0], ([0] * var_3), (['+'] * var_4))
    var_1[var_in_0] = 1
    while var_0:
        var_5 = var_0.pop()
        for (var_6, var_7) in var_8[var_5]:
            if (var_1[var_6] == 0):
                if ((var_7 * var_in_1) < 0):
                    var_2[(abs(var_7) - 1)] = '-'
                elif ((var_7 * var_in_1) > 0):
                    var_2[(abs(var_7) - 1)] = '+'
                if ((var_in_1 == 1) or (var_7 == 0)):
                    var_0.append(var_6)
                    var_1[var_6] = 1
    return (''.join(var_2), sum(var_1))
(var_3, var_4, var_0) = func_0()
var_8 = [[] for var_5 in range(var_3)]
var_7 = 1
for var_9 in range(var_4):
    (var_10, var_in_0, var_11) = func_0()
    (var_in_0, var_11) = ((var_in_0 - 1), (var_11 - 1))
    if (var_10 == 1):
        var_8[var_in_0].append((var_11, 0))
    else:
        var_8[var_in_0].append((var_11, var_7))
        var_8[var_11].append((var_in_0, (- var_7)))
        var_7 += 1
var_4 = (var_7 - 1)
(var_in_0, var_11) = func_1((var_0 - 1), 1)
print(var_11)
print(var_in_0)
(var_in_0, var_11) = func_1((var_0 - 1), (- 1))
print(var_11)
print(var_in_0)



# 🟨 🟨 🟨 🟨 


import sys
input = sys.stdin.readline

def func_0():
    return list(map(int, input().split()))

def func_1(var_in_0, var_in_1=1):
    (var_0, var_1, var_2) = ([var_in_0], ([0] * var_3), (['+'] * var_4))
    var_1[var_in_0] = 1
    while var_0:
        var_5 = var_0.pop()
        for (var_6, var_7) in var_8[var_5]:
            if (var_1[var_6] == 0):
                if ((var_7 * var_in_1) < 0):
                    var_2[(abs(var_7) - 1)] = '-'
                elif ((var_7 * var_in_1) > 0):
                    var_2[(abs(var_7) - 1)] = '+'
                if ((var_in_1 == 1) or (var_7 == 0)):
                    var_0.append(var_6)
                    var_1[var_6] = 1
    return (''.join(var_2), sum(var_1))
(var_3, var_4, var_0) = func_0()
var_8 = [[] for var_5 in range(var_3)]
var_7 = 1
for var_9 in range(var_4):
    (var_10, var_in_0, var_11) = func_0()
    (var_in_0, var_11) = ((var_in_0 - 1), (var_11 - 1))
    if (var_10 == 1):
        var_8[var_in_0].append((var_11, 0))
    else:
        var_8[var_in_0].append((var_11, var_7))
        var_8[var_11].append((var_in_0, (- var_7)))
        var_7 += 1
var_4 = (var_7 - 1)
(var_in_0, var_11) = func_1((var_0 - 1), 1)
print(var_11)
print(var_in_0)
(var_in_0, var_11) = func_1((var_0 - 1), (- 1))
print(var_11)
print(var_in_0)



# 🟨 🟨 🟨 🟨 


import sys
input = sys.stdin.readline

def func_0():
    return list(map(int, input().split()))

def func_1(var_in_0):
    var_0 = [var_in_0]
    var_1 = ([0] * var_2)
    var_3 = (['+'] * var_4)
    var_1[var_in_0] = 1
    while var_0:
        var_5 = var_0.pop()
        for (var_6, var_7) in var_8[var_5]:
            if (var_1[var_6] == 0):
                if (var_7 < 0):
                    var_3[((- var_7) - 1)] = '-'
                elif (var_7 > 0):
                    var_3[(var_7 - 1)] = '+'
                var_0.append(var_6)
                var_1[var_6] = 1
    return (''.join(var_3), sum(var_1))

def func_2(var_in_0):
    var_0 = [var_in_0]
    var_1 = ([0] * var_2)
    var_3 = (['+'] * var_4)
    var_1[var_in_0] = 1
    while var_0:
        var_5 = var_0.pop()
        for (var_6, var_7) in var_8[var_5]:
            if (var_1[var_6] == 0):
                if (var_7 < 0):
                    var_3[((- var_7) - 1)] = '+'
                elif (var_7 > 0):
                    var_3[(var_7 - 1)] = '-'
                if (var_7 == 0):
                    var_0.append(var_6)
                    var_1[var_6] = 1
    return (''.join(var_3), sum(var_1))
(var_2, var_4, var_0) = func_0()
var_8 = [[] for var_5 in range(var_2)]
var_7 = 1
for var_9 in range(var_4):
    (var_10, var_in_0, var_11) = func_0()
    (var_in_0, var_11) = ((var_in_0 - 1), (var_11 - 1))
    if (var_10 == 1):
        var_8[var_in_0].append((var_11, 0))
    else:
        var_8[var_in_0].append((var_11, var_7))
        var_8[var_11].append((var_in_0, (- var_7)))
        var_7 += 1
var_4 = (var_7 - 1)
(var_in_0, var_11) = func_1((var_0 - 1))
print(var_11)
print(var_in_0)
(var_in_0, var_11) = func_2((var_0 - 1))
print(var_11)
print(var_in_0)


