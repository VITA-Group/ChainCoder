
from collections import defaultdict, deque
var_0 = defaultdict((lambda : defaultdict((lambda : 0))))

def func_0(var_in_0, var_in_1, var_in_2, var_in_3):
    var_in_3.clear()
    var_1 = deque()
    var_1.append([var_in_1, float('Inf')])
    var_in_3[var_in_1] = (- 2)
    while len(var_1):
        (var_2, var_3) = var_1.var_4()
        for var_5 in var_0[var_2]:
            if ((var_in_3[var_5] == (- 1)) and (var_in_0[var_2][var_5] > 0)):
                var_in_3[var_5] = var_2
                var_3 = min(var_3, var_in_0[var_2][var_5])
                if (var_5 == var_in_2):
                    return var_3
                var_1.append((var_5, var_3))
    return 0

def func_1(var_in_0, var_in_1, var_in_2):
    var_3 = 0
    var_in_3 = defaultdict((lambda : (- 1)))
    while True:
        var_6 = func_0(var_in_0, var_in_1, var_in_2, var_in_3)
        if var_6:
            var_3 += var_6
            var_2 = var_in_2
            while (var_2 != var_in_1):
                var_7 = var_in_3[var_2]
                var_in_0[var_7][var_2] -= var_6
                var_in_0[var_2][var_7] += var_6
                var_2 = var_7
        else:
            break
    return var_3
(var_8, var_9, var_10) = [int(var_5) for var_5 in input().split()]
for var_11 in range(var_9):
    var_6 = [int(var_5) for var_5 in input().split()]
    var_0[var_6[0]][var_6[1]] = var_6[2]

def func_2(var_in_4):
    var_12 = defaultdict((lambda : defaultdict((lambda : 0))))
    for var_5 in var_0:
        for var_13 in var_0[var_5]:
            var_14 = (var_0[var_5][var_13] // var_in_4)
            var_12[var_5][var_13] = var_14
    var_3 = func_1(var_12, 1, var_8)
    return var_3
var_15 = (1 / var_10)
var_16 = func_2(1)
for var_11 in range(70):
    var_17 = ((var_16 + var_15) / 2)
    if ((var_16 - var_15) < 0.0):
        break
    if (func_2(var_17) >= var_10):
        var_15 = var_17
    else:
        var_16 = var_17
print(format((var_15 * var_10), '.9f'))



# 🟨 🟨 🟨 🟨 


from collections import defaultdict, deque
var_0 = defaultdict((lambda : defaultdict((lambda : 0))))

def func_0(var_in_0, var_in_1, var_in_2, var_in_3):
    var_in_3.clear()
    var_1 = deque()
    var_1.append([var_in_1, float('Inf')])
    var_in_3[var_in_1] = (- 2)
    while len(var_1):
        (var_2, var_3) = var_1.var_4()
        for var_5 in var_0[var_2]:
            if ((var_in_3[var_5] == (- 1)) and (var_in_0[var_2][var_5] > 0)):
                var_in_3[var_5] = var_2
                var_3 = min(var_3, var_in_0[var_2][var_5])
                if (var_5 == var_in_2):
                    return var_3
                var_1.append((var_5, var_3))
    return 0

def func_1(var_in_0, var_in_1, var_in_2):
    var_3 = 0
    var_in_3 = defaultdict((lambda : (- 1)))
    while True:
        var_6 = func_0(var_in_0, var_in_1, var_in_2, var_in_3)
        if var_6:
            var_3 += var_6
            var_2 = var_in_2
            while (var_2 != var_in_1):
                var_7 = var_in_3[var_2]
                var_in_0[var_7][var_2] -= var_6
                var_in_0[var_2][var_7] += var_6
                var_2 = var_7
        else:
            break
    return var_3
(var_8, var_9, var_10) = [int(var_5) for var_5 in input().split()]
for var_11 in range(var_9):
    var_6 = [int(var_5) for var_5 in input().split()]
    var_0[var_6[0]][var_6[1]] = var_6[2]

def func_2(var_in_4):
    var_12 = defaultdict((lambda : defaultdict((lambda : 0))))
    for var_5 in var_0:
        for var_13 in var_0[var_5]:
            var_14 = (var_0[var_5][var_13] // var_in_4)
            var_12[var_5][var_13] = var_14
    var_3 = func_1(var_12, 1, var_8)
    return var_3
var_15 = (1 / var_10)
var_16 = func_2(1)
for var_11 in range(70):
    var_17 = ((var_16 + var_15) / 2)
    if (func_2(var_17) >= var_10):
        var_15 = var_17
    else:
        var_16 = var_17
print(format((var_15 * var_10), '.9f'))



# 🟨 🟨 🟨 🟨 


from collections import defaultdict, deque

def func_0(var_in_0, var_in_1, var_in_2, var_in_3):
    var_in_3.clear()
    var_0 = deque()
    var_0.append([var_in_1, float('Inf')])
    var_in_3[var_in_1] = (- 2)
    while len(var_0):
        (var_1, var_2) = var_0.var_3()
        for var_4 in var_in_0[var_1]:
            if ((var_in_3[var_4] == (- 1)) and (var_in_0[var_1][var_4] > 0)):
                var_in_3[var_4] = var_1
                var_2 = min(var_2, var_in_0[var_1][var_4])
                if (var_4 == var_in_2):
                    return var_2
                var_0.append((var_4, var_2))
    return 0

def func_1(var_in_0, var_in_1, var_in_2):
    var_2 = 0
    var_in_3 = defaultdict((lambda : (- 1)))
    while True:
        var_5 = func_0(var_in_0, var_in_1, var_in_2, var_in_3)
        if var_5:
            var_2 += var_5
            var_1 = var_in_2
            while (var_1 != var_in_1):
                var_6 = var_in_3[var_1]
                var_in_0[var_6][var_1] -= var_5
                var_in_0[var_1][var_6] += var_5
                var_1 = var_6
        else:
            break
    return var_2
(var_7, var_8, var_9) = [int(var_4) for var_4 in input().split()]
var_in_0 = defaultdict((lambda : defaultdict((lambda : 0))))
for var_10 in range(var_8):
    var_5 = [int(var_4) for var_4 in input().split()]
    var_in_0[var_5[0]][var_5[1]] = var_5[2]

def func_2(var_in_4):
    var_11 = defaultdict((lambda : defaultdict((lambda : 0))))
    for var_4 in var_in_0:
        for var_12 in var_in_0[var_4]:
            var_13 = (var_in_0[var_4][var_12] // var_in_4)
            var_11[var_4][var_12] = var_13
    var_2 = func_1(var_11, 1, var_7)
    return var_2
var_14 = (1 / var_9)
var_15 = func_2(1)
for var_10 in range(70):
    var_16 = round(((var_15 + var_14) / 2), 9)
    if (func_2(var_16) >= var_9):
        var_14 = round(var_16, 9)
    else:
        var_15 = var_16
print(format((var_14 * var_9), '.9f'))



# 🟨 🟨 🟨 🟨 


from collections import defaultdict, deque

def func_0(var_in_0, var_in_1, var_in_2, var_in_3):
    var_in_3.clear()
    var_0 = deque()
    var_0.append([var_in_1, float('Inf')])
    var_in_3[var_in_1] = (- 2)
    while len(var_0):
        (var_1, var_2) = var_0.var_3()
        for var_4 in var_in_0[var_1]:
            if ((var_in_3[var_4] == (- 1)) and (var_in_0[var_1][var_4] > 0)):
                var_in_3[var_4] = var_1
                var_2 = min(var_2, var_in_0[var_1][var_4])
                if (var_4 == var_in_2):
                    return var_2
                var_0.append((var_4, var_2))
    return 0

def func_1(var_in_0, var_in_1, var_in_2):
    var_2 = 0
    var_in_3 = defaultdict((lambda : (- 1)))
    while True:
        var_5 = func_0(var_in_0, var_in_1, var_in_2, var_in_3)
        if var_5:
            var_2 += var_5
            var_1 = var_in_2
            while (var_1 != var_in_1):
                var_6 = var_in_3[var_1]
                var_in_0[var_6][var_1] -= var_5
                var_in_0[var_1][var_6] += var_5
                var_1 = var_6
        else:
            break
    return var_2
(var_7, var_8, var_9) = [int(var_4) for var_4 in input().split()]
var_in_0 = defaultdict((lambda : defaultdict((lambda : 0))))
for var_10 in range(var_8):
    var_5 = [int(var_4) for var_4 in input().split()]
    var_in_0[var_5[0]][var_5[1]] = var_5[2]

def func_2(var_in_4):
    var_11 = defaultdict((lambda : defaultdict((lambda : 0))))
    for var_4 in var_in_0:
        for var_12 in var_in_0[var_4]:
            var_13 = (var_in_0[var_4][var_12] // var_in_4)
            var_11[var_4][var_12] = var_13
    var_2 = func_1(var_11, 1, var_7)
    return var_2
var_14 = (1 / var_9)
var_15 = func_2(1)
for var_10 in range(70):
    var_16 = round(((var_15 + var_14) / 2), 8)
    if ((var_15 - var_14) <= 1e-07):
        break
    if (func_2(var_16) >= var_9):
        var_14 = round(var_16, 7)
    else:
        var_15 = var_16
print(format((var_14 * var_9), '.9f'))



# 🟨 🟨 🟨 🟨 


from queue import Queue

def func_0(var_in_0, var_in_1, var_in_2):
    E[var_in_0].append((len(E[var_in_1]), var_in_1, var_in_2))
    E[var_in_1].append(((len(E[var_in_0]) - 1), var_in_0, 0))

def func_1():
    nonlocal var_0, des, E, var_1
    for var_2 in range(var_3):
        var_1[var_2] = (- 1)
    var_1[var_0] = 0
    var_4 = Queue()
    var_4.var_5(var_0)
    while (not var_4.var_6()):
        cur = var_4.get()
        for var_7 in range(len(E[cur])):
            var_8 = E[cur][var_7][1]
            if ((var_1[var_8] < 0) and (E[cur][var_7][2] > 0)):
                var_1[var_8] = (var_1[cur] + 1)
                var_4.var_5(var_8)
                if (var_8 == des):
                    return True
    return False

def extend(cur, lim):
    nonlocal des, E
    if ((lim == 0) or (cur == des)):
        return lim
    var_in_2 = 0
    for var_7 in range(len(E[cur])):
        if (var_in_2 >= lim):
            break
        var_8 = E[cur][var_7][1]
        var_9 = min((lim - var_in_2), E[cur][var_7][2])
        if ((E[cur][var_7][2] > 0) and (var_1[var_8] == (var_1[cur] + 1))):
            var_10 = extend(var_8, var_9)
            if (var_10 > 0):
                E[cur][var_7] = (E[cur][var_7][0], E[cur][var_7][1], (E[cur][var_7][2] - var_10))
                var_11 = E[cur][var_7][0]
                E[var_8][var_11] = (E[var_8][var_11][0], E[var_8][var_11][1], (E[var_8][var_11][2] + var_10))
                var_in_2 += var_10
    if (var_in_2 == 0):
        var_1[cur] = (- 1)
    return var_in_2

def func_2():
    var_in_2 = 0
    var_10 = 0
    while func_1():
        var_10 = extend(var_0, var_12)
        while (var_10 > 0):
            var_in_2 += var_10
            var_10 = extend(var_0, var_12)
    return var_in_2

def func_3(var_in_3):
    nonlocal E
    E = [[] for var_2 in range(var_3)]
    for var_2 in range(var_13):
        if ((var_14[var_2] - (var_15 * var_in_3)) > 0):
            func_0(var_16[var_2], var_17[var_2], var_15)
        else:
            func_0(var_16[var_2], var_17[var_2], int((var_14[var_2] / var_in_3)))
    return (func_2() >= var_15)
(var_3, var_13, var_15) = list(map(int, input().split()))
var_12 = 1061109567
var_0 = 0
des = (var_3 - 1)
var_18 = 0.0
var_19 = 0.0
var_16 = [0 for var_2 in range(var_13)]
var_17 = [0 for var_2 in range(var_13)]
var_14 = [0 for var_2 in range(var_13)]
for var_2 in range(var_13):
    (var_16[var_2], var_17[var_2], var_14[var_2]) = list(map(int, input().split()))
    var_16[var_2] -= 1
    var_17[var_2] -= 1
    var_19 = max(var_19, var_14[var_2])
E = [[] for var_2 in range(var_3)]
var_1 = [0 for var_2 in range(var_3)]
for var_2 in range(100):
    var_in_3 = ((var_18 + var_19) / 2)
    if func_3(var_in_3):
        var_18 = var_in_3
    else:
        var_19 = var_in_3
print('{:.10f}'.format((var_in_3 * var_15)))



# 🟨 🟨 🟨 🟨 


from collections import deque

class Class_0():

    def __init__(self, listEdge, s, t):
        self.s = s
        self.t = t
        self.graph = {}
        self.maxCap = 1000000
        for var_0 in listEdge:
            if (var_0[0] not in self.graph):
                self.graph[var_0[0]] = []
            if (var_0[1] not in self.graph):
                self.graph[var_0[1]] = []
            self.graph[var_0[0]].append([var_0[1], var_0[2], len(self.graph[var_0[1]])])
            self.graph[var_0[1]].append([var_0[0], 0, (len(self.graph[var_0[0]]) - 1)])
        self.N = len(self.graph.keys())

    def bfs(self):
        self.dist = {}
        self.dist[self.s] = 0
        self.curIter = {var_1: [] for var_1 in self.graph}
        var_2 = deque([self.s])
        while (len(var_2) > 0):
            var_in_0 = var_2.var_3()
            for (index, var_0) in enumerate(self.graph[var_in_0]):
                if ((var_0[1] > 0) and (var_0[0] not in self.dist)):
                    self.dist[var_0[0]] = (self.dist[var_in_0] + 1)
                    self.curIter[var_in_0].append(index)
                    var_2.append(var_0[0])

    def findPath(self, var_in_0, var_in_1):
        if (var_in_0 == self.t):
            return var_in_1
        while (len(self.curIter[var_in_0]) > 0):
            var_4 = self.curIter[var_in_0][(- 1)]
            var_5 = self.graph[var_in_0][var_4][0]
            var_6 = self.graph[var_in_0][var_4][1]
            var_7 = self.graph[var_in_0][var_4][2]
            if ((var_6 > 0) and (self.dist[var_5] > self.dist[var_in_0])):
                var_in_2 = self.findPath(var_5, min(var_in_1, var_6))
                if (var_in_2 > 0):
                    self.path.append(var_in_0)
                    self.graph[var_in_0][var_4][1] -= var_in_2
                    self.graph[var_5][var_7][1] += var_in_2
                    return var_in_2
            self.curIter[var_in_0].pop()
        return 0

    def func_0(self):
        var_8 = 0
        var_in_2 = []
        while True:
            self.bfs()
            if (self.t not in self.dist):
                break
            while True:
                self.path = []
                var_in_1 = self.findPath(self.s, self.maxCap)
                if (var_in_1 == 0):
                    break
                var_in_2.append(var_in_1)
                var_8 += var_in_1
        return var_8

    def func_1(self):
        var_2 = deque([self.s])
        var_9 = {self.s: 's'}
        while (len(var_2) > 0):
            var_in_0 = var_2.var_3()
            for (index, var_0) in enumerate(self.graph[var_in_0]):
                if ((var_0[1] > 0) and (var_0[0] not in var_9)):
                    var_2.append(var_0[0])
                    var_9[var_0[0]] = 's'
        var_10 = []
        var_11 = []
        for var_12 in self.graph:
            if (var_12 in var_9):
                var_10.append(var_12)
            else:
                var_11.append(var_12)
        return (set(var_10), set(var_11))

def func_2(var_in_3, var_in_2, var_in_4):
    var_13 = 0
    for var_in_1 in var_in_2:
        var_13 += (var_in_1 // var_in_3)
    if (var_13 >= var_in_4):
        return True
    return False
(var_14, var_15, var_in_4) = map(int, input().split())
var_16 = [list(map(int, input().split())) for var_17 in range(var_15)]
(var_18, var_19) = (0, 1000000)
while ((var_19 - var_18) > 1e-09):
    var_20 = ((var_19 + var_18) / 2)
    var_21 = [[var_22, var_23, (var_24 // var_20)] for (var_22, var_23, var_24) in var_16]
    var_25 = Class_0(var_21, 1, var_14)
    var_8 = var_25.func_0()
    if (var_8 >= var_in_4):
        var_18 = var_20
    else:
        var_19 = var_20
print((var_in_4 * var_19))


