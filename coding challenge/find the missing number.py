import math

num = [1, 2, 3, 4, 5, 6, 7, 9]
num2 = [6, 7, 8, 9, 1, 2, 3, 4, 5]

def find_miss(lst):
    return sum(range(min(lst), max(lst)+1)) - sum(lst)

def find_index(lst, num):
    m = len(lst)//2
    l = lst[:m]
    r = lst[m:]
    print(l, r)
    if num > l[0]:
        

find_index(num2, 2)
