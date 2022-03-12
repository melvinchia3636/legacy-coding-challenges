import string

def abacaba_pattern(n):
    letters = string.ascii_uppercase
    count = 1
    lst = 'A'
    while n-1 > 0:
        lst = lst + letters[count] + lst
        n-=1
        count += 1
    return lst

def ABA(s):
    letters = string.ascii_uppercase
    count = 1
    lst = 'A'
    n = letters.index(s)+1
    while n-1 > 0:
        lst = lst + letters[count] + lst
        n-=1
        count += 1
    return lst

print(ABA("O"))
