times = 0

def valid(n):

    if n >= 0:
        return True
    return False

def only_5_and_3(n):

    global times

    n = (n-(5*times))
    if n % 3 == 0 and valid(n):
        return True
    times += 1
    only_5_and_3(n)

print(only_5_and_3(91))
