import math

def squares(a, b):
    lowest = int(math.ceil(math.sqrt(a)))
    highest = int(math.sqrt(b))
    return len([n**2 for n in range(lowest, highest + 1)])

print(squares(50979851, 733216221))
