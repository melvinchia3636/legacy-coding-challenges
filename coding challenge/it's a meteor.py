#will_hit("y = 2x - 5", (0, 0)) ➞ False
#will_hit("y = -4x + 6", (1, 2)) ➞ True

def will_hit(equation, position):
    splitted_equation = equation.split(' ')
    [splitted_equation.pop(0) for i in range(2)]
    splitted_equation[0] = int(splitted_equation[0].replace('x', ''))
    splitted_equation[2] = int(splitted_equation[2])
    if splitted_equation[1] == '+' and position[1] == splitted_equation[0]*position[0]+splitted_equation[2]:
            return True
    if splitted_equation[1] == '-' and position[1] == splitted_equation[0]*position[0]-splitted_equation[2]:
            return True
    return False

print(will_hit("y = -4x + 6", (1, 2)))
