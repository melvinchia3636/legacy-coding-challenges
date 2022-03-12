def find_HCF(a, b):
    num = 0
    for i in range(1, max(a, b)+1):
        if a % i == 0 and b % i == 0 and i > num:
            num = i
    return num

def simplify(txt):
    a, b = txt.split('/')
    a, b = int(a), int(b)
    hcf = find_HCF(a, b)
    if a % b == 0:
        return str(a//b)
    return '/'.join([str(a//hcf), str(b//hcf)])

print(simplify('100/400'))
