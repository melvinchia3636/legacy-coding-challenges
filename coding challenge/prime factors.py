def prime_factors(num):
    a = []
    while num % 2 == 0:
        a.append(2)
        num /= 2
    f = 3
    while f * f <= num:
        if num % f == 0:
            a.append(int(f))
            num /= f
        else:
            f += 2
    if num != 1: a.append(int(num))
    return a

print(prime_factors(8912234))
