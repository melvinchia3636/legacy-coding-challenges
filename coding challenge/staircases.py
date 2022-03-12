def staircase(n):
    return '\n'.join([('_'*(n-i)+'#'*i) for i in range(1, n+1) if n>0]) or '\n'.join([('_'*i+'#'*(abs(n)-i)) for i in range(abs(n)) if n<0])

print(staircase(3))
