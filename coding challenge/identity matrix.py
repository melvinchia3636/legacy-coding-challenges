def id_mtrx(n):
    try:
        final = [[0 for j in range(abs(n))] for i in range(abs(n))]
        for i in range(abs(n)):
            if n > 0: final[i][i] = 1
            else: final[i][-1-i] = 1
        return final
    except: return 'Error'

lst = (id_mtrx(-10))

[print(i) for i in lst]
