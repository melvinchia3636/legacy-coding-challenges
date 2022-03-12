import string

def get_alpha(num):
	result = ''
	#print([((string.ascii_uppercase.index(j)+1), 26, i) for i, j in enumerate(reversed('ABC'))])
	curr = 0

	while num >= (27**curr):
		curr+=1
	curr = curr-1

	for i in range(curr, -1, -1):
		t = num//(26**i)
		if t>26: t=26
		result += string.ascii_uppercase[t-1]
		num -= t*26**i

	return result
	#print((3*26**0)+(2*26**1)+(1*26**2))
	

print(get_alpha(348059))