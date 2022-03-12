import math
from itertools import zip_longest, combinations, islice, groupby
import string
from hashlib import sha256
from collections import Counter
from copy import deepcopy
import re
import random
from typing import List, Dict, Union

class Test(object):
	@staticmethod
	def assert_equals(first, second, desc=None):
		print(desc) if desc else None
		print('Output: {}'.format(first))
		print('Expect: {}'.format(second))
		print('Correct: {}'.format(first == second))
		print()

def ones_infection(a):
	t=deepcopy(a)
	for i in range(len(a)):
		for j in range(len(a[i])):
			if any(t[i])or any(list(zip(*t))[j]):a[i][j]=1
	return a
def deepest(l):
	c=1
	while sum([[i]if not isinstance(i,list)else i for i in l],[])!=l:c+=1;l=sum([[i]if not isinstance(i,list)else i for i in l],[])
	return c
def is_val_in_tree(t,v):
	if t[0]==v:return 1
	for i in t:
		if i and isinstance(i,list)and is_val_in_tree(i,v):return 1
	return 0
def min_length(l,n):
	try:return len(min([i for i in sum([[l[j:j+i]for j in range(0,len(l)-i+1)]for i in range(1,len(l)+1)],[])if sum(i)>n],key=len))
	except:return-1
def deep_count(l,c=0):
	for i in l:
		if isinstance(i,list):c+=deep_count(i)
		c+=1
	return c
def parking_exit(p):
	C='L';B='R';A='{}{}';r,c,f=[],[0,p[0].index(2)],[len(p)-1,len(p[0])-1]
	while c!=f:
		if p[c[0]][c[1]]==1:d=0;l=locals();exec('while p[c[0]][c[1]] == 1: c[0] += 1; d+=1',l);r.append('D{}'.format(l['d']))
		elif c[0]==len(p)-1:t=len(p[-1])-1-c[1];r.append(A.format(B if t>0 else C,abs(t)));c[1]=len(p[-1])-1
		else:t=p[c[0]].index(1)-c[1];r.append(A.format(B if t>0 else C,abs(t)));c[1]=p[c[0]].index(1)
	return r
def lemonade(b,c=0):
	for i in b:
		if c<i-5:return 0
		c-=i-10
	return 1
def get_length(l, c=0):
	for i in l: c+= 1 if not isinstance(i, list) else get_length(i)
	return c
def lychrel(n):
	for i in range(500):
		if str(n) == str(n)[::-1]: return i
		n += int(str(n)[::-1])
	return 1
def islands_perimeter(s): s,g = [[0]*(len(s[0])+2)]+[[0]+i+[0] for i in s]+[[0]*(len(s[0])+2)], lambda y, x: [i for i in [(y+1, x), (y-1, x), (y, x+1), (y, x-1)] if 0 <= i[1] < len(s[0]) and 0 <= i[0] < len(s) and not s[i[0]][i[1]]]; return len(sum(sum([[g(i, j) for j in range(len(s[i])) if s[i][j]] for i in range(len(s))], []), []))
def rolling_cipher(t, n):w = string.ascii_lowercase; return ''.join(w[w.index(i)+n if w.index(i)+n < 26 else w.index(i)+n-26] for i in t)
def divide(l, n): r = [[]]; [r[-1].append(l[i]) if sum(r[-1]+[l[i]]) <= n else r.append([l[i]]) for i in range(len(l))]; return r
def convert_to_roman(n): r, n = 'IVXLCDM', list(map(int, str(n)))[::-1]; return ''.join(r[i*2]*n[i] if 0<n[i]<4 else r[i*2]*(n[i]-5)+r[i*2+1] if 5<n[i]<9 else r[i*2+1]+r[i*2] if n[i]==4 else r[i*2+2]+r[i*2] if n[i]==9 else '' for i in range(len(n)))[::-1]
def generate_rug(n, d):l = [list(range(n)[i::-1])+list(range(n)[1:n-i:]) for i in range(0, n)];return  l if d=='left' else [i[::-1] for i in l]
def simon_says(l): t = '0'+''.join(j+')' if j.isdigit() else '+-*'['asm'.index(j[0])] for i in l if i.startswith('Simon says') for j in i.split()[2:] if j != 'by'); return eval('('*t.count(')')+t)
def boxes(w, k=1, c=0):
	for i in w: c, k = (c+i, k) if c+i<=10 else (i, k+1)
	return k
def num_then_char(l): i=iter(sorted([i for i in sum(l,[]) if type(i) in [int, float]])+sorted([i for i in sum(l,[]) if type(i) == str])); return [list(islice(i, j)) for j in [len(k) for k in l]]
def check_pattern(l, p): l = [[str(j[1]) for j in i[1]] for i in groupby(sorted(list(zip(p, l))), lambda x: x[0])]; return all(len(i) == 1 for i in [set(i) for i in l]) and len(set([str(i) for i in [set(i) for i in l]])) == len(set(p)) 
def order_people(a, p): l = [list(range(1, p+1)[i:i+a[1]]) for i in range(0, p, a[1])]; l = [i+[0]*(a[1]-len(i)) for i in l] + [[0]*a[1]]*(a[0]-len(l)); l = [l[i][::-1] if i%2 else l[i] for i in range(len(l))]; return l if a[0]*a[1] >= p else 'overcrowded'
def has_consecutive_series(v1, v2): l = [sum(i) for i in zip_longest(v1, v2, fillvalue=0)]; return sum(range(min(l), max(l)+1)) == sum(l)

def help_bobby(s):  
    a = [[0 for i in range(s)] for i in range(s)]
    for i in range(s): a[i][i] = a[i][-i-1] = 1
    return a

dif_ciph, group, classify_rug, k_th_binary_inlist, people_sort, longest_run, get_notes_distribution, bw_transform, get_sha256_hash, is_self_describing, is_unfair_hurdle, printgrid, checkout, rearrange, chosen_wine, champions, three_sum, flatten_list, find_missing, factorial, transpose_matrix, leaderboards, get_length, scrambled, pricey_prod, count_ones, bridge_shuffle, vertical_txt, can_traverse, cutting_grass,transform_matrix, largest_even, kix_code, spotlight_map, floyd, special_reverse_string, equal_count = lambda s: [ord(s[0])]+[ord(s[i])-ord(s[i-1]) for i in range(1, len(s))] if type(s) == str else chr(s[0])+''.join([chr(sum(s[:i])+s[i]) for i in range(1, len(s))]), lambda l, s: [list([j for j in i if j]) for i in list(zip_longest(*[l[i:i+math.ceil(len(l)/s)] for i in range(0, len(l), math.ceil(len(l)/s))]))], lambda p: 'perfect' if all(i==i[::-1] for i in p) and all(i==tuple(reversed(i)) for i in zip(*p)) else 'vertically symmetric' if all(i==i[::-1] for i in p) else 'horizontally symmetric' if all(i==tuple(reversed(i)) for i in zip(*p)) else 'imperfect', lambda n, k: sorted(reversed([bin(i) for i in range(2**n)]), key=lambda x: x.count('1'))[::-1][k-1], lambda l, a: sorted(l, key=lambda x: eval('x.'+a)), lambda l: max(len(max(''.join(str(int(l[i-1]+1==l[i])) for i in range(1, len(l))).split('0'), key=len))+1, len(max(''.join(str(int(l[i-1]-1==l[i])) for i in range(1, len(l))).split('0'), key=len))+1), lambda s: Counter(j for i in s for j in i['notes'] if j in range(1, 6)), lambda t:''.join(i[-1] for i in sorted(t[i:]+t[:i] for i in range(len(t)))), lambda t: sha256(t.encode('utf-8')).hexdigest(), lambda n: len(str(n)) % 2 == 0 and all(int(x) == str(n).count(y) for x, y in (str(n)[i:i+2] for i in range(0, len(str(n)), 2))), lambda h:len(h)>=4 or h[0].count(' ')/(h[0].count('#')-1)<4, lambda r, c: [list(i) for i in zip(*[range(1, r*c+1)[i:i+r] for i in range(0, r*c, r)])], lambda c: sum(i['prc']*i['qty']*(1.06 if i['taxable'] else 1) for i in c), lambda s: ' '.join(i[1] for i in sorted((re.search(r'\d', i).group(), i.replace(re.search(r'\d', i).group(), '')) for i in s.split())), lambda w: sorted(w, key=lambda x: x['price'])[1]['name'] if len(w) > 1 else w[0]['name'] if w else None, lambda t: sorted([(i['wins']*3+i['draws'], i['scored']-i['conceded'], i['name']) for i in t])[-1][-1], lambda l: list(map(list, dict.fromkeys(sorted([tuple(i) for i in list(combinations(l, 3)) if not sum(i)], key=lambda x: l.index(x[0]))))), lambda l: sum(map(flatten_list,l),[]) if isinstance(l,list) else [l], lambda l: sum(range(len(min(l, key=len)), len(max(l, key=len))+1)) - sum(len(i) for i in l) if l and all(l) else 0, lambda n: n*factorial(n-1) if n>0 else 1, lambda l: list(map(list, zip(*l))), lambda u: sorted(u, key=lambda k: k['score']+k['reputation']*2, reverse=1), lambda l: sum(map(get_length, l)) if isinstance(l, list) else 1 , lambda w, m: [i for i in w if re.match("^"+m.replace('*', r'\w')+"$", i)], lambda d: sorted([i for i in d if d[i] >= 500], key=lambda k: -d[k]), lambda l: len([i for i in ''.join(map(str, l)).split('0') if len(i)>=2]), lambda m, n: [i for i in sum(zip_longest(m, n), ()) if i], lambda t: [list(i) for i in zip_longest(*t.split(), fillvalue=' ')], lambda x: all(abs(i-j) < 2 for i, j in zip([sum(i) for i in zip(*x)], [sum(i) for i in zip(*x)][1:])), lambda l, *c: [i if sum(i) >= len(i) else 'Done' for i in [[j-sum(c[0:i+1]) for j in l] for i in range(len(c))]], lambda l: [[sum(l[i])+sum(list(zip(*l))[j])-l[i][j]*2 for j in range(len(l[0]))] for i in range(len(l))], lambda l, n=0, c=-1:  c if n >= len(l) else largest_even(l, n+1, l[n] if l[n]%2==0 and l[n]>c else c), lambda a: next(map(lambda l: l[1]+l[2]+re.sub(r'[^\d\w]', 'X', l[0]), re.findall(r'.(?=\d)(?<=\s)(.*?),\s(\d{4})\s(.{2})', a))).upper(), lambda g:[[sum(map(lambda k:g[B+k[0]][C+k[1]]if 0<=B+k[0]<len(g)and 0<=C+k[1]<len(g[0])else 0,[(0,0),(1,0),(0,1),(1,1),(-1,0),(0,-1),(-1,-1),(-1,1),(1,-1)]))for C in range(len(g[B]))]for B in range(len(g))], lambda up_to=None, n_row=None: (n:=1) and [list(range(n, (n:=n+i))) for i in range(1, n_row+1 if n_row else (math.ceil((math.sqrt(8*up_to+1)-1)/2)+1))], lambda t: [(reverse := list(t.replace(' ', ''))[::-1]), [reverse.insert(i, ' ') for i in [i[0] for i in list(enumerate(list(' '.join([('*'*i) for i in [len(i) for i in t.split(' ')]])))) if i[1] == ' ']], ''.join([[i.lower() for i in reverse][i].upper() if ['u' if i.isupper() else 'l' if i.islower() else 'o' for i in t][i] == 'u' else [i.lower() for i in reverse][i].lower() if ['u' if i.isupper() else 'l' if i.islower() else 'o' for i in t][i] == 'l' else [i.lower() for i in reverse][i] for i in range(len(reverse))])][-1], lambda t, m: [(m := m.split('&')), {**dict(zip(m, [t.count(n) for n in m])), **{'equality':[t.count(n) for n in m][0] == [t.count(n) for n in m][1]}, **{'difference':(max([t.count(n) for n in m]) - min([t.count(n) for n in m]))}} if [t.count(n) for n in m][0] != [t.count(n) for n in m][1] else {**dict(zip(m, [t.count(n) for n in m])), **{'equality':[t.count(n) for n in m][0] == [t.count(n) for n in m][1]}}][-1]

def num_regions(g):
	def visit(i, j):
		if (i, j) in u:
			u.remove((i, j))
			if g[i][j]: 
				visit(i, j-1)
				visit(i, j+1)
				visit(i-1, j)
				visit(i+1, j)
	u,c = [(i, j) for i in range(len(g)) for j in range(len(g[0]))],0
	while u:
		i, j = random.choice(u)
		if g[i][j]: c += 1; visit(i, j)
		else: u.remove((i, j))
	return c

def vending_machine(p: list, m: int, n: int) -> Union[dict, str]:
	#if user enter invalid product number
	if n not in [i['number'] for i in p]: return 'Enter a valid product number'
	#if user have no enough money to but the product
	if m < p[n-1]['price']: return 'Not enough money for this product'
	#if user give just enough money to buy the product
	if m == p[n-1]['price']: c: List[int] = []
	#if user give more than the product price to buy the product
	else: 
		c = [] #the changes that will soon be returned to user
		s = iter([500, 200, 100, 50, 20, 10]) #the available coins value
		m = m-p[n-1]['price'] #the total changes that needed to be returned to user
		while m > 0: #while we still have remaining change:
			t = next(s) #next coin value
			c += [t]*(m//t) #put the right amount of coins into the change list
			m -= (m//t)*t #deduct the values of coins that has been put into the changes list from the remaining changes

	return {'product': p[n-1]['name'], 'c': c} #return the final result

gn = lambda i, j, l, m, n: [i for i in ([(i+1, j), (i-1, j), (i, j+1), (i, j-1)] if n=='+' else [(i+1, j-1), (i-1, j+1), (i+1, j+1), (i-1, j-1)]) if 0<=i[0]<l and 0<=i[1]<m]
def all_explode(g, c = (0, 0), f=1):
	b = g[c[0]][c[1]]
	if str(b) in '+x':
		g[c[0]][c[1]] = 0
		[all_explode(g, i, 0) for i in gn(*c, len(g), len(g[0]), b)]
	if f: return all(not any(str(j)!='0' for j in i) for i in g)

Test.assert_equals(boxes([7, 1, 2, 6, 1, 2, 3, 5, 9, 2, 1, 2, 5]), 5)
Test.assert_equals(boxes([2, 7, 1, 3, 3, 4, 7, 4, 1, 8, 2]), 5)
Test.assert_equals(boxes([1, 3, 3, 3, 2, 1, 1, 9, 7, 10, 8, 6, 1, 2, 9]), 8)
Test.assert_equals(boxes([4, 1, 2, 3, 5, 5, 1, 9]), 3)
Test.assert_equals(boxes([3, 1, 2, 7, 2, 6, 1]), 3)
Test.assert_equals(boxes([4, 6, 1, 9, 6, 1, 1, 9, 2, 9]), 6)
Test.assert_equals(boxes([2, 2, 10, 10, 1, 5, 2]), 4)
Test.assert_equals(boxes([9, 6, 2, 3, 1, 2, 4, 8, 3, 1, 3]), 5)
Test.assert_equals(boxes([2, 5, 1, 6, 2, 9, 5, 2, 1, 6, 1, 6, 6, 1]), 7)
Test.assert_equals(boxes([1, 2, 3, 2, 6, 4, 1]), 3)
Test.assert_equals(boxes([1, 1, 2, 1, 2, 10, 2, 2, 5, 1, 5]), 4)
Test.assert_equals(boxes([8, 3, 2, 1, 1, 2, 1, 3, 2, 1]), 3)
Test.assert_equals(boxes([1, 5, 3, 1, 2, 3, 2, 6, 3, 1, 3, 8, 1]), 5)
Test.assert_equals(boxes([8, 1, 1, 4, 7, 2, 1, 3, 1, 9, 7, 1, 5, 1, 1]), 7)
Test.assert_equals(boxes([2, 3, 4, 10, 1, 2, 5, 1, 1, 1, 1, 8, 2, 1]), 5)
Test.assert_equals(boxes([4, 6, 7, 3, 2, 2, 3, 1, 2, 2, 10, 3, 2]), 6)
Test.assert_equals(boxes([9, 2, 3, 4, 1, 3, 1, 1, 3]), 3)
Test.assert_equals(boxes([6, 2, 1, 9, 1, 8, 2, 8, 6, 6]), 6)
Test.assert_equals(boxes([6, 9, 3, 8, 10, 4, 7]), 7)
Test.assert_equals(boxes([4, 3, 1, 1, 1, 4, 7, 2, 1, 10, 1, 3, 8]), 6)
Test.assert_equals(boxes([10]), 1)

M='Jake'
L='Marcus'
K='Mark'
J='Sarah'
I='Jay'
H='Joel'
G='Jacob'
F='Joshua'
E='Joseph'
D='Adam'
C='Kevin'
B='notes'
A='name'
Test.assert_equals(get_notes_distribution([{A:'Steve',B:[5,5,3,-1,6]},{A:'John',B:[3,2,5,0,-3]}]),{5:3,3:2,2:1})
Test.assert_equals(get_notes_distribution([{A:F,B:[2,-2,4,5,-3]},{A:C,B:[-3,2,-1,1,-3]},{A:F,B:[5,-1,-1,4,5]},{A:I,B:[4,4,1,3,-2]},{A:L,B:[4,4,3,-1,-2]},{A:D,B:[3,3,-1,5,3]}]),{2:2,4:6,5:4,1:2,3:5})
Test.assert_equals(get_notes_distribution([{A:E,B:[2,-1,2,4,4]},{A:F,B:[4,1,-3,0,1]},{A:G,B:[3,5,0,4,2]},{A:H,B:[2,1,5,0,1]},{A:K,B:[2,2,4,4,5]},{A:G,B:[3,5,-3,0,4]},{A:G,B:[3,0,-2,2,0]}]),{2:7,4:7,1:4,3:3,5:4})
Test.assert_equals(get_notes_distribution([{A:E,B:[0,2,0,3,4]},{A:L,B:[2,1,-2,-2,-3]},{A:E,B:[-2,-3,-1,4,1]}]),{2:2,3:1,4:2,1:2})
Test.assert_equals(get_notes_distribution([{A:C,B:[0,3,-2,0,4]},{A:G,B:[0,-3,2,2,3]},{A:C,B:[5,3,-2,-3,0]},{A:J,B:[3,1,3,4,-2]}]),{3:5,4:2,2:2,5:1,1:1})
Test.assert_equals(get_notes_distribution([{A:F,B:[2,1,2,0,2]}]),{2:3,1:1})
Test.assert_equals(get_notes_distribution([{A:C,B:[5,1,0,-2,-1]},{A:I,B:[-1,1,5,0,2]},{A:K,B:[1,5,2,3,-2]},{A:C,B:[1,2,2,-3,2]}]),{5:3,1:4,2:5,3:1})
Test.assert_equals(get_notes_distribution([{A:K,B:[5,-3,-3,-3,0]},{A:E,B:[0,-2,-1,5,-3]}]),{5:2})
Test.assert_equals(get_notes_distribution([{A:L,B:[-3,0,4,1,3]},{A:F,B:[0,-3,-1,0,1]},{A:D,B:[-3,4,2,-3,-3]}]),{4:2,1:2,3:1,2:1})
Test.assert_equals(get_notes_distribution([{A:D,B:[3,-1,-3,1,-2]},{A:D,B:[5,2,5,2,3]},{A:H,B:[0,4,-2,3,-1]},{A:E,B:[-1,5,-2,0,-2]}]),{3:3,1:1,5:3,2:2,4:1})
Test.assert_equals(get_notes_distribution([{A:L,B:[-3,-2,2,2,2]},{A:H,B:[-3,1,4,3,4]},{A:C,B:[0,-1,4,1,-3]},{A:D,B:[-1,1,2,2,2]},{A:D,B:[4,0,-1,-2,-1]}]),{2:6,1:3,4:4,3:1})
Test.assert_equals(get_notes_distribution([{A:G,B:[0,-1,2,-3,4]},{A:D,B:[-2,5,1,1,2]},{A:C,B:[-3,0,2,-3,-2]},{A:D,B:[5,-1,3,5,1]},{A:H,B:[1,-3,-2,2,-3]},{A:C,B:[2,-3,4,3,0]}]),{2:5,4:2,5:3,1:4,3:2})
Test.assert_equals(get_notes_distribution([{A:C,B:[-1,-1,-2,-3,0]},{A:L,B:[-1,-1,3,5,1]},{A:J,B:[3,0,4,-1,-3]},{A:I,B:[-3,-2,0,0,0]},{A:H,B:[3,4,-3,1,0]},{A:G,B:[3,5,1,4,4]},{A:E,B:[2,0,-1,-2,-3]}]),{3:4,5:2,1:3,4:4,2:1})
Test.assert_equals(get_notes_distribution([{A:E,B:[1,2,3,3,-3]},{A:H,B:[0,5,5,5,2]},{A:F,B:[5,4,2,0,3]}]),{1:1,2:3,3:3,5:4,4:1})
Test.assert_equals(get_notes_distribution([{A:D,B:[3,5,5,4,4]},{A:J,B:[4,0,-1,0,5]},{A:G,B:[-3,-1,0,-1,-2]},{A:F,B:[-3,5,5,-1,3]}]),{3:2,5:5,4:3})
Test.assert_equals(get_notes_distribution([{A:F,B:[-3,-2,2,4,5]}]),{2:1,4:1,5:1})
Test.assert_equals(get_notes_distribution([{A:H,B:[-2,5,-2,3,3]},{A:E,B:[3,4,-2,-1,2]},{A:M,B:[2,4,0,-3,-3]}]),{5:1,3:3,4:2,2:2})
Test.assert_equals(get_notes_distribution([{A:I,B:[-1,2,1,2,-1]},{A:I,B:[3,1,-3,0,1]},{A:J,B:[0,-2,1,4,3]},{A:I,B:[-3,0,5,5,0]},{A:C,B:[5,-2,5,-1,1]},{A:J,B:[-3,-3,4,1,-2]},{A:D,B:[2,4,-1,0,4]},{A:H,B:[1,0,-1,-1,-2]}]),{2:3,1:7,3:2,4:4,5:4})
Test.assert_equals(get_notes_distribution([{A:C,B:[-2,0,4,0,2]},{A:E,B:[1,1,-1,-1,-2]},{A:L,B:[1,-3,5,3,-1]},{A:C,B:[2,-3,-1,4,-3]},{A:G,B:[-3,5,-3,5,-1]}]),{4:2,2:2,1:3,5:3,3:1})
Test.assert_equals(get_notes_distribution([{A:G,B:[-2,3,4,3,4]},{A:K,B:[-1,-1,2,1,5]},{A:I,B:[5,5,2,3,2]},{A:C,B:[0,-2,5,4,4]},{A:M,B:[0,5,0,4,3]},{A:F,B:[5,4,5,-3,2]},{A:K,B:[-1,5,2,3,5]}]),{3:5,4:6,2:5,1:1,5:9})
Test.assert_equals(get_notes_distribution([{A:C,B:[5,-3,-3,3,3]},{A:D,B:[5,1,3,1,-1]},{A:C,B:[1,5,2,2,-3]},{A:D,B:[-2,2,2,5,3]},{A:H,B:[1,-2,5,2,4]},{A:J,B:[3,4,3,3,-2]},{A:D,B:[-3,3,-1,-1,0]}]),{5:5,3:8,1:4,2:5,4:2})
Test.assert_equals(get_notes_distribution([{A:J,B:[2,-1,4,-3,-2]}]),{2:1,4:1})
Test.assert_equals(get_notes_distribution([{A:E,B:[3,-1,5,4,-3]}]),{3:1,5:1,4:1})
Test.assert_equals(get_notes_distribution([{A:D,B:[0,4,3,-3,-2]},{A:D,B:[3,0,4,-2,-1]},{A:H,B:[5,3,-1,-3,0]},{A:G,B:[-1,1,5,0,0]},{A:F,B:[0,1,-2,0,3]}]),{4:2,3:4,5:2,1:2})
Test.assert_equals(get_notes_distribution([{A:E,B:[2,-3,1,-1,1]},{A:I,B:[4,0,-3,4,4]},{A:I,B:[-1,2,2,5,3]},{A:H,B:[3,5,2,0,-2]},{A:G,B:[-1,4,1,0,-1]},{A:F,B:[0,4,-3,-3,1]}]),{2:4,1:4,4:5,5:2,3:2})
Test.assert_equals(get_notes_distribution([{A:K,B:[-2,-3,-2,4,-1]},{A:M,B:[5,3,-3,3,1]},{A:F,B:[3,-1,-3,0,5]},{A:C,B:[-1,4,-2,-1,0]},{A:F,B:[-3,-3,-1,4,1]}]),{4:3,5:2,3:3,1:2})
Test.assert_equals(get_notes_distribution([{A:E,B:[0,-1,-3,1,5]},{A:D,B:[-2,-1,2,4,-2]},{A:L,B:[1,-1,1,0,3]},{A:E,B:[5,0,-2,0,2]},{A:G,B:[-2,-3,-1,-1,0]},{A:C,B:[-3,0,2,2,4]}]),{1:3,5:2,2:4,4:2,3:1})
Test.assert_equals(get_notes_distribution([{A:M,B:[-1,0,-2,-1,1]},{A:G,B:[1,4,3,-1,4]},{A:E,B:[-1,3,-2,-3,2]},{A:J,B:[-1,-1,-2,-3,-1]},{A:H,B:[1,4,3,-3,-2]},{A:C,B:[4,0,1,5,2]},{A:I,B:[1,-1,0,4,0]},{A:D,B:[-1,1,3,-1,1]}]),{1:7,4:5,3:4,2:2,5:1})
Test.assert_equals(get_notes_distribution([{A:D,B:[-3,1,-2,-1,3]}]),{1:1,3:1})
Test.assert_equals(get_notes_distribution([{A:K,B:[-2,2,3,-2,2]},{A:M,B:[5,-3,2,5,-1]},{A:H,B:[3,-3,4,1,-3]},{A:C,B:[4,3,0,-1,-3]},{A:G,B:[-3,-3,5,5,1]},{A:M,B:[3,5,2,-3,5]},{A:E,B:[0,1,0,1,-2]},{A:K,B:[2,-3,-1,-3,2]},{A:F,B:[0,4,-2,0,0]}]),{2:6,3:4,5:6,4:3,1:4})
Test.assert_equals(get_notes_distribution([{A:J,B:[4,-2,5,1,0]},{A:C,B:[-1,4,0,-2,4]}]),{4:3,5:1,1:1})

Test.assert_equals(deep_count([1, 2, 3]), 3)
Test.assert_equals(deep_count(["x", "y", ["z"]]), 4)
Test.assert_equals(deep_count(["a", "b", ["c", "d", ["e"]]]), 7)
Test.assert_equals(deep_count([[1], [2], [3]]), 6)
Test.assert_equals(deep_count([[[[[[[[[]]]]]]]]]), 8)
Test.assert_equals(deep_count([None]), 1)
Test.assert_equals(deep_count([[]]), 1)
Test.assert_equals(deep_count([[None], [0, ["edabit"]], [0]]), 8)

Test.assert_equals(is_self_describing(10123331), 1, "Example #1")
Test.assert_equals(is_self_describing(224444), 1, "Example #2")
Test.assert_equals(is_self_describing(2211), 0, "Example #3")
Test.assert_equals(is_self_describing(333), 0, "Example #4")
Test.assert_equals(is_self_describing(1), 0)
Test.assert_equals(is_self_describing(27273332), 1)
Test.assert_equals(is_self_describing(11), 0)
Test.assert_equals(is_self_describing(22), 1)
Test.assert_equals(is_self_describing(19212332), 1)
Test.assert_equals(is_self_describing(4444332231), 0)
Test.assert_equals(is_self_describing(881722888888), 1)

Test.assert_equals(rolling_cipher('abcd', 1), 'bcde')
Test.assert_equals(rolling_cipher('abcd', -1), 'zabc')
Test.assert_equals(rolling_cipher('abcd', 3), 'defg')
Test.assert_equals(rolling_cipher('abcd', 25), 'zabc')
Test.assert_equals(rolling_cipher('abcd', 26), 'abcd')
Test.assert_equals(rolling_cipher('abcd', 27), 'bcde')
Test.assert_equals(rolling_cipher('abcd', 0), 'abcd')

Test.assert_equals(bw_transform("banana$"), "annb$aa")
Test.assert_equals(bw_transform("mississippi$"), "ipssm$pissii")
Test.assert_equals(bw_transform("abaaba$"), "abba$aa")
Test.assert_equals(bw_transform("acccgtttgtttcaatagatccatcaa$"), "aacc$tacgttctaccatcaatatttgg")

Test.assert_equals(get_sha256_hash("hi"), "8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4")
Test.assert_equals(get_sha256_hash("password123"), "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f")
Test.assert_equals(get_sha256_hash("don't use easy passwords"), "9fdfef802f06e384101080935fd3c938c60f92f528d520528b5c0491471a2be1")
	
Test.assert_equals(divide([1, 2, 3, 4, 1, 0, 2, 2], 5), [[1, 2], [3], [4, 1, 0], [2, 2]])
Test.assert_equals(divide([1, 2, 3, 4, 1, 0, 2, 2], 4), [[1, 2], [3], [4], [1, 0, 2], [2]])
Test.assert_equals(divide([1, 3, 2, -1, 2, 1, 1, 3, 1], 3), [[1], [3], [2, -1, 2], [1, 1], [3], [1]])
Test.assert_equals(divide([1, 2, 2, -1, 2, 0, 1, 0, 1], 2), [[1], [2], [2, -1], [2, 0], [1, 0, 1]])
Test.assert_equals(divide([1, 2, 2, -1, 2, 0, 1, 0, 1], 3), [[1, 2], [2, -1, 2, 0], [1, 0, 1]])
Test.assert_equals(divide([1, 2, 2, -1, 2, 0, 1, 0, 1], 5), [[1, 2, 2, -1], [2, 0, 1, 0, 1]])
Test.assert_equals(divide([2, 1, 0, -1, 0, 0, 2, 1, 3], 3), [[2, 1, 0, -1, 0, 0], [2, 1], [3]])
Test.assert_equals(divide([2, 1, 0, -1, 0, 0, 2, 1, 3], 4), [[2, 1, 0, -1, 0, 0, 2], [1, 3]])
Test.assert_equals(divide([1, 0, 1, 1, -1, 0, 0], 1), [[1, 0], [1], [1, -1, 0, 0]])
Test.assert_equals(divide([1, 0, 1, 1, -1, 0, 0], 2), [[1, 0, 1], [1, -1, 0, 0]])
Test.assert_equals(divide([1, 0, 1, 1, -1, 0, 0], 3), [[1, 0, 1, 1, -1, 0, 0]])

G='#      #      #      #'
F='#    #    #'
E='#  #  #  #'
D='########'
C='#    #    #    #    #'
B='#    #    #    #'
A='#    #'
Test.assert_equals(is_unfair_hurdle([B,B,B,B]),1)
Test.assert_equals(is_unfair_hurdle([E,E,E]),1)
Test.assert_equals(is_unfair_hurdle([B,B,B]),0)
Test.assert_equals(is_unfair_hurdle([B,B]),0)
Test.assert_equals(is_unfair_hurdle([G,G]),0)
Test.assert_equals(is_unfair_hurdle([D,D,D,D,D]),1)
Test.assert_equals(is_unfair_hurdle([A,A]),0)
Test.assert_equals(is_unfair_hurdle([A,A,A]),0)
Test.assert_equals(is_unfair_hurdle([F,F,F]),0)
Test.assert_equals(is_unfair_hurdle([C,C,C]),0)
Test.assert_equals(is_unfair_hurdle([C,C,C,C,C]),1)
Test.assert_equals(is_unfair_hurdle([A,A,A,A,A]),1)

Test.assert_equals(printgrid(3, 6), [[1, 4, 7, 10, 13, 16], [2, 5, 8, 11, 14, 17], [3, 6, 9, 12, 15, 18]])
Test.assert_equals(printgrid(5, 3), [[1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14], [5, 10, 15]])
Test.assert_equals(printgrid(4, 1), [[1], [2], [3], [4]])
Test.assert_equals(printgrid(1, 3), [[1, 2, 3]])
Test.assert_equals(printgrid(10, 2), [[1, 11], [2, 12], [3, 13], [4, 14], [5, 15], [6, 16], [7, 17], [8, 18], [9, 19], [10, 20]])

G='paper plates'
F='potato chips'
E='soda'
D='taxable'
C='qty'
B='prc'
A='desc'
Test.assert_equals(checkout([{A:'grill',B:300,C:1,D:1},{A:'hotdogs',B:10,C:2,D:0},{A:'US Flag',B:30,C:1,D:1}]),369.8)
Test.assert_equals(checkout([{A:'hamburger',B:5,C:2,D:0},{A:'potato salad',B:8,C:1,D:0},{A:F,B:2,C:2,D:0},{A:E,B:3,C:2,D:0},{A:G,B:5,C:1,D:1}]),33.3)
Test.assert_equals(checkout([{A:'beach umbrella',B:58,C:1,D:1},{A:'beach towel',B:9,C:2,D:1},{A:'swim suit',B:25,C:2,D:0},{A:E,B:3,C:2,D:0},{A:'cooler',B:25,C:1,D:1}]),163.06)
Test.assert_equals(checkout([{A:F,B:2,C:2,D:0},{A:E,B:3,C:2,D:0},{A:G,B:5,C:1,D:1}]),15.3)

Test.assert_equals(convert_to_roman(2), "II")
Test.assert_equals(convert_to_roman(12),"XII")
Test.assert_equals(convert_to_roman(16), "XVI")
Test.assert_equals(convert_to_roman(44), "XLIV")
Test.assert_equals(convert_to_roman(68), "LXVIII")
Test.assert_equals(convert_to_roman(400), "CD")
Test.assert_equals(convert_to_roman(798), "DCCXCVIII")
Test.assert_equals(convert_to_roman(1000), "M")
Test.assert_equals(convert_to_roman(3999),"MMMCMXCIX")
Test.assert_equals(convert_to_roman(649), "DCXLIX")

Test.assert_equals(rearrange("is2 Thi1s T4est 3a"), "This is a Test")
Test.assert_equals(rearrange("4of Fo1r pe6ople g3ood th5e the2"), "For the good of the people")
Test.assert_equals(rearrange(" "), "")

Test.assert_equals(chosen_wine([{"name": "Wine A", "price": 8.99}, {"name": "Wine 32", "price": 13.99}, {"name": "Wine 9", "price": 10.99}]), "Wine 9")
Test.assert_equals(chosen_wine([{"name": "Wine A", "price": 8.99}, {"name": "Wine B", "price": 9.99}]), "Wine B")
Test.assert_equals(chosen_wine([{"name": "Wine A", "price": 8.99}]), "Wine A")
Test.assert_equals(chosen_wine([]), None)
Test.assert_equals(chosen_wine([{"name": "Wine A", "price": 8.99}, {"name": "Wine 389", "price": 109.99}, {"name": "Wine 44", "price": 38.44}, {"name": "Wine 72", "price": 22.77}]), "Wine 72")

A,B,C,D,E,F,G,H,I,J,K,L = 'name','wins','loss','draws','scored','conceded','Arsenal','Manchester City','Liverpool','Chelsea','Leicester City','Manchester United'
Test.assert_equals(champions([{A:L,B:30,C:3,D:5,E:88,F:20},{A:G,B:24,C:6,D:8,E:98,F:29},{A:J,B:22,C:8,D:8,E:98,F:29}]),L)
Test.assert_equals(champions([{A:H,B:30,C:8,D:0,E:67,F:20},{A:I,B:34,C:2,D:2,E:118,F:29},{A:K,B:22,C:8,D:8,E:98,F:29}]),I)
Test.assert_equals(champions([{A:H,B:30,C:8,D:0,E:67,F:20},{A:'New Castle United',B:34,C:2,D:2,E:118,F:36},{A:K,B:34,C:2,D:2,E:108,F:21}]),K)
Test.assert_equals(champions([{A:H,B:30,C:6,D:2,E:102,F:20},{A:I,B:24,C:6,D:8,E:118,F:29},{A:G,B:28,C:2,D:8,E:87,F:39}]),H)
Test.assert_equals(champions([{A:H,B:30,C:6,D:2,E:102,F:20},{A:I,B:24,C:6,D:8,E:118,F:29},{A:G,B:30,C:0,D:8,E:87,F:39}]),G)
Test.assert_equals(champions([{A:J,B:35,C:3,D:0,E:102,F:20},{A:I,B:24,C:6,D:8,E:118,F:29},{A:G,B:28,C:2,D:8,E:87,F:39}]),J)

Test.assert_equals(three_sum([0, 1, -1, -1, 2]), [[0, 1, -1], [-1, -1, 2]])
Test.assert_equals(three_sum([0, 0, 0, 5, -5]), [[0, 0, 0], [0, 5, -5]])
Test.assert_equals(three_sum([0, -1, 1, 0, -1, 1]), [[0, -1, 1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [1, 0, -1]])
Test.assert_equals(three_sum([0, 5, 5, 0, 0]), [[0, 0, 0]])
Test.assert_equals(three_sum([0, 5, -5, 0, 0]), [[0, 5, -5], [0, 0, 0], [5, -5, 0]])
Test.assert_equals(three_sum([1, 2, 3, -5, 8, 9, -9, 0]), [[1, 8, -9], [2, 3, -5], [9, -9, 0]])
Test.assert_equals(three_sum([0, 0, 0]), [[0, 0, 0]])
Test.assert_equals(three_sum([1, 5, 5, 2]), [])
Test.assert_equals(three_sum([1, 1]), [])
Test.assert_equals(three_sum([]), [])

B='right'
A='left'
Test.assert_equals(generate_rug(4,A),[[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]])
Test.assert_equals(generate_rug(5,B),[[4,3,2,1,0],[3,2,1,0,1],[2,1,0,1,2],[1,0,1,2,3],[0,1,2,3,4]])
Test.assert_equals(generate_rug(6,A),[[0,1,2,3,4,5],[1,0,1,2,3,4],[2,1,0,1,2,3],[3,2,1,0,1,2],[4,3,2,1,0,1],[5,4,3,2,1,0]])
Test.assert_equals(generate_rug(1,A),[[0]])
Test.assert_equals(generate_rug(2,B),[[1,0],[0,1]])

Test.assert_equals(flatten_list([1, '2', [3, [4]]]), [1, '2', 3, 4])
Test.assert_equals(flatten_list([1]), [1])
Test.assert_equals(flatten_list([]), [])

Test.assert_equals(parking_exit([[1, 0, 0, 0, 2], [0, 0, 0, 0, 0]]), ["L4", "D1", "R4"])
Test.assert_equals(parking_exit([[2, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]), ["R3", "D2", "R1"])
Test.assert_equals(parking_exit([[0, 2, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]), ["R3", "D3"])
Test.assert_equals(parking_exit([[1, 0, 0, 0, 2], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), ["L4", "D1", "R4", "D1", "L4", "D1", "R4"])
Test.assert_equals(parking_exit([[0, 0, 0, 0, 2]]), [])

Test.assert_equals(simon_says(['Simon says add 4','Simon says add 6','Then add 5']),10)
Test.assert_equals(simon_says(['Susan says add 10','Simon says add 3','Simon says multiply by 8']),24)
Test.assert_equals(simon_says(['Firstly, add 4','Simeon says subtract 100']),0)

Test.assert_equals(simon_says(["Simeon says subtract 46", "Firstly, multiply by 3", "Simon says add 18", "Then subtract 50", "Next, multiply by 2", "Then add 17", "Simeon says multiply by 43", "Now add 13", "Now subtract 32", "Firstly, multiply by 35", "Simon says subtract 22", "Joshua says subtract 48", "Simon says subtract 45", "Simon says add 7", "Simon says add 25", "Simeon says add 13"]), -17)
Test.assert_equals(simon_says(["Firstly, multiply by 19", "Simon says add 6", "Next, add 29", "Simon says add 50", "Joshua says multiply by 27"]), 56)
Test.assert_equals(simon_says(["Now add 44", "Now multiply by 27", "Simon says multiply by 30", "Now subtract 4", "Then multiply by 45"]), 0)
Test.assert_equals(simon_says(["Firstly, multiply by 26", "Simon says add 13", "Simeon says add 21"]), 13)
Test.assert_equals(simon_says(["Now subtract 41", "Now add 30", "Simon says multiply by 46", "Firstly, subtract 37", "Now multiply by 6", "Then multiply by 19", "Simon says add 23", "Simon says subtract 28"]), -5)
Test.assert_equals(simon_says(["Sarah says subtract 36", "Sieon says add 25", "Now add 24", "Sarah says add 27", "Then multiply by 8", "Firstly, add 35", "Sarah says add 21"]), 0)
Test.assert_equals(simon_says(["Simon says subtract 19", "Firstly, subtract 26", "Now add 48", "Then subtract 22", "Now subtract 15", "Then add 1", "Simon says add 28", "Firstly, multiply by 22", "Then add 44", "Simeon says multiply by 16", "Then multiply by 50", "Simon says subtract 26", "Firstly, add 49"]), -17)
Test.assert_equals(simon_says(["Now add 48", "Simeon says subtract 30", "Firstly, subtract 46", "Simon says subtract 21", "Sieon says add 32", "Joshua says add 45", "Now subtract 4", "Then multiply by 5", "Next, add 36"]), -21)
Test.assert_equals(simon_says(["Next, subtract 2", "Simon says add 32", "Then multiply by 46", "Sarah says multiply by 3", "Firstly, multiply by 3", "Simon says subtract 32"]), 0)
Test.assert_equals(simon_says(["Now multiply by 40"]), 0)
Test.assert_equals(simon_says(["Simeon says multiply by 45", "Sieon says subtract 19", "Simeon says subtract 32", "Firstly, multiply by 8", "Firstly, multiply by 39"]), 0)
Test.assert_equals(simon_says(["Next, subtract 19", "Simon says add 31", "Sarah says subtract 9"]), 31)
Test.assert_equals(simon_says(["Simon says add 35", "Firstly, add 12", "Now add 25", "Next, multiply by 43"]), 35)
Test.assert_equals(simon_says(["Then multiply by 9"]), 0)
Test.assert_equals(simon_says(["Simon says multiply by 23", "Simon says subtract 34", "Firstly, multiply by 45", "Now multiply by 24", "Now add 16", "Now add 37", "Sarah says multiply by 28", "Next, multiply by 8", "Simon says subtract 31", "Simon says add 12", "Simon says subtract 10", "Then add 34", "John says multiply by 50", "John says multiply by 27"]), -63)
Test.assert_equals(simon_says(["Now multiply by 46", "Simeon says subtract 15", "Then subtract 46", "Simon says subtract 18", "Next, multiply by 48", "Simeon says subtract 46", "Simeon says multiply by 24", "Next, add 38", "Now multiply by 14", "Simon says subtract 46", "Simon says multiply by 30"]), -1920)
Test.assert_equals(simon_says(["Then multiply by 42", "Firstly, add 16", "Joshua says multiply by 1", "Simon says multiply by 33", "Sarah says multiply by 26", "Firstly, subtract 48", "Simon says subtract 26", "Now add 1"]), -26)
Test.assert_equals(simon_says(["Firstly, add 10", "Now multiply by 2", "Simeon says add 40"]), 0)
Test.assert_equals(simon_says(["Then multiply by 22", "John says multiply by 18", "Simon says multiply by 14", "Firstly, add 41", "Simeon says add 45", "Sarah says subtract 9", "Sarah says subtract 24", "Now subtract 18"]), 0)
Test.assert_equals(simon_says(["Simeon says multiply by 40", "Next, multiply by 10", "Sarah says multiply by 7", "Then subtract 43", "Sieon says multiply by 10", "Firstly, add 21", "Next, subtract 22"]), 0)
Test.assert_equals(simon_says(["Sarah says subtract 13", "John says subtract 34", "Next, multiply by 22", "Sieon says add 9", "Joshua says subtract 43", "Now add 3", "Sieon says add 36", "Next, multiply by 46", "Next, subtract 50", "Sieon says subtract 50", "John says multiply by 8", "Next, multiply by 4", "Simon says subtract 22", "Simon says subtract 18", "Now subtract 23"]), -40)
Test.assert_equals(simon_says(["Then subtract 45", "Simon says multiply by 13", "Joshua says add 5", "Then subtract 1", "Simon says multiply by 2", "Firstly, add 25", "Now subtract 25", "Then multiply by 26", "Then multiply by 45", "Then multiply by 16", "Next, multiply by 8", "Firstly, add 15", "Joshua says multiply by 6", "Firstly, add 3", "Now multiply by 46"]), 0)
Test.assert_equals(simon_says(["Then multiply by 36", "Simon says multiply by 32", "Next, add 48", "Now subtract 28", "Firstly, subtract 38", "Sieon says subtract 7", "Sarah says multiply by 8", "Simon says multiply by 50", "Firstly, subtract 5", "Next, multiply by 50", "Joshua says add 13", "Now add 1", "Simon says subtract 49", "Firstly, multiply by 41", "Simon says add 17"]), -32)
Test.assert_equals(simon_says(["Next, add 7", "Firstly, multiply by 7", "Joshua says subtract 46", "Sarah says add 21", "Next, add 14", "Simeon says multiply by 32", "Simon says multiply by 11", "Then multiply by 19", "Sieon says multiply by 5", "Simon says multiply by 41"]), 0)
Test.assert_equals(simon_says(["Simon says subtract 40", "Sarah says add 7", "Then add 35", "Simon says multiply by 25", "Simon says add 7", "Next, multiply by 46", "Simon says add 3"]), -990)
Test.assert_equals(simon_says(["Next, subtract 21", "Now multiply by 13", "John says add 16", "Sarah says subtract 32", "Sarah says add 37", "Firstly, add 6", "Firstly, add 38", "Simon says subtract 21"]), -21)
Test.assert_equals(simon_says(["Then add 5", "Simon says multiply by 8", "Then multiply by 35", "Sieon says multiply by 23", "Simeon says multiply by 32", "Firstly, subtract 26", "John says add 3", "Simeon says add 9", "Simeon says add 20", "Simeon says subtract 26", "Next, multiply by 50", "Then subtract 32", "Now multiply by 13", "Simon says subtract 25", "Sarah says add 19", "Then subtract 45", "Now subtract 41", "Then subtract 12", "Now multiply by 31"]), -25)
Test.assert_equals(simon_says(["Simon says add 18", "Simon says multiply by 5"]), 90)
Test.assert_equals(simon_says(["Simon says multiply by 47", "Simeon says subtract 1", "Firstly, add 33", "Then multiply by 36", "Simon says multiply by 38", "Now subtract 15", "Simon says multiply by 41", "Sarah says subtract 14", "Then subtract 3", "Now multiply by 44", "Firstly, add 49", "Firstly, subtract 16"]), 0)
Test.assert_equals(simon_says(["Sieon says add 49", "Next, subtract 30", "Simon says add 36", "Firstly, add 9", "Simon says subtract 11", "Next, add 42", "Simon says multiply by 24", "Now subtract 4", "Now multiply by 40", "Simeon says add 44", "Simon says multiply by 32", "Simeon says multiply by 22"]), 19200)
Test.assert_equals(simon_says(["Now add 7", "John says multiply by 11", "Simon says multiply by 25", "Sieon says subtract 5", "Simon says multiply by 29", "Firstly, multiply by 43", "Now add 1", "Now subtract 15", "Simon says add 45", "Then subtract 22", "Simon says subtract 4", "Next, subtract 11", "Simon says subtract 17", "Firstly, add 32", "Firstly, subtract 34", "Then subtract 36", "Now subtract 40", "Sarah says add 48"]), 24)
Test.assert_equals(simon_says(["Simeon says add 24", "Firstly, multiply by 23", "Simon says add 50", "Simon says add 45", "Then subtract 11", "Firstly, subtract 20", "Simeon says subtract 40", "Simon says add 35", "Simeon says multiply by 3", "Now multiply by 27", "Now add 46", "Simon says multiply by 42", "Simon says subtract 12", "Simeon says multiply by 19", "Then add 44", "Next, add 38", "John says multiply by 39", "Firstly, add 25", "Then subtract 44"]), 5448)
Test.assert_equals(simon_says(["Simon says add 45", "Simon says add 33", "Sieon says subtract 23", "Simon says multiply by 46", "Then subtract 4", "Next, subtract 46"]), 3588)
Test.assert_equals(simon_says(["Firstly, subtract 36", "Now add 19", "Firstly, add 50", "Sarah says multiply by 32", "Next, multiply by 45", "Firstly, add 4", "Now multiply by 23", "Next, multiply by 46", "Next, multiply by 15", "Simon says multiply by 49", "Then add 6", "Simon says multiply by 39", "Firstly, add 2", "Next, subtract 7", "Simon says subtract 20", "Simon says multiply by 13", "Simeon says subtract 32", "Simon says add 15"]), -245)
Test.assert_equals(simon_says(["Firstly, subtract 37", "Sieon says multiply by 5", "Firstly, multiply by 10"]), 0)
Test.assert_equals(simon_says(["Simon says multiply by 33", "Sarah says subtract 19", "Now subtract 32", "Next, add 41", "Simeon says subtract 27", "Firstly, multiply by 48", "Then multiply by 46", "Simon says subtract 41", "Now multiply by 50", "Simon says subtract 6", "Simon says add 20", "Sieon says add 47", "Sarah says subtract 13", "Next, add 49", "Simon says multiply by 2", "Simon says subtract 50", "Simon says subtract 47", "Now subtract 7", "Joshua says subtract 21", "Simon says multiply by 3"]), -453)
Test.assert_equals(simon_says(["Simon says add 14", "Simon says add 24"]), 38)
Test.assert_equals(simon_says(["Simon says subtract 34", "John says add 1", "Simon says subtract 40", "Next, multiply by 7", "Firstly, subtract 10", "Next, add 13", "Simon says multiply by 36", "Now multiply by 7", "Now multiply by 6", "Next, multiply by 19", "Simon says multiply by 47", "Simeon says multiply by 40", "Simon says subtract 13", "Joshua says multiply by 45", "Simeon says multiply by 1", "Simon says add 32", "Next, multiply by 28"]), -125189)
Test.assert_equals(simon_says(["Then multiply by 6"]), 0)
Test.assert_equals(simon_says(["Simon says multiply by 48", "Firstly, subtract 14", "Next, add 46", "John says add 44", "Simon says multiply by 16", "Firstly, subtract 42", "Firstly, add 34", "Then multiply by 26", "Then multiply by 32", "Simeon says add 40", "Simon says multiply by 48", "Sarah says multiply by 46"]), 0)
Test.assert_equals(simon_says(["Next, multiply by 20", "Sarah says subtract 18", "Now add 47", "Sarah says multiply by 4", "Simon says subtract 47", "Simon says multiply by 31", "Firstly, multiply by 39"]), -1457)
Test.assert_equals(simon_says(["Firstly, multiply by 13"]), 0)
Test.assert_equals(simon_says(["Sieon says add 29", "Next, multiply by 14", "Sieon says multiply by 25", "Simon says subtract 10", "Simeon says add 39", "Simeon says multiply by 13", "Simon says multiply by 8", "Next, subtract 18", "Next, add 28", "Simon says add 11", "Next, add 5", "John says add 21"]), -69)
Test.assert_equals(simon_says(["Sarah says multiply by 35", "Then multiply by 11", "Simeon says subtract 5"]), 0)
Test.assert_equals(simon_says(["Firstly, multiply by 4", "Now multiply by 4", "Firstly, add 40", "John says add 30", "Simon says multiply by 35"]), 0)
Test.assert_equals(simon_says(["Next, subtract 27", "Next, subtract 33", "Then multiply by 3", "Simon says add 46", "Next, subtract 48", "Now add 37", "Simon says subtract 29", "Next, add 14"]), 17)
Test.assert_equals(simon_says(["Simon says subtract 48", "John says subtract 50", "Sieon says subtract 6", "Simon says subtract 4", "Sieon says subtract 10", "Now multiply by 46", "Now multiply by 44", "Simeon says multiply by 23", "Simon says multiply by 18", "Now subtract 8", "Then subtract 49", "Simon says subtract 48", "Simon says add 43"]), -941)
Test.assert_equals(simon_says(["Then multiply by 26", "Simon says add 37", "Now subtract 28", "Now add 3", "Next, add 5", "Simeon says multiply by 42", "Simon says subtract 45", "Firstly, multiply by 30", "Now add 11"]), -8)
Test.assert_equals(simon_says(["Simon says add 6", "Sieon says multiply by 3", "Then add 48", "Next, subtract 48", "Simon says multiply by 9", "Simon says add 10", "Simeon says multiply by 41", "Simon says subtract 8", "Next, add 1", "Then add 31", "Simon says subtract 37", "Simon says multiply by 3", "Now multiply by 10", "Then add 33", "Firstly, multiply by 17", "Next, multiply by 20", "Simeon says multiply by 28", "Sieon says multiply by 28", "Then add 32", "Then add 34"]), 57)
Test.assert_equals(simon_says(["Simeon says subtract 27", "Next, add 31", "Firstly, subtract 16", "Sieon says add 5", "Firstly, multiply by 49", "Firstly, add 20", "Now multiply by 11", "Simon says add 43", "Simon says add 48", "Simeon says multiply by 9", "Sieon says subtract 50", "Now multiply by 14", "Firstly, subtract 14", "Then multiply by 27", "Sieon says multiply by 23", "Simon says subtract 33", "Simon says multiply by 45", "Firstly, subtract 25"]), 2610)

Test.assert_equals(lychrel(33), 0)
Test.assert_equals(lychrel(65), 1)
Test.assert_equals(lychrel(348), 3)
Test.assert_equals(lychrel(196), 1)
Test.assert_equals(lychrel(89), 24)
Test.assert_equals(lychrel(7582), 4)
Test.assert_equals(lychrel(1945), 1)
Test.assert_equals(lychrel(3673), 1)
Test.assert_equals(lychrel(9485367), 2)
Test.assert_equals(lychrel(695), 3)
Test.assert_equals(lychrel(10911), 55)

Test.assert_equals(find_missing([[1], [1, 2], [4, 5, 1, 1], [5, 6, 7, 8, 9]]), 3)
Test.assert_equals(find_missing([[5, 6, 7, 8, 9], [1, 2], [4, 5, 1, 1], [1] ]), 3)
Test.assert_equals(find_missing([[4, 4, 4, 4], [1], [3, 3, 3]]), 2)
Test.assert_equals(find_missing([[0], [0, 0, 0]]), 2)
Test.assert_equals(find_missing([["f", "r", "s"], ["d", "e"], ["a", "f", "b", "n"], ["z"], ["fg", "gty", "d", "dfr", "dr", "q"]]), 5)
Test.assert_equals(find_missing([[5, 2, 9], [4, 5, 1, 1, 5, 6], [1, 1], [5, 6, 7, 8, 9]]), 4)
Test.assert_equals(find_missing([]), 0, "When the main list is empty, return 0.")
Test.assert_equals(find_missing(None), 0, "Return 0 if you are given None as an argument.")
Test.assert_equals(find_missing([[], [1, 2, 2]]), 0, "If a list within the parent list is empty, return 0.")

Test.assert_equals(factorial(7), 5040)
Test.assert_equals(factorial(1), 1)
Test.assert_equals(factorial(9), 362880)
Test.assert_equals(factorial(2), 2)

Test.assert_equals(lemonade([5, 5, 5, 10, 20]), 1)
Test.assert_equals(lemonade([5, 5, 10]), 1)
Test.assert_equals(lemonade([10, 10]), 0)
Test.assert_equals(lemonade([5, 5, 10, 10, 20]), 0)
Test.assert_equals(lemonade([5, 5, 5, 5, 10, 5, 10, 10, 10, 20]), 1)
Test.assert_equals(lemonade([5, 10, 5, 5, 5, 20, 5, 10, 5, 5, 10, 20]), 1)
Test.assert_equals(lemonade([5, 10, 5, 5, 5, 20, 5, 10, 20, 5, 10, 20, 10]), 0)

Test.assert_equals(transpose_matrix([[1,1,1],[2,2,2],[3,3,3]]),[[1,2,3],[1,2,3],[1,2,3]])
Test.assert_equals(transpose_matrix([[1,1,1],[2,2,2]]),[[1,2],[1,2],[1,2]])
Test.assert_equals(transpose_matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12]]),[[1,5,9],[2,6,10],[3,7,11],[4,8,12]])

Test.assert_equals(
  leaderboards([
    { 'name': 'a', 'score': 100, 'reputation': 20 },
    { 'name': 'b', 'score': 90, 'reputation': 40 },
    { 'name': 'c', 'score': 115, 'reputation': 30 },
  ]),
  [
    { 'name': 'c', 'score': 115, 'reputation': 30 },
    { 'name': 'b', 'score': 90, 'reputation': 40 },
    { 'name': 'a', 'score': 100, 'reputation': 20 },
  ]
)

Test.assert_equals(
  leaderboards([
    { 'name': 'tkincaid0', 'score': 4128, 'reputation': 3002 },
    { 'name': 'sblackater1', 'score': 6208, 'reputation': 3050 },
    { 'name': 'ocallis2', 'score': 6883, 'reputation': 3812 },
    { 'name': 'shoofe3', 'score': 4900, 'reputation': 174 },
    { 'name': 'cbrazear4', 'score': 7862, 'reputation': 2940 },
    { 'name': 'oszachnie5', 'score': 6217, 'reputation': 1772 },
    { 'name': 'lingcourt6', 'score': 5746, 'reputation': 1263 },
    { 'name': 'tquincey7', 'score': 4209, 'reputation': 1419 },
    { 'name': 'mcapsey8', 'score': 6961, 'reputation': 2699 },
    { 'name': 'cbester9', 'score': 4090, 'reputation': 3934 },
  ]),
  [
    { 'name': 'ocallis2', 'score': 6883, 'reputation': 3812 },
    { 'name': 'cbrazear4', 'score': 7862, 'reputation': 2940 },
    { 'name': 'mcapsey8', 'score': 6961, 'reputation': 2699 },
    { 'name': 'sblackater1', 'score': 6208, 'reputation': 3050 },
    { 'name': 'cbester9', 'score': 4090, 'reputation': 3934 },
    { 'name': 'tkincaid0', 'score': 4128, 'reputation': 3002 },
    { 'name': 'oszachnie5', 'score': 6217, 'reputation': 1772 },
    { 'name': 'lingcourt6', 'score': 5746, 'reputation': 1263 },
    { 'name': 'tquincey7', 'score': 4209, 'reputation': 1419 },
    { 'name': 'shoofe3', 'score': 4900, 'reputation': 174 },
  ]
)

Test.assert_equals(
  leaderboards([
    { 'name': 'kdavet0', 'score': 8680, 'reputation': 3149 },
    { 'name': 'rollerearn1', 'score': 7127, 'reputation': 968 },
    { 'name': 'hcastel2', 'score': 8375, 'reputation': 1650 },
    { 'name': 'mslorach3', 'score': 8097, 'reputation': 1925 },
    { 'name': 'hseefus4', 'score': 5526, 'reputation': 1747 },
    { 'name': 'ddiggles5', 'score': 7519, 'reputation': 3433 },
    { 'name': 'estalman6', 'score': 8516, 'reputation': 755 },
    { 'name': 'eklemt7', 'score': 8487, 'reputation': 3289 },
    { 'name': 'eskitch8', 'score': 7762, 'reputation': 329 },
    { 'name': 'jroos9', 'score': 6288, 'reputation': 3043 },
  ]),
  [
    { 'name': 'eklemt7', 'score': 8487, 'reputation': 3289 },
    { 'name': 'kdavet0', 'score': 8680, 'reputation': 3149 },
    { 'name': 'ddiggles5', 'score': 7519, 'reputation': 3433 },
    { 'name': 'jroos9', 'score': 6288, 'reputation': 3043 },
    { 'name': 'mslorach3', 'score': 8097, 'reputation': 1925 },
    { 'name': 'hcastel2', 'score': 8375, 'reputation': 1650 },
    { 'name': 'estalman6', 'score': 8516, 'reputation': 755 },
    { 'name': 'rollerearn1', 'score': 7127, 'reputation': 968 },
    { 'name': 'hseefus4', 'score': 5526, 'reputation': 1747 },
    { 'name': 'eskitch8', 'score': 7762, 'reputation': 329 },
  ]
)

Test.assert_equals(
  leaderboards([
    { 'name': 'jlordon0', 'score': 7775, 'reputation': 789 },
    { 'name': 'srosenshine1', 'score': 5055, 'reputation': 3928 },
    { 'name': 'wendrighi2', 'score': 8039, 'reputation': 3519 },
    { 'name': 'rburt3', 'score': 5944, 'reputation': 3451 },
    { 'name': 'mgreest4', 'score': 7333, 'reputation': 2452 },
    { 'name': 'khugues5', 'score': 5304, 'reputation': 2465 },
    { 'name': 'bhazeman6', 'score': 4164, 'reputation': 3203 },
    { 'name': 'vcauson7', 'score': 4918, 'reputation': 3781 },
    { 'name': 'ffarrears8', 'score': 6438, 'reputation': 1929 },
    { 'name': 'jtwamley9', 'score': 4690, 'reputation': 3731 },
  ]),
  [
    { 'name': 'wendrighi2', 'score': 8039, 'reputation': 3519 },
    { 'name': 'srosenshine1', 'score': 5055, 'reputation': 3928 },
    { 'name': 'rburt3', 'score': 5944, 'reputation': 3451 },
    { 'name': 'vcauson7', 'score': 4918, 'reputation': 3781 },
    { 'name': 'mgreest4', 'score': 7333, 'reputation': 2452 },
    { 'name': 'jtwamley9', 'score': 4690, 'reputation': 3731 },
    { 'name': 'bhazeman6', 'score': 4164, 'reputation': 3203 },
    { 'name': 'ffarrears8', 'score': 6438, 'reputation': 1929 },
    { 'name': 'khugues5', 'score': 5304, 'reputation': 2465 },
    { 'name': 'jlordon0', 'score': 7775, 'reputation': 789 },
  ]
)

Test.assert_equals(get_length([1, [2,3]]), 3)
Test.assert_equals(get_length([1, [2, [3, 4]]]), 4)
Test.assert_equals(get_length([1, [2, [3, [4, [5, 6]]]]]), 6)
Test.assert_equals(get_length([1, 7, 8]), 3)
Test.assert_equals(get_length([2]), 1)
Test.assert_equals(get_length([2, [3], 4, [7]]), 4)
Test.assert_equals(get_length([2, [3, [5, 7]], 4, [7]]), 6)
Test.assert_equals(get_length([2, [3, [4, [5]]], [9]]), 5)
Test.assert_equals(get_length([]), 0)

recede = ["cee","dee","eer","erd","ere","red","ree","cede","cere","cree","deer","dere","dree","rede","reed","ceder","cedre","cered","creed","decree","recede"]

Test.assert_equals(scrambled(recede, "*re**"), ["creed"])
Test.assert_equals(scrambled(recede, "***"), ["cee","dee","eer","erd","ere","red","ree"])
Test.assert_equals(scrambled(recede, "******"), ["decree","recede"])
Test.assert_equals(scrambled(recede, "c*d**"), ["ceder","cedre"])
Test.assert_equals(scrambled(recede, "d***"), ["deer","dere","dree"])
Test.assert_equals(scrambled(recede, "r***"), ["rede","reed"])

Test.assert_equals(pricey_prod({'Computer' : 600, 'TV' : 800, 'Radio' : 100}), ['TV','Computer'])
Test.assert_equals(pricey_prod({'Bike1' : 510, 'Bike2' : 401, 'Bike3' : 501}), ['Bike1', 'Bike3'])
Test.assert_equals(pricey_prod({'Calvin Klein' : 500, 'Armani' : 5000, 'Dolce & Gabbana' : 2000}), ['Armani', 'Dolce & Gabbana', 'Calvin Klein'])
Test.assert_equals(pricey_prod({'Loafers' : 50, 'Vans' : 10, 'Crocs' : 20}), [])
Test.assert_equals(pricey_prod({'Dell' : 400, 'HP' : 300, 'Apple' : 1200}), ['Apple'])

Test.assert_equals(boxes([7, 1, 2, 6, 1, 2, 3, 5, 9, 2, 1, 2, 5]), 5)
Test.assert_equals(boxes([2, 7, 1, 3, 3, 4, 7, 4, 1, 8, 2]), 5)
Test.assert_equals(boxes([1, 3, 3, 3, 2, 1, 1, 9, 7, 10, 8, 6, 1, 2, 9]), 8)
Test.assert_equals(boxes([4, 1, 2, 3, 5, 5, 1, 9]), 3)
Test.assert_equals(boxes([3, 1, 2, 7, 2, 6, 1]), 3)
Test.assert_equals(boxes([4, 6, 1, 9, 6, 1, 1, 9, 2, 9]), 6)
Test.assert_equals(boxes([2, 2, 10, 10, 1, 5, 2]), 4)
Test.assert_equals(boxes([9, 6, 2, 3, 1, 2, 4, 8, 3, 1, 3]), 5)
Test.assert_equals(boxes([2, 5, 1, 6, 2, 9, 5, 2, 1, 6, 1, 6, 6, 1]), 7)
Test.assert_equals(boxes([1, 2, 3, 2, 6, 4, 1]), 3)
Test.assert_equals(boxes([1, 1, 2, 1, 2, 10, 2, 2, 5, 1, 5]), 4)
Test.assert_equals(boxes([8, 3, 2, 1, 1, 2, 1, 3, 2, 1]), 3)
Test.assert_equals(boxes([1, 5, 3, 1, 2, 3, 2, 6, 3, 1, 3, 8, 1]), 5)
Test.assert_equals(boxes([8, 1, 1, 4, 7, 2, 1, 3, 1, 9, 7, 1, 5, 1, 1]), 7)
Test.assert_equals(boxes([2, 3, 4, 10, 1, 2, 5, 1, 1, 1, 1, 8, 2, 1]), 5)
Test.assert_equals(boxes([4, 6, 7, 3, 2, 2, 3, 1, 2, 2, 10, 3, 2]), 6)
Test.assert_equals(boxes([9, 2, 3, 4, 1, 3, 1, 1, 3]), 3)
Test.assert_equals(boxes([6, 2, 1, 9, 1, 8, 2, 8, 6, 6]), 6)
Test.assert_equals(boxes([6, 9, 3, 8, 10, 4, 7]), 7)
Test.assert_equals(boxes([4, 3, 1, 1, 1, 4, 7, 2, 1, 10, 1, 3, 8]), 6)
Test.assert_equals(boxes([10]), 1)

Test.assert_equals(count_ones([1, 1, 1, 1, 1]), 1)
Test.assert_equals(count_ones([1, 1, 1, 1, 0]), 1)
Test.assert_equals(count_ones([0, 0, 0, 0, 0]), 0)
Test.assert_equals(count_ones([1, 0, 0, 0, 0]), 0)
Test.assert_equals(count_ones([1, 0, 1, 0, 1]), 0)
Test.assert_equals(count_ones([1, 0, 0, 0, 1, 0, 0, 1, 1]), 1)
Test.assert_equals(count_ones([1, 1, 0, 1, 1, 0, 0, 1, 1]), 3)
Test.assert_equals(count_ones([1, 0, 0, 1, 1, 0, 0, 1, 1]), 2)
Test.assert_equals(count_ones([1, 0, 0, 1, 1, 0, 1, 1, 1]), 2)
Test.assert_equals(count_ones([1, 0, 1, 0, 1, 0, 1, 0]), 0)
Test.assert_equals(count_ones([1, 1, 1, 1, 0, 0, 0, 0]), 1)
Test.assert_equals(count_ones([1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1]), 3)

Test.assert_equals(bridge_shuffle(['A', 'A', 'A'], ['B', 'B', 'B']), ['A', 'B', 'A', 'B', 'A', 'B'])
Test.assert_equals(bridge_shuffle(['C', 'C', 'C', 'C'], ['D']), ['C', 'D', 'C', 'C', 'C'])
Test.assert_equals(bridge_shuffle([1, 3, 5, 7], [2, 4, 6]), [1, 2, 3, 4, 5, 6, 7])
Test.assert_equals(bridge_shuffle([10, 9, 8], [1, 2, 3, 4]), [10, 1, 9, 2, 8, 3, 4])
Test.assert_equals(bridge_shuffle(['h', 'h', 'h'], ['a', 'a', 'a']), ['h', 'a', 'h', 'a', 'h', 'a'])

Test.assert_equals(num_then_char([
    [1, 2, 4, 3, "a", "b"],
    [6, "c", 5],
    [7, "d"],
    ["f", "e", 8]
]), [[1, 2, 3, 4, 5, 6], [7, 8, 'a'], ['b', 'c'], ['d', 'e', 'f']])

Test.assert_equals(num_then_char([
    [1, 2, 4.4, "f", "a", "b"],
    [0],
    [0.5, "d","X",3,"s"],
    ["f", "e", 8],
    ["p","Y","Z"],
    [12,18]
]), [[0, 0.5, 1, 2, 3, 4.4], [8], [12, 18, 'X', 'Y', 'Z'], ['a', 'b', 'd'], ['e', 'f', 'f'], ['p', 's']])

Test.assert_equals(num_then_char([
    [10, 2],
    ["a",3],
    [2.2, "d","h",6,"s",14,1],
    ["f", "e"],
    ["p","y","z","V"],
    [5]
]), [[1, 2], [2.2, 3], [5, 6, 10, 14, 'V', 'a', 'd'], ['e', 'f'], ['h', 'p', 's', 'y'], ['z']])

Test.assert_equals(num_then_char([
    [10, 2,6,6.5,8.1,"q","w","a","s"],
    ["f",4],
    [2, 3,"h",6,"x",1,0],
    ["g"],
    ["p",7,"j","k","l"],
    [5,"C","A","B"],
    ["L",9]
]), [[0, 1, 2, 2, 3, 4, 5, 6, 6], [6.5, 7], [8.1, 9, 10, 'A', 'B', 'C', 'L'], ['a'], ['f', 'g', 'h', 'j', 'k'], ['l', 'p', 'q', 's'], ['w', 'x']])

Test.assert_equals(vertical_txt("Holy bananas"), [['H', 'b'], ['o', 'a'], ['l', 'n'], ['y', 'a'], [' ', 'n'], [' ', 'a'], [' ', 's']])
Test.assert_equals(vertical_txt("Hello fellas"), [['H', 'f'], ['e', 'e'], ['l', 'l'], ['l', 'l'], ['o', 'a'], [' ', 's']])
Test.assert_equals(vertical_txt("I hope you have a great day"), [['I', 'h', 'y', 'h', 'a', 'g', 'd'], [' ', 'o', 'o', 'a', ' ', 'r', 'a'], [' ', 'p', 'u', 'v', ' ', 'e', 'y'], [' ', 'e', ' ', 'e', ' ', 'a', ' '], [' ', ' ', ' ', ' ', ' ', 't', ' ']])
Test.assert_equals(vertical_txt("Piri piri over there"), [['P', 'p', 'o', 't'], ['i', 'i', 'v', 'h'], ['r', 'r', 'e', 'e'], ['i', 'i', 'r', 'r'], [' ', ' ', ' ', 'e']])
Test.assert_equals(vertical_txt("Skill the baboon king"), [['S', 't', 'b', 'k'], ['k', 'h', 'a', 'i'], ['i', 'e', 'b', 'n'], ['l', ' ', 'o', 'g'], ['l', ' ', 'o', ' '], [' ', ' ', 'n', ' ']])
Test.assert_equals(vertical_txt("He took one for the team"), [['H', 't', 'o', 'f', 't', 't'], ['e', 'o', 'n', 'o', 'h', 'e'], [' ', 'o', 'e', 'r', 'e', 'a'], [' ', 'k', ' ', ' ', ' ', 'm']])
Test.assert_equals(vertical_txt("Schneid! 700 in to the face!"), [['S', '7', 'i', 't', 't', 'f'], ['c', '0', 'n', 'o', 'h', 'a'], ['h', '0', ' ', ' ', 'e', 'c'], ['n', ' ', ' ', ' ', ' ', 'e'], ['e', ' ', ' ', ' ', ' ', '!'], ['i', ' ', ' ', ' ', ' ', ' '], ['d', ' ', ' ', ' ', ' ', ' '], ['!', ' ', ' ', ' ', ' ', ' ']])
Test.assert_equals(vertical_txt("I hope you are ready for your daily dose of skill"), [['I', 'h', 'y', 'a', 'r', 'f', 'y', 'd', 'd', 'o', 's'], [' ', 'o', 'o', 'r', 'e', 'o', 'o', 'a', 'o', 'f', 'k'], [' ', 'p', 'u', 'e', 'a', 'r', 'u', 'i', 's', ' ', 'i'], [' ', 'e', ' ', ' ', 'd', ' ', 'r', 'l', 'e', ' ', 'l'], [' ', ' ', ' ', ' ', 'y', ' ', ' ', 'y', ' ', ' ', 'l']])
Test.assert_equals(vertical_txt("0 11 222 3333 44444 6666666 77777777 888888888 9999999999"), [['0', '1', '2', '3', '4', '6', '7', '8', '9'], [' ', '1', '2', '3', '4', '6', '7', '8', '9'], [' ', ' ', '2', '3', '4', '6', '7', '8', '9'], [' ', ' ', ' ', '3', '4', '6', '7', '8', '9'], [' ', ' ', ' ', ' ', '4', '6', '7', '8', '9'], [' ', ' ', ' ', ' ', ' ', '6', '7', '8', '9'], [' ', ' ', ' ', ' ', ' ', '6', '7', '8', '9'], [' ', ' ', ' ', ' ', ' ', ' ', '7', '8', '9'], [' ', ' ', ' ', ' ', ' ', ' ', ' ', '8', '9'], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '9']])

Test.assert_equals(can_traverse([
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 1, 0, 0, 0, 0, 0], 
	[0, 0, 1, 1, 0, 0, 1, 0, 0], 
	[0, 1, 1, 1, 1, 1, 1, 1, 0]
]), False)

Test.assert_equals(can_traverse([
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 1, 0, 0, 0, 0, 0], 
	[0, 0, 1, 1, 1, 0, 1, 0, 0], 
	[0, 1, 1, 1, 1, 1, 1, 1, 0]
]), True)

Test.assert_equals(can_traverse([
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 1, 0, 0, 0, 0, 0], 
	[0, 0, 1, 1, 1, 1, 1, 0, 0], 
	[0, 1, 1, 1, 1, 1, 1, 1, 0]
]), True)

Test.assert_equals(can_traverse([
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 1, 0, 0, 0, 0, 0], 
	[0, 1, 1, 1, 1, 1, 1, 0, 0], 
	[0, 1, 1, 1, 1, 1, 1, 1, 0]
]), False)

Test.assert_equals(can_traverse([
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 1, 1, 0, 0, 0, 0], 
	[0, 0, 1, 1, 1, 1, 1, 0, 0], 
	[0, 1, 1, 1, 1, 1, 1, 1, 0]
]), True)

Test.assert_equals(can_traverse([
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 1, 0, 0, 0, 1, 0, 0], 
	[0, 1, 1, 1, 0, 1, 1, 1, 0]
]), True)

Test.assert_equals(can_traverse([
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 1, 1, 1, 0, 1, 1, 1, 0]
]), True)

Test.assert_equals(can_traverse([
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 1, 0, 0, 0, 0, 1], 
	[0, 0, 1, 1, 1, 0, 1, 1, 1], 
	[0, 1, 1, 1, 1, 1, 1, 1, 1]
]), True)

Test.assert_equals(can_traverse([
	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 1, 0, 0, 0, 0, 1], 
	[0, 0, 1, 1, 1, 0, 1, 0, 1], 
	[0, 1, 1, 1, 1, 1, 1, 1, 1]
]), False)

Test.assert_equals(cutting_grass([4, 4, 4, 4], 1, 1, 1, 1), 
	[[3, 3, 3, 3], [2, 2, 2, 2], [1, 1, 1, 1], "Done"])

Test.assert_equals(cutting_grass([5, 6, 7, 5], 1, 2, 1), 
	[[4, 5, 6, 4], [2, 3, 4, 2], [1, 2, 3, 1]])

Test.assert_equals(cutting_grass([8, 9, 9, 8, 8], 2, 3, 2, 1), 
	[[6, 7, 7, 6, 6], [3, 4, 4, 3, 3], [1, 2, 2, 1, 1], "Done"])

Test.assert_equals(cutting_grass([1, 0, 1, 1], 1, 1, 1), 
	["Done", "Done", "Done"])

Test.assert_equals(cutting_grass([4, 5, 4, 5], 2, 1, 1), 
	[[2, 3, 2, 3], [1, 2, 1, 2], "Done"])

Test.assert_equals(cutting_grass([4, 2, 2], 2, 1, 1), 
	["Done", "Done", "Done"])


Test.assert_equals(check_pattern([[1, 1], [2, 2], [1, 1], [2, 2]], "ABAB"), True)
Test.assert_equals(check_pattern([[1, 2], [1, 2], [1, 2], [1, 2]], "AAAA"), True)
Test.assert_equals(check_pattern([[1, 2], [1, 3], [1, 4], [1, 2]], "ABCA"), True)
Test.assert_equals(check_pattern([[1, 2, 3], [1, 2, 3], [3, 2, 1], [3, 2, 1]], "AABB"), True)
Test.assert_equals(check_pattern([[8, 8, 8, 8], [7, 7, 7, 7], [6, 6, 6, 6], [5, 5, 5, 5]], "ABCD"), True)
Test.assert_equals(check_pattern([[8, 8, 8, 8], [7, 7, 7, 7], [6, 6, 6, 6], [5, 5, 5, 5]], "DCBA"), True)
Test.assert_equals(check_pattern([[8], [7], [6], [6]], "ABCC"), True)
Test.assert_equals(check_pattern([[8], [9], [9], [9]], "ABBB"), True)
Test.assert_equals(check_pattern([[1, 1], [2, 2], [1, 1], [1, 2]], "ABAB"), False)
Test.assert_equals(check_pattern([[1, 2], [1, 2], [2, 2], [3, 2]], "AAAA"), False)
Test.assert_equals(check_pattern([[8], [9], [9], [8]], "ABBB"), False)
Test.assert_equals(check_pattern([[8], [7], [6], [5]], "ABCC"), False)
Test.assert_equals(check_pattern([[8, 8, 8, 8], [7, 7, 7, 7], [6, 6, 6, 6], [5, 5, 5, 5]], "DDBA"), False)
Test.assert_equals(check_pattern([[1, 2], [1, 2], [1, 2], [1, 2]], "AABA"), False)

Test.assert_equals(transform_matrix(
[
[1, 0, 0, 0, 1], 
[0, 1, 0, 0, 0], 
[0, 0, 0, 1, 0], 
[0, 1, 0, 1, 0], 
[0, 1, 0, 0, 0]])
,[
[1, 5, 2, 4, 1], 
[2, 2, 1, 3, 2], 
[2, 4, 1, 1, 2], 
[3, 3, 2, 2, 3], 
[2, 2, 1, 3, 2]
])

Test.assert_equals(transform_matrix([
[1, 0, 0, 0], 
[0, 1, 0, 0], 
[0, 0, 1, 0]
]), [
[0, 2, 2, 1], 
[2, 0, 2, 1], 
[2, 2, 0, 1]
])

Test.assert_equals(transform_matrix([
[1, 1], 
[1, 1], 
[1, 1]
]), [
[3, 3], 
[3, 3], 
[3, 3]
])

Test.assert_equals(transform_matrix([
[1, 0, 0], 
[0, 1, 0], 
[0, 0, 1]
]), [
[0, 2, 2], 
[2, 0, 2], 
[2, 2, 0]
])

Test.assert_equals(transform_matrix([
[1, 1, 1], 
[0, 0, 1], 
[0, 0, 1]
]), [
[2, 2, 4], 
[2, 2, 2], 
[2, 2, 2]
])

Test.assert_equals(transform_matrix([
[1, 1, 1], 
[0, 1, 1], 
[0, 0, 1]
]), [
[2, 3, 4], 
[3, 2, 3], 
[2, 3, 2]
])

Test.assert_equals(three_sum([0, 1, -1, -1, 2]), [[0, 1, -1], [-1, -1, 2]])
Test.assert_equals(three_sum([0, 0, 0, 5, -5]), [[0, 0, 0], [0, 5, -5]])
Test.assert_equals(three_sum([0, -1, 1, 0, -1, 1]), [[0, -1, 1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [1, 0, -1]])
Test.assert_equals(three_sum([0, 5, 5, 0, 0]), [[0, 0, 0]])
Test.assert_equals(three_sum([0, 5, -5, 0, 0]), [[0, 5, -5], [0, 0, 0], [5, -5, 0]])
Test.assert_equals(three_sum([1, 2, 3, -5, 8, 9, -9, 0]), [[1, 8, -9], [2, 3, -5], [9, -9, 0]])
Test.assert_equals(three_sum([0, 0, 0]), [[0, 0, 0]])
Test.assert_equals(three_sum([1, 5, 5, 2]), [])
Test.assert_equals(three_sum([1, 1]), [])
Test.assert_equals(three_sum([]), [])


Test.assert_equals(order_people([5, 3], 15), [
	[1, 2, 3],
	[6, 5, 4],
	[7, 8, 9],
	[12, 11, 10],
	[13, 14, 15]
])

Test.assert_equals(order_people([4, 3], 5), [
	[1, 2, 3],
	[0, 5, 4],
	[0, 0, 0],
	[0, 0, 0]
])

Test.assert_equals(order_people([3, 3], 8), [
	[1, 2, 3],
	[6, 5, 4],
	[7, 8, 0]
])

Test.assert_equals(order_people([2, 4], 5), [
	[1, 2, 3, 4],
	[0, 0, 0, 5]
])

Test.assert_equals(order_people([4, 4], 15), [
	[1, 2, 3, 4],
	[8, 7, 6, 5],
	[9, 10, 11, 12],
	[0, 15, 14, 13]
])

Test.assert_equals(order_people([4, 4], 12), [
	[1, 2, 3, 4],
	[8, 7, 6, 5],
	[9, 10, 11, 12],
	[0, 0, 0, 0]
])

Test.assert_equals(order_people([2, 2], 4), [
	[1, 2],
	[4, 3]
])

Test.assert_equals(order_people([2, 2], 5),"overcrowded")

Test.assert_equals(order_people([2, 2], 3), [
	[1, 2],
	[0, 3]
])

Test.assert_equals(order_people([3, 4], 1), [
	[1, 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0]
])

Test.assert_equals(order_people([2, 4], 10), "overcrowded")

Test.assert_equals(has_consecutive_series([1, 2, 3], [1, 1, 1]), True)
Test.assert_equals(has_consecutive_series([1, 2, 3], [1, 2, 1]), False)
Test.assert_equals(has_consecutive_series([4, 6, -5, 8, 4], [-2, -3, 9, -3, 2]), True)
Test.assert_equals(has_consecutive_series([12, 3], [0, 10, 14, 15, 16]), True)
Test.assert_equals(has_consecutive_series([8, 6, 10], [25, 28, 25, 26, 28, 29]), False)
Test.assert_equals(has_consecutive_series([11, 5, 3], [-2, 5, 8, 12]), True)
Test.assert_equals(has_consecutive_series([11, 5, 3], [-2, 5, 8, 11]), False)

from time import perf_counter
tic = perf_counter()

Test.assert_equals(num_regions([
    [1, 1, 1, 1, 0],
    [1, 1, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]]), 1)
Test.assert_equals(num_regions([
    [1, 1, 1, 1, 0],
    [1, 0, 0, 1, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1]]), 2)
Test.assert_equals(num_regions([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1]]), 3)
Test.assert_equals(num_regions([[1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 0, 1], [1, 0, 0, 0, 1, 0]]), 3)
Test.assert_equals(num_regions([[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1]]), 5)
Test.assert_equals(num_regions([[1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1], [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]]), 8)
Test.assert_equals(num_regions([[1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0]]), 11)
Test.assert_equals(num_regions([[1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1], [1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]]), 30)
Test.assert_equals(num_regions([[1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1], [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0], [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1], [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0]]), 130)
print('t_sec = {:.6f}'.format(perf_counter() - tic))

# p available
p = [
  { 'number': 1, 'price': 100, 'name': 'Orange juice' },
  { 'number': 2, 'price': 200, 'name': 'Soda' },
  { 'number': 3, 'price': 150, 'name': 'Chocolate snack' },
  { 'number': 4, 'price': 250, 'name': 'Cookies' },
  { 'number': 5, 'price': 180, 'name': 'Gummy bears' },
  { 'number': 6, 'price': 500, 'name': 'Condoms' },
  { 'number': 7, 'price': 120, 'name': 'Crackers' },
  { 'number': 8, 'price': 220, 'name': 'Potato chips' },
  { 'number': 9, 'price': 80,  'name': 'Small snack' }
]

# Tests
Test.assert_equals(vending_machine(p, 500, 8), { 'product': 'Potato chips', 'c': [ 200, 50, 20, 10 ] })
Test.assert_equals(vending_machine(p, 500, 1), { 'product': 'Orange juice', 'c': [ 200, 200 ] })
Test.assert_equals(vending_machine(p, 200, 7), { 'product': 'Crackers', 'c': [ 50, 20, 10 ] })
Test.assert_equals(vending_machine(p, 100, 9), { 'product': 'Small snack', 'c': [ 20 ] })
Test.assert_equals(vending_machine(p, 1000, 6), { 'product': 'Condoms', 'c': [ 500 ] })
Test.assert_equals(vending_machine(p, 250, 4), { 'product': 'Cookies', 'c': [] })
Test.assert_equals(vending_machine(p, 500, 0), 'Enter a valid product number')
Test.assert_equals(vending_machine(p, 90, 1), 'Not enough money for this product')
Test.assert_equals(vending_machine(p, 0, 0), 'Enter a valid product number')

# Translated from JavaScript
# Originally posted by Pustur



combinator =  lambda l, d='': eval('[{} {}]'.format('+"{}"+'.format(d).join('el{}'.format(i) for i in range(len(l))), ' '.join('for el{0} in l[{0}]'.format(i) for i in range(len(l)))), locals())
		
from itertools import combinations
word, num = 'ABCD', 3
sort = sorted(word)
for i in range(1, num+1):
	[print(''.join(w)) for w in list(combinations(sort, i))]
		

Test.assert_equals(combinator([['a']]), ['a'])
Test.assert_equals(combinator([['ab'], ['ba']]), ['abba'])
Test.assert_equals(combinator([['a', 'b']]), ['a', 'b'])
Test.assert_equals(combinator([['a', 'b'], ['c', 'd']]), ['ac', 'ad', 'bc', 'bd'])
Test.assert_equals(combinator([['a', 'b'], ['c', 'd'], ['e', 'f']]), ['ace', 'acf', 'ade', 'adf', 'bce', 'bcf', 'bde', 'bdf'])
Test.assert_equals(combinator([['a'], ['a', 'b'], 'abc']), ['aaa', 'aab', 'aac', 'aba', 'abb', 'abc'])
Test.assert_equals(combinator([['foo', 'bar'], ['baz', 'bamboo']], ' '), ['foo baz', 'foo bamboo', 'bar baz', 'bar bamboo'])
Test.assert_equals(combinator(['abcd', 'efgh', 'ijkl']), ['aei', 'aej', 'aek', 'ael', 'afi', 'afj', 'afk', 'afl', 'agi', 'agj', 'agk', 'agl', 'ahi', 'ahj', 'ahk', 'ahl', 'bei', 'bej', 'bek', 'bel', 'bfi', 'bfj', 'bfk', 'bfl', 'bgi', 'bgj', 'bgk', 'bgl', 'bhi', 'bhj', 'bhk', 'bhl', 'cei', 'cej', 'cek', 'cel', 'cfi', 'cfj', 'cfk', 'cfl', 'cgi', 'cgj', 'cgk', 'cgl', 'chi', 'chj', 'chk', 'chl', 'dei', 'dej', 'dek', 'del', 'dfi', 'dfj', 'dfk', 'dfl', 'dgi', 'dgj', 'dgk', 'dgl', 'dhi', 'dhj', 'dhk', 'dhl'])
Test.assert_equals(combinator([[]]), [])
Test.assert_equals(combinator([['a', 'b'], [], ['e', 'f']]), [])
Test.assert_equals(combinator([[], ['e', 'f']]), [])

Test.assert_equals(help_bobby(1),[[1]])
Test.assert_equals(help_bobby(2),[[1, 1], [1, 1]])
Test.assert_equals(help_bobby(5),[[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
Test.assert_equals(help_bobby(8),[[1, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 1]])

Test.assert_equals(kix_code("PostNL, Postbus 30250, 2500 GG 's Gravenhage"), "2500GG30250")
Test.assert_equals(kix_code("De Jong, Havendijk 13 hs, 1231 FZ POSTDAM"), "1231FZ13XHS")
Test.assert_equals(kix_code("B. Bartelds, Boerheem 46, 9421 MC Bovensmilde"), "9421MC46")
Test.assert_equals(kix_code("Huisman, Koninginneweg 182 B, 3331 CH Zwijndrecht"), "3331CH182XB")
Test.assert_equals(kix_code("Liesanne B Wilkens, Kogge 11-1, 1657 KA Abbekerk"), "1657KA11X1")
Test.assert_equals(kix_code("Dijk, Antwoordnummer 80430, 2130 VA Hoofddorp"), "2130VA80430")
Test.assert_equals(kix_code("Van Eert, Dirk van Heinsbergstraat 200-A, 5575 BM Luyksgestel"), "5575BM200XA")
Test.assert_equals(kix_code("B.C. Dieker, Korhoenlaan 130b, 3847 LN Harderwijk"), "3847LN130B")
Test.assert_equals(kix_code("Mahir F Schipperen, IJsselmeerlaan 31, 8304 DE Emmeloord"), "8304DE31")
Test.assert_equals(kix_code("Jet de Wit, Wielingenstraat 129/7, 3522 PG Utrecht"), "3522PG129X7")

Test.assert_equals(spotlight_map([
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]), [
  [12, 21, 16],
  [27, 45, 33],
  [24, 39, 28]
])

Test.assert_equals(spotlight_map([
	[2, 6, 1, 3, 7],
	[8, 5, 9, 4, 0]
]), [
	[21, 31, 28, 24, 14],
	[21, 31, 28, 24, 14]
])

Test.assert_equals(spotlight_map([
	[7, 8],
	[3, 1],
	[4, 2],
	[0, 5],
	[9, 6]
]), [
	[19, 19],
	[25, 25],
	[15, 15],
	[26, 26],
	[20, 20]
])

Test.assert_equals(spotlight_map([
	[5, 5, 5, 5, 5],
	[5, 5, 5, 5, 5],
  [5, 5, 5, 5, 5],
  [5, 5, 5, 5, 5],
  [5, 5, 5, 5, 5]
]), [
	[20, 30, 30, 30, 20],
	[30, 45, 45, 45, 30],
	[30, 45, 45, 45, 30],
	[30, 45, 45, 45, 30],
	[20, 30, 30, 30, 20]
])

Test.assert_equals(spotlight_map([[3]]), [[3]])
Test.assert_equals(spotlight_map([[]]), [[]])
Test.assert_equals(spotlight_map([]), [])

Test.assert_equals(all_explode([
	["+", "+", "+", "+", "+", "+", "x"]
]), True)

Test.assert_equals(all_explode([
	["+", "+", "+", "+", "+", "0", "x"]
]), False)

Test.assert_equals(all_explode([
	["+", "+", "0", "x", "x", "+", "0"],
	["0", "+", "+", "x", "0", "+", "x"]
]), False)

Test.assert_equals(all_explode([
  ["x", "0", "0"],
  ["0", "0", "0"],
  ["0", "0", "x"]
]), False)

Test.assert_equals(all_explode([
  ["x", "0", "x"],
  ["0", "x", "0"],
  ["x", "0", "x"]
]), True)

Test.assert_equals(all_explode([
  ["x", "+", "x"],
  ["+", "x", "+"],
  ["x", "+", "x"]
]), False)

Test.assert_equals(all_explode([
  ["x", "+", "+"],
  ["+", "x", "+"],
  ["+", "+", "x"]
]), True)

Test.assert_equals(all_explode([
	["+", "+", "+", "+", "+", "+", "+"],
	["x", "+", "+", "+", "+", "+", "+"],
	["+", "x", "+", "+", "+", "+", "+"]
]), False)

Test.assert_equals(all_explode([
	["x", "0", "x", "0", "+", "+"],
	["+", "+", "0", "x", "0", "+"],
	["x", "0", "x", "0", "0", "x"],
	["+", "+", "0", "x", "x", "+"],
	["x", "0", "x", "0", "0", "+"],	
	["x", "+", "+", "0", "0", "+"]	
]), True)

Test.assert_equals(all_explode([
	["x", "0", "x", "0", "+", "+"],
	["+", "+", "0", "x", "0", "+"],
	["x", "0", "x", "0", "0", "x"],
	["+", "+", "0", "x", "x", "+"],
	["x", "0", "x", "0", "0", "x"],	
	["x", "+", "+", "0", "0", "+"]	
]), False)

c = dict(zip('MDCLXVI0', [1000, 500, 100, 50, 10, 5, 1, 0]))
d = dict(zip([1000, 500, 100, 50, 10, 5, 1], 'MDCLXVI'))
def roman_numerals(args):
	if isinstance(args, str):
		res = 0
		i = 0
		if len(args) == 1: return c[args]
		while i <= len(args)-1:
			j = args[i]; k=args[i+1] if i+1<len(args) else '0'
			if c[k]<=c[j]: 
				res += c[j]
				i+=1
			elif c[k]>c[j]:
				i+=2
				res += c[k]-c[j]
		return res
	else:
		res = ''
		for i, j in enumerate(str(args)):
			if int(j) <= 5: res+=''
		
column = lambda a: sum((string.ascii_uppercase.index(j)+1)*26**i for i, j in enumerate(reversed(a)))

Test.assert_equals(column("A"), 1)
Test.assert_equals(column("B"), 2)
Test.assert_equals(column("Z"), 26)
Test.assert_equals(column("AA"), 27)
Test.assert_equals(column("BA"), 53)
Test.assert_equals(column("BB"), 54)
Test.assert_equals(column("CW"), 101)
Test.assert_equals(column("DD"), 108)
Test.assert_equals(column("PQ"), 433)
Test.assert_equals(column("ZZ"), 702)
Test.assert_equals(column("ABC"), 731)
Test.assert_equals(column("ZZT"), 18272)
Test.assert_equals(column("STVW"), 348059)

print(column('CV'))

def get_alpha(num):
	result = ''
	#print([((string.ascii_uppercase.index(j)+1), 26, i) for i, j in enumerate(reversed('ABC'))])
	curr = 0

	print(num)

	num -= 3*26**curr
	curr += 1

	num -= 2*26**curr
	curr += 1

	num -= 1*26**curr
	curr += 1
	#print((3*26**0)+(2*26**1)+(1*26**2))
	

print(get_alpha(731))


Test.assert_equals(roman_numerals('I'), 1)
Test.assert_equals(roman_numerals('V'), 5)
Test.assert_equals(roman_numerals('X'), 10)
Test.assert_equals(roman_numerals('L'), 50)
Test.assert_equals(roman_numerals('C'), 100)
Test.assert_equals(roman_numerals('D'), 500)
Test.assert_equals(roman_numerals('M'), 1000)
Test.assert_equals(roman_numerals('IV'), 4)
Test.assert_equals(roman_numerals('VI'), 6)
Test.assert_equals(roman_numerals('XIV'), 14)
Test.assert_equals(roman_numerals('LIX'), 59)
Test.assert_equals(roman_numerals('XCIX'), 99)
Test.assert_equals(roman_numerals('CII'), 102)
Test.assert_equals(roman_numerals('XLV'), 45)
Test.assert_equals(roman_numerals('XXX'), 30)
Test.assert_equals(roman_numerals('XXXVI'), 36)
Test.assert_equals(roman_numerals('DCCXIV'), 714)
Test.assert_equals(roman_numerals('MMXVIII'), 2018)
Test.assert_equals(roman_numerals('MDCLXVI'), 1666)
Test.assert_equals(roman_numerals('MCCCXXIV'), 1324)

Test.assert_equals(floyd(up_to=1), [[1]])
Test.assert_equals(floyd(up_to=2), [[1], [2, 3]])
Test.assert_equals(floyd(up_to=7), [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10]])
Test.assert_equals(floyd(up_to=9), [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10]])
Test.assert_equals(floyd(up_to=15), [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14, 15]])
Test.assert_equals(floyd(up_to=50), [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21], [22, 23, 24, 25, 26, 27, 28], [29, 30, 31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42, 43, 44, 45], [46, 47, 48, 49, 50, 51, 52, 53, 54, 55]])
Test.assert_equals(floyd(n_row=1), [[1]])
Test.assert_equals(floyd(n_row=2), [[1], [2, 3]])
Test.assert_equals(floyd(n_row=5), [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14, 15]])
Test.assert_equals(floyd(n_row=6), [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21]])
Test.assert_equals(floyd(n_row=11), [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21], [22, 23, 24, 25, 26, 27, 28], [29, 30, 31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42, 43, 44, 45], [46, 47, 48, 49, 50, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]])

Test.assert_equals(special_reverse_string("Stee hsely tsgn IDA csacs Naemsscta Htnwo Nks'ti"), "It's known that CSS means Cascading Style Sheets")
Test.assert_equals(equal_count("Peter!@#$Paul&*#Peter!--@|#$Paul#$Peter@|Paul$%^^Peter++Paul%$%^Peter++Paul#$#$#Peter@|Paul", "Peter&Paul"), {'Peter': 6, 'Paul': 6, 'equality': True})