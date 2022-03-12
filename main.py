import math
import string
from typing import Union, List, Dict, Any, Callable

class Test(object):
	@staticmethod
	def assert_equals(first, second, desc=None):
		print(desc) if desc else None
		print('Output: {}'.format(first))
		print('Expect: {}'.format(second))
		print('Correct: {}'.format(first == second))
		print()

'''
Interquartile Range (IQR)
The median of a data sample is the value that separates the higher half and the lower half of the data. For example, the median of [1, 2, 3] is 2, and the median of [1, 2, 3, 4] is 2.5 (because (2 + 3) / 2 = 2.5). Another way of saying "median" is to say "Q2" (it's the second quartile). Q1 and Q3 are the medians of the values above or below the Q2. The IQR is equal to Q3 - Q1. Here's an example:

Let's say your data sample is: 1, 2, 3, 4

The median (Q2) is: (2+3)/2 =2.5
The lower half is: 1, 2
The upper half is: 3, 4
Q1 (median of the first half): (1+2)/2 = 1.5
Q3 (median of the second half): (3+4)/2 = 3.5
IQR = Q3 - Q1 = 3.5 - 1.5 = 2
If the length of the data sample is odd, as in: 1, 2, 3, 4, 5

The median (Q2) is: 3 (the number is in the middle, so no need to average).
3 is the number that separates the upper and lower data, so we exclude it.
The lower half is: 1, 2
The upper half is: 4, 5
Q1 (median of the first half): (1+2)/2 = 1.5
Q3 (median of the second half): (4+5)/2 = 4.5
IQR = Q3 - Q1 = 4.5 - 1.5 = 3
For a more detailed explanation, please check the Resources tab.

Make a function that takes a list of floats and/or integers and returns the IQR for that list. The return type can be float or int. It doesn't matter.

Examples
iqr([1, 2, 3, 4]) ➞ 2.0

iqr([5, 4, 3, 2, 1]) ➞ 3.0

iqr([-3.1, -3.8, -14, 23, 0]) ➞ 20.4
'''

class iterquartile_range:
	def find_median(lst):
		return (lst[int(len(lst)/2)]+lst[int(len(lst)/2-1)])/2 if len(lst) % 2 == 0 else lst[math.ceil(len(lst) / 2) - 1]

	def iqr(lst):
		lst = sorted(lst)
		if len(lst) % 2 == 0:  # even
			median = int(len(lst) / 2)
			upper = lst[median:]
			lower = lst[:median]
		else:
			median = math.ceil(len(lst) / 2)
			upper = lst[median:]
			lower = lst[:median - 1]
		Q1 = find_median(lower)
		Q3 = find_median(upper)
		return Q3 - Q1

"""
Caesar's Cipher
Julius Caesar protected his confidential information by encrypting it using a cipher. Caesar's cipher (check Resources tab for more info) shifts each letter by a number of letters. If the shift takes you past the end of the alphabet, just rotate back to the front of the alphabet. In the case of a rotation by 3, w, x, y and z would map to z, a, b and c.

Create a function that takes a string s (text to be encrypted) and an integer k (the rotation factor). It should return an encrypted string.

Examples
caesar_cipher("middle-Outz", 2) ➞ "okffng-Qwvb"

# m -> o
# i -> k
# d -> f
# d -> f
# l -> n
# e -> g
# -	-
# O -> Q
# u -> w
# t -> v
# z -> b

caesar_cipher("Always-Look-on-the-Bright-Side-of-Life", 5)
➞ "Fqbfdx-Qttp-ts-ymj-Gwnlmy-Xnij-tk-Qnkj"

caesar_cipher("A friend in need is a friend indeed", 20)
➞ "U zlcyhx ch hyyx cm u zlcyhx chxyyx"
"""

class caesar_chiper:
	def caesar_cipher_2(self, s, k):
		k=k%26 if k > 26 else k; return ''.join((string.ascii_lowercase[string.ascii_lowercase.index(i)-1+k] if string.ascii_lowercase.index(i)+1+k <=26 else string.ascii_lowercase[(string.ascii_lowercase.index(i)+k)%26]) if i.islower() else (string.ascii_uppercase[string.ascii_uppercase.index(i)-1+k] if string.ascii_uppercase.index(i)+1+k <=26 else string.ascii_uppercase[(string.ascii_uppercase.index(i)+k)%26]) if i.isupper() else i for i in s)
	#old lol lol lol
	def caesar_cipher(self, s, k):
		uppercase = string.ascii_uppercase
		lowercase = string.ascii_lowercase
		result = []
		for i in s:
			if not i.isupper() and not i.islower():
				result.append(i)
				continue
			current = i
			for j in range(k):
				if i.isupper():
					index = uppercase.index(current)
					index  = index + 1 if index < len(uppercase)-1 else 0
					current = uppercase[index]
				elif i.islower():
					index = lowercase.index(current)
					index  = index + 1 if index < len(lowercase)-1 else 0
					current = lowercase[index]
			result.append(current)
		return ''.join(result)

	def check(self):
		print(Test.assert_equals(self, self.caesar_cipher("middle-Outz", 2), "okffng-Qwvb"))
		print(Test.assert_equals(self, self.caesar_cipher("Always-Look-on-the-Bright-Side-of-Life", 5), "Fqbfdx-Qttp-ts-ymj-Gwnlmy-Xnij-tk-Qnkj"))
		print(Test.assert_equals(self, self.caesar_cipher("A friend in need is a friend indeed", 20), "U zlcyhx ch hyyx cm u zlcyhx chxyyx"))
		print(Test.assert_equals(self, self.caesar_cipher("A Fool and His Money Are Soon Parted.", 27), "B Gppm boe Ijt Npofz Bsf Tppo Qbsufe."))
		print(Test.assert_equals(self, self.caesar_cipher("One should not worry over things that have already happened and that cannot be changed.", 49), "Lkb pelria klq tloov lsbo qefkdp qexq exsb xiobxav exmmbkba xka qexq zxkklq yb zexkdba."))
		print(Test.assert_equals(self, self.caesar_cipher( "Back to Square One is a popular saying that means a person has to start over, similar to: back to the drawing board.", 126), "Xwyg pk Omqwna Kja eo w lklqhwn owuejc pdwp iawjo w lanokj dwo pk opwnp kran, oeiehwn pk: xwyg pk pda znwsejc xkwnz."))

test = caesar_chiper()
test.check()
'''
Eight Sums Up
Create a function that gets every pair of numbers from a list that sums up to eight and returns it as a list of pairs (pair sorted ascendingly) collated into an object.

Examples
sums_up([1, 2, 3, 4, 5]) ➞ {"pairs": [[3, 5]]}

sums_up([10, 9, 7, 2, 8]) ➞ {"pairs": []}

sums_up([1, 6, 5, 4, 8, 2, 3, 7]) ➞ {"pairs": [[2, 6], [3, 5], [1, 7]]}
# [6, 2] first to complete the cycle (to sum up to 8)
# [5, 3] follows
# [1, 7] lastly
# [2, 6], [3, 5], [1, 7] sorted according to cycle completeness, then pair-wise
Notes
Remember the idea of "completes the cycle first" when getting the sort order of pairs.
Only unique numbers are present in the list.
'''

class sums_up_to_eight:
	def sums_up(self, lst):
		final = []; [final.append(i) for i in [sorted(i[0]) for i in reversed([[[i, j] for j in lst if i + j == 8 and i != j] for i in lst]) if i] if i not in final]; return {"pairs": list(reversed(final))}

	def check(self):
		print(Test.assert_equals(sums_up([1, 2, 3, 4, 5]), {"pairs": [[3, 5]]}))
		print(Test.assert_equals(sums_up([10, 9, 7, 2, 8]), {"pairs": []}))
		print(Test.assert_equals(sums_up([1, 6, 5, 4, 8, 2, 3, 7]), {"pairs": [[2, 6], [3, 5], [1, 7]]}))
		print(Test.assert_equals(sums_up([5, 7, 2, 3, 0, 1, 6, 4, 8]), {"pairs": [[3, 5], [1, 7], [2, 6], [0, 8]]}))
		print(Test.assert_equals(sums_up([10, 9, 7, 1, 8, -2, -1, 2, 6]), {"pairs": [[1, 7], [-2, 10], [-1, 9], [2, 6]]}))
		print(Test.assert_equals(sums_up([0, 1, -2, 7, 9, 5, 4, 10, 8, -1, 6]), {"pairs": [[1, 7], [-2, 10], [0, 8], [-1, 9]]}))

'''
Pilish Strings
In this challenge, transform a string into a series of words (or sequences of characters) separated by a single space, with each word having the same length given by the first 15 digits of the decimal representation of Pi:

3.14159265358979
If a string contains more characters than the total quantity given by the sum of the Pi digits, the unused characters are discarded and you will use only those needed to form 15 words.

String = "HOWINEEDADRINKALCOHOLICINNATUREAFTERTHEHEAVYLECTURESINVOLVINGQUANTUMMECHANICSANDALLTHESECRETSOFTHEUNIVERSE"

Pi String = "HOW I NEED A DRINK ALCOHOLIC IN NATURE AFTER THE HEAVY LECTURES INVOLVING QUANTUM MECHANICS"

# Every word has the same length of the digit of Pi found at the same index ?
# "HOW" = 3, "I" = 1, "NEED" = 4, "A" = 1, "DRINK" = 5
# "ALCOHOLIC" = 9, "IN" = 2, "NATURE" = 6, "AFTER" = 5
# "THE" = 3, "HEAVY" = 5, "LECTURES" = 8, "INVOLVING" = 9
# "QUANTUM" = 7, "MECHANICS" = 9
# 3.14159265358979
Also if a string contains less characters than the total quantity given by the sum of the Pi digits, in any case you have to respect the sequence of the digits to obtain the words.

String = "FORALOOP"

Pi String = "FOR A LOOP"

# Every word has the same length of the digit of Pi found at the same index ?
# "FOR" = 3, "A" = 1, "LOOP" = 4
# 3.14
If the letters contained in the string don't match exactly the digits, in this case you will repeat the last letter until the word will have the correct length.

String = "CANIMAKEAGUESSNOW"

Pi String = "CAN I MAKE A GUESS NOWWWWWWW"

# Every word has the same length of the digit of Pi found at the same index ?
# "CAN" = 3, "I" = 1, "MAKE" = 4, "A" = 1, "GUESS" = 5, "NOW" = 3
# 3.14153 (Wrong!)
# The length of the sixth word "NOW" (3)...
# ...doesn't match the sixth digit of Pi (9)
# The last letter "W" will be repeated...
# ...until the length of the word will match the digit

# "CAN" = 3, "I" = 1, "MAKE" = 4, "A" = 1, "GUESS" = 5, "NOWWWWWWW" = 9
# 3.14159 (Correct!)
If the given string is empty, an empty string has to be returned.

Given a string txt, implement a function that returns the same string formatted accordingly to the above instructions.

Examples
pilish_string("33314444") ➞ "333 1 4444"
# 3.14

pilish_string("TOP") ➞ "TOP"
# 3

pilish_string("X")➞ "XXX"
# "X" has to match the same length of the first digit (3)
# The last letter of the word is repeated

pilish_string("")➞ ""
Notes
This challenge is a simplified concept inspired by the Pilish, a peculiar type of constrained writing that uses all the known possible digits of Pi. A potentially infinite text can be written allowing punctuation and a set of additional rules, that permits to avoid long sequences of small digits, substituting them with words bigger than 9 letters and making so appear the composition more similar to a free-verse poem.
The dot that separes the integer part of Pi from the decimal part hasn't to be considered in the function: it's present in Instructions and Examples only for readability.
'''

def pilish_string(txt): txt = list(txt); return ' '.join([''.join(i) for i in [[txt.pop(0) if len(txt) > 1 else txt.pop(0)*(i-j) for j in range(i) if len(txt) > 0] for i in [int(i) for i in list('314159265358979')]] if i != []])
'''
if __name__ == '__main__':
	print(Test.assert_equals(pilish_string("FORALOOP"), "FOR A LOOP"))
	print(Test.assert_equals(pilish_string("CANIMAKEAGUESS"), "CAN I MAKE A GUESS"))
	print(Test.assert_equals(pilish_string("CANIMAKEAGUESSNOW"), "CAN I MAKE A GUESS NOWWWWWWW"))
	print(Test.assert_equals(pilish_string("X"), "XXX"))
	print(Test.assert_equals(pilish_string("ARANDOMSEQUENCEOFLETTERS"), "ARA N DOMS E QUENC EOFLETTER SS"))
	print(Test.assert_equals(pilish_string(""), ""))
	print(Test.assert_equals(pilish_string("33314444155555999999999226666665555533355555888888889999999997777777999999999"), "333 1 4444 1 55555 999999999 22 666666 55555 333 55555 88888888 999999999 7777777 999999999"))
	print(Test.assert_equals(pilish_string("###*@"), "### * @@@@"))
	print(Test.assert_equals(pilish_string(".........."), "... . .... . ....."))
	# Below, quoting Mike Keith
	print(Test.assert_equals(pilish_string("NowIfallAtiredsuburbianInliquidunderthetreesDriftingalongsideforestssimm"), "Now I fall A tired suburbian In liquid under the trees Drifting alongside forests simmmmmmm"))
	# Below, quoting Sir James Hopwood Jeans
	print(Test.assert_equals(pilish_string("HOWINEEDADRINKALCOHOLICINNATUREAFTERTHEHEAVYLECTURESINVOLVINGQUANTUMMECHANICSANDALLTHESECRETSOFTHEUNIVERSE"), "HOW I NEED A DRINK ALCOHOLIC IN NATURE AFTER THE HEAVY LECTURES INVOLVING QUANTUM MECHANICS"))
	print(Test.assert_equals(pilish_string("HOWINEEDADRINKALCOHOLICINNATUREAFTERTHEHEAVYCODING"), "HOW I NEED A DRINK ALCOHOLIC IN NATURE AFTER THE HEAVY CODINGGG"))
'''










































'''
input: 
List product_list [Dict each_products {'number': Int product_number, 'price': product_price, 'name': Str product_name}]
Int money
Int product_number

output:
List result_list [Str product_name, List changes)] or Str error_message

notes:
I'm the second shortest code in this question :) but just one character less than third place :)

'''

def vending_machine(p: List[Dict[str, Any]], m: int, n: int) -> Union[Dict[str, Union[str, List[int]]], str]:
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

	return {'product': p[n-1]['name'], 'change': c} #return the final result

p: List[Dict[str, Union[str, int]]] = [
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

Test.assert_equals(vending_machine(p, 500, 8), { 'product': 'Potato chips', 'change': [ 200, 50, 20, 10 ] })
Test.assert_equals(vending_machine(p, 500, 1), { 'product': 'Orange juice', 'change': [ 200, 200 ] })
Test.assert_equals(vending_machine(p, 200, 7), { 'product': 'Crackers', 'change': [ 50, 20, 10 ] })
Test.assert_equals(vending_machine(p, 100, 9), { 'product': 'Small snack', 'change': [ 20 ] })
Test.assert_equals(vending_machine(p, 1000, 6), { 'product': 'Condoms', 'change': [ 500 ] })
Test.assert_equals(vending_machine(p, 250, 4), { 'product': 'Cookies', 'change': [] })
Test.assert_equals(vending_machine(p, 500, 0), 'Enter a valid product number')
Test.assert_equals(vending_machine(p, 90, 1), 'Not enough money for this product')
Test.assert_equals(vending_machine(p, 0, 0), 'Enter a valid product number')

'''
input:
Str text

output
List hint_letter [Str hints]

notes:
I'm the second shortest code in this question :)
'''

grant_the_hint: Callable[[str], List[str]] = lambda t: [' '.join(j[:i]+'_'*(len(j)-i) for j in t.split()) for i in range(len(max(t.split(), key=len))+1)]
	

Test.assert_equals(grant_the_hint('Mary Queen of Scots'), [
'____ _____ __ _____',
'M___ Q____ o_ S____',
'Ma__ Qu___ of Sc___',
'Mar_ Que__ of Sco__',
'Mary Quee_ of Scot_',
'Mary Queen of Scots'
])

Test.assert_equals(grant_the_hint('The Life of Pi'), [
'___ ____ __ __',
'T__ L___ o_ P_',
'Th_ Li__ of Pi',
'The Lif_ of Pi',
'The Life of Pi'
])


Test.assert_equals(grant_the_hint('The River Nile'), [
'___ _____ ____',
'T__ R____ N___',
'Th_ Ri___ Ni__',
'The Riv__ Nil_',
'The Rive_ Nile',
'The River Nile'
])

Test.assert_equals(grant_the_hint('The Colour Purple'), [
'___ ______ ______',
'T__ C_____ P_____',
'Th_ Co____ Pu____',
'The Col___ Pur___',
'The Colo__ Purp__',
'The Colou_ Purpl_',
'The Colour Purple'
])

Test.assert_equals(grant_the_hint('The Battle of Hastings'), [
'___ ______ __ ________', 
'T__ B_____ o_ H_______', 
'Th_ Ba____ of Ha______', 
'The Bat___ of Has_____', 
'The Batt__ of Hast____', 
'The Battl_ of Hasti___', 
'The Battle of Hastin__', 
'The Battle of Hasting_', 
'The Battle of Hastings'])

Test.assert_equals(grant_the_hint('Ant-Man and the Wasp'), [
'_______ ___ ___ ____', 
'A______ a__ t__ W___', 
'An_____ an_ th_ Wa__', 
'Ant____ and the Was_', 
'Ant-___ and the Wasp', 
'Ant-M__ and the Wasp', 
'Ant-Ma_ and the Wasp', 
'Ant-Man and the Wasp'
])

Test.assert_equals(grant_the_hint('A billion seconds is almost 32 years'), [
'_ _______ _______ __ ______ __ _____', 
'A b______ s______ i_ a_____ 3_ y____', 
'A bi_____ se_____ is al____ 32 ye___', 
'A bil____ sec____ is alm___ 32 yea__', 
'A bill___ seco___ is almo__ 32 year_', 
'A billi__ secon__ is almos_ 32 years', 
'A billio_ second_ is almost 32 years', 
'A billion seconds is almost 32 years'
])
