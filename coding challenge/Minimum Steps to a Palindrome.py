def min_palindrome_steps(txt):
    if len(txt) < 2 or list(txt) == list(reversed(txt)): return 0
    count = 0
    return [len(i) for i in list(reversed([''.join(list(reversed(txt)))[i:] for i in range(len(txt))])) if list(txt+i) == list(reversed(txt+i))][0]
    
print(min_palindrome_steps("race"))
print(min_palindrome_steps("mada"))
print(min_palindrome_steps("mirror"))
print(min_palindrome_steps("maa"))
print(min_palindrome_steps("m"))
print(min_palindrome_steps("rad"))
print(min_palindrome_steps("madam"))
print(min_palindrome_steps("radar"))
print(min_palindrome_steps("www"))
print(min_palindrome_steps("me"))
print(min_palindrome_steps("rorr"))
print(min_palindrome_steps("pole"))
