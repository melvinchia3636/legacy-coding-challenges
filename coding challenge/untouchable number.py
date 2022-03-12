def aliquotSum(n) : 
    sm = 0
    for i in range(1,n) : 
        if (n % i == 0) : 
            sm = sm + i      
      
    return sm # return sum 

def is_untouchable(number):
    num = []
    if number >= 2:
        for i in range(2, (number ** 2) +1):
            if number == aliquotSum(i):
                num.append(i)
    else:
        return 'Invalid Input'
    return True if not num else num
	
print(is_untouchable(2))
print(is_untouchable(3))
print(is_untouchable(6))
print(is_untouchable(1))
print(is_untouchable(5))
print(is_untouchable(8))
print(is_untouchable(52))
print(is_untouchable(30))
print(is_untouchable(-10))
print(is_untouchable(188))
print(is_untouchable(60))
print(is_untouchable(100))
print(is_untouchable(120))
print(is_untouchable(121))
print(is_untouchable(0))
