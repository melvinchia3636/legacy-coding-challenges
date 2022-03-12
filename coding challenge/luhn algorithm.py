def valid_credit_card(number):
    global reversed_num
    number = [int(i) for i in list(str(number))]
    reversed_num = list(reversed(number))
    for i in range(len(reversed_num)):
        if (i+1) % 2 == 0:
            reversed_num[i] *= 2
    reversed_num = [str(i) for i in reversed_num]
    for i in range(len(reversed_num)):
        if len(reversed_num[i]) == 2:
            reversed_num[i] = [int(i) for i in list(reversed_num[i])]
            reversed_num[i] = sum(reversed_num[i])

    reversed_num = [int(i) for i in reversed_num]    
            
    if sum(reversed_num) % 10 == 0:
        return True
    else:
        return False

valid_credit_card(2111111111121111)
valid_credit_card(4111111111111111)
valid_credit_card(5500000000000004)
valid_credit_card(378282246310005)
valid_credit_card(7777777777777777)
valid_credit_card(6011000000000004)
valid_credit_card(6451623895684318)
