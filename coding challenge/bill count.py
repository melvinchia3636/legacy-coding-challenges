def bill_count(money, bills):
    bills, min_bill = list(reversed(sorted(bills))), 0
    for i in bills:
        count = int(money/i)
        min_bill += count
        money -= count*i
    return min_bill

bill_count(1050, [1, 10, 20, 100])
