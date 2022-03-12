def tallest_skyscraper(lst):
    tall, count1 = [],[];[tall.append([]) for i in range(len(lst[0]))];[[tall[row].append(lst[col][row]) for col in range(len(lst))] for row in range(len(lst[0]))];[count1.append(tall[i].count(1)) for i in range(len(tall))];return max(count1)
