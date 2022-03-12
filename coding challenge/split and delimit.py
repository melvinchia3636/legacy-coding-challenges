def split_and_delimit(txt, num, delimiter):
    return ''.join([txt[i:i+num] for i in range(len(txt))[::num]])
    
split_and_delimit("magnify", 3, ':')
