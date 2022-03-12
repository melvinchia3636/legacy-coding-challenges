def uncensor(txt, vowels):
    vlst,txtlst,done = list(vowels), list(txt), False
    for i in vlst:
        done=False
        for v in range(len(txtlst)):
            if txtlst[v]=='*' and done==False:
                txtlst[v]=i;done=True    
    return ''.join(txtlst)
