def pie_chart(data):
    total, test=sum(list(map(int, str(data.values()).replace('dict_values([','').replace('])','').split(', ')))), list(map(int, str(data.values()).replace('dict_values([','').replace('])','').split(', ')))
    for i in range(len(test)):
        data[list(data.keys())[i]]=int(data[list(data.keys())[i]]) if str(data[list(data.keys())[i]]).endswith('.0') else round(data[list(data.keys())[i]]*(360/total),1)
    return data
