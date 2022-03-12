import datetime
def sort_dates(lst, mode):
    return sorted(lst, key= lambda x: datetime.datetime.strptime(x, '%d-%m-%Y_%H:%M')) if mode == 'ASC' else sorted(lst, key= lambda x: datetime.datetime.strptime(x, '%d-%m-%Y_%H:%M'), reverse=True)
