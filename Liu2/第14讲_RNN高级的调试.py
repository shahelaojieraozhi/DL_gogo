import gzip
import csv

'''Data Preparing'''
filename = './dataset/names_train.csv.gz'
with gzip.open(filename, 'rt') as f:
    reader = csv.reader(f)
    rows = list(reader)

# print(rows)
names = [row[0] for row in rows]
countries = [row[1] for row in rows]
country_list = sorted(set(countries))


# print(country_list)


def getCountryDict():
    country_dict = dict()
    for idx, country_name in enumerate(country_list, 0):
        country_dict[country_name] = idx
    return country_dict


# country_list = [Arabic, Chinese, Czech, Dutch.....]
# country_dict = {'Arabic': 0, 'Chinese': 1, 'Czech': 2, 'Dutch': 3......}

country_dict = getCountryDict()


# print(country_dict)


def getitem(index):
    return names[index], country_dict[countries[index]]
    # Country相当于 key, Index 相当于 value


print(getitem(2))


