import csv

f = open('../Memorability_data/database/Memorability.csv', 'rt')
train_dataReader = csv.reader(f)
train_data = [ e for e in train_dataReader]
f.close()

List = []

for i in train_data:
    List.append( [ i[0], int(float(i[1])/10) ] )

f = open('../Memorability_data/database/hotMemorability.csv', 'w')
writer = csv.writer(f)
for i in List:
        writer.writerow(i)
f.close()
