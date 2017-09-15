import sys

filename = sys.argv[1]

amap = {}

fo = open(sys.argv[1], 'r')
stringy = fo.read()
listy = stringy.split()
fo.close()
# listy = list(enumerate(stringy.split()))
n = 0
for e in listy:
    if e in amap:
        amap[e] = (amap[e][0], amap[e][1] + 1)
    else:
        amap[e] = (n, 1)
        n += 1

uslist = amap.items()

slist = sorted(uslist, key=lambda item: item[1][0])

out = open("Q1.txt", 'w')

for i in slist:
    out.write("{} {} {}\n".format(i[0], i[1][0], i[1][1]))

out.close()
