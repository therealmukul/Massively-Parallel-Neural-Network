f = open("times.txt")

s = 0.0
count = 0
l = []
for num in f:
    val = float(num)
    s += float(num)
    l.append(val)
    count += 1

print "Max %f" % max(l)
print "Min %f" % min(l)
print "Avg %f" % (s / count)
