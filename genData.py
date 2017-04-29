num_classes = 100
times = 1

genInput = open("genTestInput.txt", "w")
genOutput = open("genTestLabels.txt", "w")

for k in range(times):
    for i in range(num_classes):
        line = ""
        for j in range(num_classes):
            if (i == j):
                if (j == num_classes - 1):
                    line += "1"
                else:
                    line += "1,"
            else:
                if (j == num_classes-1 ):
                    line += "0"
                else:
                    line += "0,"

        genInput.write(line)
        genInput.write('\n')

        genOutput.write(str(i + 1))
        genOutput.write('\n')
