# Using scipy requires that you install scipy.
# sudo apt-get install python-scipy

import os, glob, sys
from scipy import misc

def getMatrix(filename):
	image= misc.imread(filename, flatten = 1)
	return image



if __name__ == "__main__":
	for file in glob.glob("*.bmp"):
		# file is str. Filename.
		matrix = getMatrix(file)
		print("Processing "+file);
		file_to_write = open(file+".txt", 'w+')
		for i in range(0, 28):
			for j in range(0, 28):
				if (matrix[i][j] == 255):
					file_to_write.write("1")
				elif (matrix[i][j] == 0):
					file_to_write.write("0")
				if (j != 27):
					file_to_write.write(",")
				
			file_to_write.write('\n')
		file_to_write.close()
	
	

print("Finished.")
