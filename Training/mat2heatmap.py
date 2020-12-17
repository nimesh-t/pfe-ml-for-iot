import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--savename", type=str, default="heatmap_default.png",
	help="save_name")
args = vars(ap.parse_args())

SAVE_NAME = args['savename']#'array_VGG16_6_asos2000_3_last'

#read matrix from text as list
with open(SAVE_NAME+'.txt', 'r') as f:
    matrix = [[int(num) for num in line.split(' ')] for line in f]

matrix = np.array(matrix)

normalise=True

if normalise:
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:,np.newaxis]
    accuracy=np.trace(matrix)/len(matrix)
    
# print(matrix)

Type=["Coat","Dress","Jacket","Shirt","Skirt","Sweatshirt","Trousers"]
Colours=["Black","Blue","Brown","Green","Grey","Orange","Pink","Purple","Red","White","Yellow"]
Class = Type


fig, ax = plt.subplots()
plt.imshow(matrix,interpolation='nearest')

ax.set_xticks(np.arange(len(Class)))
ax.set_yticks(np.arange(len(Class)))
ax.set_xticklabels(Class)
ax.set_yticklabels(Class)
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

fmt = '.2f' if normalise else 'd'
for i in range(len(Class)):
    for j in range(len(Class)):
        text = ax.text(j, i, format(matrix[i, j],fmt),ha="center", va="center", color="w")

if normalise:
    ax.set_title("Normalised Confusion Matrix\n Validation Accuracy = "+"{:.3f}".format(accuracy))
else:
    ax.set_title("Confusion Matrix")
fig.tight_layout()
plt.colorbar()
plt.savefig(SAVE_NAME+'.png')
#plt.show()
