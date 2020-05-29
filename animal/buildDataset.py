from myLibrary.util import OldDataset
import xml.etree.ElementTree as ET
import os

img_list = []
def read_data(db, img_db):
    X = []
    Y = []

    for file in os.listdir(db):
        tree = ET.parse(db+'/'+file)
        root = tree.getroot()
        object = root.find('object')
        Y.append(object.find('name').text)
        filename = root.find('filename').text
        for files in os.listdir(img_db):
            if files != 'label' and os.path.isdir(img_db + files + '/'):
                for img in os.listdir(img_db + files + '/'):
                    if img.find(filename) != -1:
                        X.append(img_db + files + '/' + img)
    return X, Y

img_db = "animal_train_data/"
db = "animal_train_data/label"
dataX, dataY = read_data(db, img_db)
print(dataX)
print(dataY)