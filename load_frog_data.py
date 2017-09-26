import csv
import numpy as np

label_names = ['AdenomeraAndre', 'AdenomeraHylaedactylus', 'Ameeregatrivittata', 'HylaMinuta', 'HypsiboasCinerascens', 'HypsiboasCordobae',
               'LeptodactylusFuscus', 'OsteocephalusOophagus', 'Rhinellagranulosa', 'ScinaxRuber']

def load_frog_data():

    with open("data/Frogs_MFCCs.csv") as f:
        reader = csv.reader(f, delimiter=",")

        X = []
        y = []
        for i, row in enumerate(reader):
            if i ==0:
                continue

            X.append(row[:-4])
            y.append(label_names.index(row[-2]))

    return np.array(X).astype(float), np.array(y).astype(int)