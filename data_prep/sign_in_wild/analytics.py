import pandas as pd
import more_itertools as mit

cols = ['ids','frame','label']
gt = pd.read_csv('data_prep/sign_in_wild/groundtruth.txt',names=cols,delimiter=' ',header=None)
print()