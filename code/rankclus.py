import sys
import pandas as pd

csvfile = sys.argv[1]
data = pd.read_csv(csvfile, sep='$', names=['journal', 'authors'])
# csvfile = 'data/extract.csv'

journals = set(data['journal'])
authors = set(";".join(data['authors'].tolist()).split(";"))

data['authors'] = data['authors'].apply(lambda x: x.split(';'))
