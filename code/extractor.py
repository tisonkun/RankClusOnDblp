import sys
import lxml.etree as ET

xmlfile = sys.argv[1]
dtdfile = sys.argv[2]
csvfile = sys.argv[3]
dtd = ET.DTD(file=dtdfile)

records = list()
for event, element in ET.iterparse(xmlfile, load_dtd=True):
  if element.tag == 'article':
    try:
      year = int(element.find('year').text)
      if 2010 < year <= 2018:
        record = dict()
        record['year'] = year
        record['authors'] = [author.text for author in element.findall("author")]
        record['journal'] = element.find('journal').text
        records.append(record)
    except AttributeError:
      continue

import collections
authority = collections.defaultdict(int)
for record in records:
  for author in record['authors']:
    authority[author] += 1

import heapq
authorities = heapq.nlargest(20000, authority, lambda x: authority[x])

out = open(csvfile, "w")
for record in records:
  authors = list(filter((lambda x: x in authorities), record['authors']))
  if len(authors) > 0:
    out.write(record['journal'])
    out.write('$')
    out.write(';'.join(authors))
    out.write("\n")
