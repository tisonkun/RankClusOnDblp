import sys
import csv

from pprint import pprint

"""
Configure for algorithm.

@param K     : number of cluster
@param MT    : number of main iterate turn
@param RT    : number of ranking algorithm iterate turn
@param EMT   : number of EM algorithm iterate turn
@param alpha : learning rate
"""

K = 15
MT = 25
RT = 10
EMT = 5
alpha = 0.95

csvfile = sys.argv[1]

"""
Compose network from pandas.DataFrame.

@value journals       : set of all journals
@value authors        : set of all authors
@value author2author  : weight of edge, (author, author) -> int
@value author2journal : weight of edge, (author, journal) -> int
@value journal2author : weight of edge, (journal, author) -> int
"""

journals = set()
authors = set()

records = csv.reader(open(csvfile), delimiter=";")
for record in records:
    journals.add(record[0])
    for author in record[1:]:
        authors.add(author)

pprint('ack')

from collections import defaultdict    
author2author = { author : defaultdict(int) for author in authors }
author2journal = { author : defaultdict(int) for author in authors }
journal2author = { journal : defaultdict(int) for journal in journals }
records = csv.reader(open(csvfile), delimiter=";")
for record in records:
    line_journal = record[0]
    line_authors = record[1:]
    for i in range(len(line_authors)):
        journal2author[line_journal][line_authors[i]] += 1
        author2journal[line_authors[i]][line_journal] += 1
        for j in range(i+1, len(line_authors)):
            author2author[line_authors[i]][line_authors[j]] += 1
            author2author[line_authors[j]][line_authors[i]] += 1

pprint('ack')

"""
Standalone functions.
"""

def init_clusters():
    """
    Randomly initialize clusters.

    @return clusters : clusters of journals
    """

    from random import randint
    from collections import defaultdict
    clusters = defaultdict(list)
    while True:
        for journal in journals:
            clusters[randint(0, K-1)].append(journal)
        if validate_clusters(clusters): break
    return clusters

def validate_clusters(clusters):
    """
    Prevent cluster from being empty.

    @param clusters : clusters to be validated
    @return <bool>  : whether no empty cluster
    """
    return all(len(cluster) > 0 for cluster in clusters.values())

"""
End standalone functions.
"""

clusters = init_clusters()
pprint(clusters)
pprint('ack')