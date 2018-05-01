import sys
import pandas as pd

from pprint import pprint

def main():
    csvfile = sys.argv[1]
    data = pd.read_csv(csvfile, sep='$', names=['journal', 'authors'])

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

    """
    Compose network from pandas.DataFrame.

    @value journals       : set of all journals
    @value authors        : set of all authors
    @value author2author  : weight of edge, (author, author) -> int
    @value author2journal : weight of edge, (author, journal) -> int
    @value journal2author : weight of edge, (journal, author) -> int
    """

    from collections import defaultdict
    journals = set(data['journal'])
    authors = set(";".join(data['authors'].tolist()).split(";"))
    author2author = { author : defaultdict(int) for author in authors }
    author2journal = { author : defaultdict(int) for author in authors }
    journal2author = { journal : defaultdict(int) for journal in journals }

    from itertools import combinations
    data['authors'] = data['authors'].apply(lambda x: x.split(';'))
    for _, row in data.iterrows():
        for author in row['authors']:
            author2journal[author][row['journal']] += 1
            journal2author[row['journal']][author] += 1
        for a, b in combinations(row['authors'], 2):
            author2author[a][b] += 1
            author2author[b][a] += 1

    pprint('ack')
    # pprint(author2author)

    """
    Randomly initialize clusters.

    @variable clusters : clusters of journals
    """

    from random import randint
    clusters = None
    while True:
        clusters = [[ ] for _ in range(K)]
        for journal in journals:
            clusters[randint(0, K-1)].append(journal)
        if validate_clusters(clusters): break

    pprint(clusters)

"""
Standalone functions.
"""

def validate_clusters(clusters):
    """
    Prevent cluster from being empty.

    @param clusters : clusters to be validated
    """
    return all(len(cluster) > 0 for cluster in clusters)

"""
Trigger main function.
"""
main()
