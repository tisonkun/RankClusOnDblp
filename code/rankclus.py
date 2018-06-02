from collections import defaultdict
from random import randint
from csv import reader
from sys import argv

import numpy as np
import multiprocessing

"""
Configure for algorithm.

@param K     : number of cluster
@param MT    : number of main iterate turn
@param RT    : number of ranking algorithm iterate turn
@param EMT   : number of EM algorithm iterate turn
@param alpha : learning rate
"""

K = 15
MT = 10
RT = 10
EMT = 5
alpha = 0.95

csvfile = argv[1]

"""
Compose network from extract.csv

@value journals       : set of all journals
@value authors        : set of all authors
@value author2author  : weight of edge, (author, author) -> int
@value author2journal : weight of edge, (author, journal) -> int
@value journal2author : weight of edge, (journal, author) -> int
"""

journals = set()
authors = set()

records = reader(open(csvfile), delimiter=";")
for record in records:
    journals.add(record[0])
    for author in record[1:]:
        authors.add(author)

print('ack')

author2author = { author : defaultdict(int) for author in authors }
author2journal = { author : defaultdict(int) for author in authors }
journal2author = { journal : defaultdict(int) for journal in journals }
records = reader(open(csvfile), delimiter=";")
for record in records:
    line_journal = record[0]
    line_authors = record[1:]
    for i in range(len(line_authors)):
        journal2author[line_journal][line_authors[i]] += 1
        author2journal[line_authors[i]][line_journal] += 1
        for j in range(i+1, len(line_authors)):
            author2author[line_authors[i]][line_authors[j]] += 1
            author2author[line_authors[j]][line_authors[i]] += 1

print('ack')

"""
Standalone functions.
"""

def init_clusters():
    """
    Randomly initialize clusters.

    @return : clusters of journals
    """

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
    @return         : whether no empty cluster
    """

    return all(len(cluster) > 0 for cluster in clusters.values())

def authority_rank(i, author_rank, journal_rank, cluster):
    """
    Authority rank algorithm, calculate for one cluster
    modify author_rank and journal_rank as side affect

    @variable internal_author_rank      : final author_rank
    @variable internal_journal_rank     : final journal_rank
    @variable internal_journal_rank_aux : journal_rank after step 1
    """

    internal_author_rank = defaultdict(float)
    internal_journal_rank = defaultdict(float)
    internal_journal_rank_aux = defaultdict(float)

    """
    Initialize author_rank(r_y)
    """
    for author in author2journal:
        internal_author_rank[author] = 1.0 / len(author2journal)

    """
    Rank loop
    """
    for _ in range(RT):
        """
        Calculate journal_rank in this cluster
                                 sum(author_rank * journal2author)
        Formula : journal_rank = -----------------------------------
                                        sum(all journal_rank)
        """
        sum_journal_rank = 0.0
        for journal in cluster:
            one_journal_rank = sum([
                journal2author[journal][author] * internal_author_rank[author]
                for author in journal2author[journal]
            ])
            internal_journal_rank_aux[journal] = one_journal_rank
            sum_journal_rank += one_journal_rank
        for journal in internal_journal_rank_aux:
            internal_journal_rank_aux[journal] /= sum_journal_rank

        """
        Calculate author_rank
                                alpha*sum(journal_rank * journal2author) + (1-alpha)*sum(author_rank*author2author)
        Formula : author_rank = ------------------------------------------------------------------------------------
                                                            sum(all author_rank)
        """
        internal_author_rank_copy = internal_author_rank.copy()
        for author in internal_author_rank:
            internal_author_rank[author] = 0.0
        for journal in cluster:
            for author in journal2author[journal]:
                internal_author_rank[author] += (
                    alpha * journal2author[journal][author] * internal_journal_rank_aux[journal]
                )
        for author in internal_author_rank:
            for coauthor in author2author[author]:
                internal_author_rank[author] += (
                    (1-alpha) * internal_author_rank_copy[coauthor] * author2author[author][coauthor]
                )
        sum_author_rank = sum(internal_author_rank.values())
        for author in internal_author_rank:
            internal_author_rank[author] /= sum_author_rank

    """
    Calculate all journal_rank
                                sum(author_rank * journal2author)
    Formula : journal_rank = -----------------------------------
                                    sum(all journal_rank)
    """
    sum_journal_rank = 0.0
    for journal in journal2author:
        one_journal_rank = sum([
            journal2author[journal][author] * internal_author_rank[author]
            for author in journal2author[journal]
        ])
        internal_journal_rank[journal] = one_journal_rank
        sum_journal_rank += one_journal_rank
    for journal in internal_journal_rank:
        internal_journal_rank[journal] /= sum_journal_rank

    author_rank[i], journal_rank[i] = internal_author_rank, internal_journal_rank


"""
End standalone functions.
"""

"""
Main iterate configure

@value    manager  : process manager for multiprocessing
@variable mt       : number of turns, bound to MT
@variable clusters : clusters of journals
"""
clusters = init_clusters()
print('ack')

manager = multiprocessing.Manager()

mt = 0
while mt < MT:
    print("Ranking Turn %d" % (mt))

    """
    Main iterate loop
    """

    """
    Authority rank algorithm, do in parallel

    @value    pool         : process pool for multiprocessing
    @variable author_rank  : rank of authors as vector
    @variable journal_rank : rank of journal as vector
    """
    author_rank = manager.dict()
    journal_rank = manager.dict()
    pool = multiprocessing.Pool(processes=K)
    
    for i in range(K):
        pool.apply_async(authority_rank, (i, author_rank, journal_rank, clusters[i]))
    
    pool.close()
    pool.join()
    author_rank = dict(author_rank)
    journal_rank = dict(journal_rank)
