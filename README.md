# A Demo of Running RankClus Algorithm on Dblp

## Synopsis

This repository holds a demo of running [RankClus Algorithm](https://openproceedings.org/2009/conf/edbt/SunHZYCW09.pdf) on dblp data set.

## XML Extract

Require Python 3 and package lxml.

The extractor is code/extractor.py.

XML and DTD file are too large, so you'd better download it from http://dblp.org/xml/release/. This repository is based on [dblp-2018-04-01.xml](http://dblp.org/xml/release/dblp-2018-04-01.xml.gz) with its DTD file [dblp-2017-08-29.dtd](http://dblp.org/xml/release/dblp-2017-08-29.dtd).

Sample extracted result file extract.csv is under data/.

You can modify extractor.py and extract by your self.

## Run the Algorithm

```
$ python3 code/rankclus.py data/extract.csv
```

This requires python3 lib numpy and so on.

## License

See [LICENSE](LICENSE)
