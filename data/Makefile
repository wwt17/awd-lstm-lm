all: wikitext-2 wikitext-103 enwik8 penn pennchar

wikitext-2-v1.zip:
	echo "- Downloading WikiText-2 (WT2)"
	wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip

wikitext-2: wikitext-2-v1.zip
	unzip -q wikitext-2-v1.zip && \
	cd wikitext-2; \
	mv wiki.train.tokens train.txt; \
	mv wiki.valid.tokens valid.txt; \
	mv wiki.test.tokens test.txt

wikitext-103-v1.zip:
	echo "- Downloading WikiText-103 (WT2)"
	wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip

wikitext-103: wikitext-103-v1.zip
	unzip -q wikitext-103-v1.zip && \
	cd wikitext-103; \
	mv wiki.train.tokens train.txt; \
	mv wiki.valid.tokens valid.txt; \
	mv wiki.test.tokens test.txt

enwik8/enwik8.zip:
	echo "- Downloading enwik8 (Character)"
	mkdir -p enwik8 && \
	cd enwik8; \
	wget --continue http://mattmahoney.net/dc/enwik8.zip

enwik8: enwik8/enwik8.zip
	cd enwik8; \
	python prep_enwik8.py

simple-examples.tgz:
	echo "- Downloading Penn Treebank (PTB)"
	wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

simple-examples:
	tar -xzf simple-examples.tgz

penn: simple-examples
	mkdir -p penn && \
	cd penn; \
	mv ../simple-examples/data/ptb.train.txt train.txt; \
	mv ../simple-examples/data/ptb.test.txt test.txt; \
	mv ../simple-examples/data/ptb.valid.txt valid.txt

pennchar: simple-examples
	mkdir -p pennchar && \
	cd pennchar; \
	mv ../simple-examples/data/ptb.char.train.txt train.txt; \
	mv ../simple-examples/data/ptb.char.test.txt test.txt; \
	mv ../simple-examples/data/ptb.char.valid.txt valid.txt

clean:
	rm -rf simple-examples/
