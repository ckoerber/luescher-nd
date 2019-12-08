TEX=pdflatex -halt-on-error
BIB=bibtex

REPO=git
OLD?=$(shell git rev-parse --short HEAD)
NEW?=--
ROOT:=$(shell pwd)

MASTER=master
TARGET?=$(MASTER)
SECTIONS = $(shell find section -type f)

BIBS = $(find . -name '*.bib')

ifndef VERBOSE
	REDIRECT=1>/dev/null 2>/dev/null
endif

ifndef FINAL
	OPTIONS?=$(shell ./repo/$(REPO).sh $(OLD) $(NEW))
endif

all: $(TARGET).pdf

.PHONY: figures
figures:
	cd figure; $(MAKE)

ifndef DIFF
%.pdf: figures $(SECTIONS) $(BIBS) macros.tex %.tex
	DIFF=1 SHORTCIRCUIT=1 $(MAKE) $*.pdf
else
ifdef SHORTCIRCUIT
%.pdf: figures $(SECTIONS) $(BIBS) macros.tex %.tex
	@echo $@
	$(TEX) -jobname=$* "$(OPTIONS)\input{$*}" $(REDIRECT)
	-$(BIB) $* $(REDIRECT)
	$(TEX) -jobname=$* "$(OPTIONS)\input{$*}" $(REDIRECT)
	$(TEX) -jobname=$* "$(OPTIONS)\input{$*}" $(REDIRECT)
	# $(MAKE) tidy
else
%.pdf: figures $(SECTIONS) $(BIBS) macros.tex %.tex
	SHORTCIRCUIT=1 OPTIONS="$(OPTIONS)" git latexdiff --whole-tree --main $(TARGET).tex --prepare "rm -rf repo; ln -s $(ROOT)/repo" -o $(TARGET).pdf $(OLD) $(NEW)
endif
endif

.PHONY: git-hooks
git-hooks:
	for h in hooks/*; do ln -f -s "../../$$h" ".git/$$h"; done

.PHONY: remove-git-hooks
remove-git-hooks:
	for h in hooks/*; do rm ".git/$$h"; done

.PHONY: tidy
tidy:
	$(RM) section/*.aux
	$(RM) $(TARGET)Notes.bib
	$(RM) $(TARGET).{out,log,aux,synctex.gz,blg,toc,fls,fdb_latexmk}
	$(RM) arxiv.tgz

.PHONY: clean
clean: tidy
	$(RM) $(TARGET).bbl
	$(RM) $(TARGET).pdf
	cd figure; $(MAKE) clean

.PHONY: watch
watch: $(TARGET).pdf
	when-changed -s -1 -r . make

figure/figure-package.tar.gz:
	make -C figure package

arXiv.tgz: master.pdf figure/figure-package.tar.gz figure/bubbleSum.pdf
	mkdir -p arXiv/figure
	cp figure/figure-package.tar.gz arXiv
	tar -xf arXiv/figure-package.tar.gz --directory arXiv/figure
	cp figure/bubbleSum.pdf arxiv/figure # is not created during make
	$(RM) arXiv/figure-package.tar.gz
	cp  -r section arXiv
	cp  macros.tex arXiv
	cp *.tex arXiv
	cp  master.bbl arXiv
	tar -czf arXiv.tgz arXiv
	$(RM) -r arXiv
