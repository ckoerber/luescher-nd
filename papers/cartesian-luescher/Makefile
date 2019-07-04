TEX=pdflatex -halt-on-error
BIB=bibtex

REPO=git
OLD?=$(shell git rev-parse --short HEAD)
NEW?=--
ROOT:=$(shell pwd)

MASTER=master
TARGET?=$(MASTER)
SECTIONS = $(shell ls -1 section/ | sed -e 's/^/section\//g')

COUNTERTERMS = $(shell ls -1 section/counterterm/ | sed -e 's/^/section\/counterterm\//g')
SECTIONS+=$(COUNTERTERMS)

BIBS = $(find . -name '*.bib')


ifndef VERBOSE
	REDIRECT=1>/dev/null 2>/dev/null
endif

ifdef DRAFT
	OPTIONS?=$(shell ./repo/$(REPO).sh $(OLD) $(NEW))
endif

all: $(TARGET).pdf

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
	$(MAKE) tidy
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

.PHONY: clean
clean: tidy
	$(RM) $(TARGET).bbl
	$(RM) $(TARGET).pdf

.PHONY: watch
watch: $(TARGET).pdf
	watchman-make -p '**/*.tex' '*/*.tex' '*.tex' '*.bib' -t $(TARGET).pdf
