.PHONY: docs
docs:
	make -C docs/ html

.PHONY: paper
paper:
	make -C paper/luescher-nd

.PHONY: arXiv
arXiv:
	make -C paper/luescher-nd arXiv.tgz

.PHONY: install
install:
	pip install -r requirements.txt
	pip install -e .

.PHONY: test
test:
	pytest .

.PHONY: clean
clean:
	make -C docs clean
	make -C paper/luescher-nd clean
