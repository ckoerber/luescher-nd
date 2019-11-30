.PHONY: docs
docs:
	make -C docs/ html

.PHONY: paper
paper:
	make -C papers/luescher-nd

.PHONY: install
install:
	pip install -e .
