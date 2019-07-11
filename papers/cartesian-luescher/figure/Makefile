
FIGURES=I0.pdf	\
 		dispersion.pdf	\
		db-contact-fv-c-fitted.pdf 	\
		db-contact-fv-d-fitted.pdf

all:	$(FIGURES)

I0.pdf:  I0.tex
	latex I0.tex
	dvips I0.dvi
	ps2pdf I0.ps
	$(RM) I0.dvi I0.ps I0.aux I0.log

dispersion.pdf: dispersion.py
	./dispersion.py

db-contact-fv-c-fitted.pdf: ere-plots.py
	./ere-plots.py

db-contact-fv-d-fitted.pdf: ere-plots.py
	./ere-plots.py
