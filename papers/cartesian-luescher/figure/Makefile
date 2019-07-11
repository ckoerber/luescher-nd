
FIGURES=I0.pdf	\
 		dispersion.pdf	\
		db-contact-fv-c-fitted-parity-lg.pdf 	\
		db-contact-fv-d-fitted-parity-lg.pdf	\
		db-contact-fv-c-fitted-a-inv-parity-lg.pdf	\
		db-contact-fv-d-fitted-a-inv-parity-lg.pdf	\
		continuum-ere-db-contact-fv-c-fitted-parity-lg.pdf

all:	$(FIGURES)

I0.pdf:  I0.tex
	latex I0.tex
	dvips I0.dvi
	ps2pdf I0.ps
	$(RM) I0.dvi I0.ps I0.aux I0.log

dispersion.pdf: dispersion.py
	./dispersion.py

ere: ere-plots.py
	./ere-plots.py

db-contact-fv-c-fitted-parity-lg.pdf: ere
db-contact-fv-d-fitted-parity-lg.pdf: ere
db-contact-fv-c-fitted-a-inv-parity-lg.pdf: ere
db-contact-fv-d-fitted-a-inv-parity-lg.pdf: ere

continuum-ere-db-contact-fv-c-fitted-parity-lg.pdf: continuum-extrapolation.py
	./continuum-extrapolation.py
