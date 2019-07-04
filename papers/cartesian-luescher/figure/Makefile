
all:	I0.pdf

I0.pdf:  I0.tex
	latex I0.tex
	dvips I0.dvi
	ps2pdf I0.ps
	$(RM) I0.dvi I0.ps I0.aux I0.log
