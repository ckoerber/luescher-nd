
TARBALL=figure-package.tar.gz

FIGURES=I0.pdf	\
 		dispersion.pgf	\
		counterterm-nstep.pgf					\
		ere-contact-fitted_a-inv_+0.0_zeta_cartesian_projector_a1g_n-eigs_200.pgf	 \
		ere-contact-fitted_a-inv_+0.0_zeta_dispersion_projector_a1g_n-eigs_200.pgf	 \
		ere-contact-fitted_a-inv_+0.0_zeta_spherical_projector_a1g_n-eigs_200.pgf	 \
		ere-contact-fitted_a-inv_-5.0_zeta_cartesian_projector_a1g_n-eigs_200.pgf	 \
		ere-contact-fitted_a-inv_-5.0_zeta_dispersion_projector_a1g_n-eigs_200.pgf	 \
		ere-contact-fitted_a-inv_-5.0_zeta_spherical_projector_a1g_n-eigs_200.pgf	 \
		continuum-ere_a-inv_+0.0_zeta_spherical_projector_a1g_n-eigs_200.pgf	\
		continuum-spectrum_a-inv_+0.0_zeta_spherical_projector_a1g_n-eigs_200.pgf	\
		contact-scaling-contact-fitted_a-inv_+0.0_zeta_dispersion_projector_a1g_n-eigs_200.pgf	\
		a1g-state-overlap.pgf

all:	$(FIGURES)

I0.pdf:  I0.tex
	latex I0.tex 1>/dev/null 2>/dev/null
	dvips I0.dvi 1>/dev/null 2>/dev/null
	ps2pdf I0.ps 1>/dev/null 2>/dev/null
	$(RM) I0.dvi I0.ps I0.aux I0.log

dispersion.pgf: dispersion.py
	./dispersion.py

counterterm-nstep.pgf: counterterm-nstep.py
	./counterterm-nstep.py

ere: ere-plots.py
	./ere-plots.py
	touch ere

continuum: continuum-extrapolation.py
		./continuum-extrapolation.py
		touch continuum

a1g-state-overlap.pgf: a1g-state-overlap.ipynb
	jupyter nbconvert --to notebook --execute --output a1g-state-overlap-run.ipynb a1g-state-overlap.ipynb
	$(RM) a1g-state-overlap-run.ipynb

ere-contact-fitted_a-inv_+0.0_zeta_spherical_projector_a1g_n-eigs_200.pgf: ere
ere-contact-fitted_a-inv_+0.0_zeta_cartesian_projector_a1g_n-eigs_200.pgf: ere
ere-contact-fitted_a-inv_+0.0_zeta_dispersion_projector_a1g_n-eigs_200.pgf: ere
ere-contact-fitted_a-inv_-5.0_zeta_spherical_projector_a1g_n-eigs_200.pgf: ere
ere-contact-fitted_a-inv_-5.0_zeta_cartesian_projector_a1g_n-eigs_200.pgf: ere
ere-contact-fitted_a-inv_-5.0_zeta_dispersion_projector_a1g_n-eigs_200.pgf: ere

contact-scaling-contact-fitted_a-inv_+0.0_zeta_dispersion_projector_a1g_n-eigs_200.pgf: contact-scaling.py
	./contact-scaling.py

continuum-ere_a-inv_+0.0_zeta_spherical_projector_a1g_n-eigs_200.pgf: continuum
continuum-spectrum_a-inv_+0.0_zeta_spherical_projector_a1g_n-eigs_200.pgf: continuum

tidy:
	$(RM) I0.dvi I0.ps I0.aux I0.log
	$(RM) ere
	$(RM) continuum

clean: tidy
	$(RM) $(FIGURES)
	$(RM) $(TARBALL)

package: $(TARBALL)
$(TARBALL): all
	tar -cvzf $(TARBALL) $(FIGURES)
