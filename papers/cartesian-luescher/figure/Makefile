
TARBALL=figure-package.tar.gz

FIGURES=I0.pdf	\
 		dispersion.pdf	\
		counterterm-nstep.pdf					\
		ere-contact-fitted_a-inv_+0.0_zeta_cartesian_projector_a1g_n-eigs_200.pdf	 \
		ere-contact-fitted_a-inv_+0.0_zeta_dispersion_projector_a1g_n-eigs_200.pdf	 \
		ere-contact-fitted_a-inv_+0.0_zeta_spherical_projector_a1g_n-eigs_200.pdf	 \
		ere-contact-fitted_a-inv_-5.0_zeta_cartesian_projector_a1g_n-eigs_200.pdf	 \
		ere-contact-fitted_a-inv_-5.0_zeta_dispersion_projector_a1g_n-eigs_200.pdf	 \
		ere-contact-fitted_a-inv_-5.0_zeta_spherical_projector_a1g_n-eigs_200.pdf	 \
		continuum-ere-contact-fitted_a-inv_+0.0_zeta_spherical_projector_a1g_n-eigs_200.pdf
all:	$(FIGURES)

I0.pdf:  I0.tex
	latex I0.tex 1>/dev/null 2>/dev/null
	dvips I0.dvi 1>/dev/null 2>/dev/null
	ps2pdf I0.ps 1>/dev/null 2>/dev/null
	$(RM) I0.dvi I0.ps I0.aux I0.log

dispersion.pdf: dispersion.py
	./dispersion.py

counterterm-nstep.pdf: counterterm-nstep.py
	./counterterm-nstep.py

ere: ere-plots.py
	./ere-plots.py
	touch ere

ere-contact-fitted_a-inv_+0.0_zeta_cartesian_projector_a1g_n-eigs_200.pdf: ere
ere-contact-fitted_a-inv_+0.0_zeta_cartesian_projector_a1g_n-eigs_200.pdf: ere
ere-contact-fitted_a-inv_+0.0_zeta_dispersion_projector_a1g_n-eigs_200.pdf: ere
ere-contact-fitted_a-inv_-5.0_zeta_spherical_projector_a1g_n-eigs_200.pdf: ere
ere-contact-fitted_a-inv_-5.0_zeta_cartesian_projector_a1g_n-eigs_200.pdf: ere
ere-contact-fitted_a-inv_-5.0_zeta_dispersion_projector_a1g_n-eigs_200.pdf: ere

continuum-ere-contact-fitted_a-inv_+0.0_zeta_spherical_projector_a1g_n-eigs_200.pdf: continuum-extrapolation.py
	./continuum-extrapolation.py

tidy:
	$(RM) I0.dvi I0.ps I0.aux I0.log
	$(RM) ere

clean: tidy
	$(RM) $(FIGURES)
	$(RM) $(TARBALL)

package: $(TARBALL)
$(TARBALL): all
	tar -cvzf $(TARBALL) $(FIGURES)
