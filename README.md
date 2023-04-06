# GibbsToMaxwell
Framework for phase transitions between the Maxwell and Gibbs constructions

This python codes generates the equations of state and chemical composition of first order phase transition between Maxwell and gibbs construction, see https://arxiv.org/abs/2302.04289 for details.

The code is tested under python version 3.10. It's self-contained except for the standard python library: numpy, scipy and sympy.

``GibbsToMaxwell.ipynb'' generates the output date file and ``read_plot.ipynb'' reads the data and generates some plots.

``ZLA.txt'' is hadronic EOS data with 13 columns representing lepton fraction, (number desntiy, energy density, pressure) for (nuclear matter, electron and muon), chemical potential for (lepton, proton and neutron) and adiabatic sound speed.
\begin{verbatim}
# n_lep/nB\\
nB(fm-3) eps_N(MeVfm-3) p_N(MeVfm-3)\\
chempo_lep(MeV) chempo_p(MeV) chempo_n(MeV)\\
n_e(fm-3) eps_e(MeVfm-3) p_e(MeVfm-3)\\
n_mu(fm-3) eps_mu(MeVfm-3) p_mu(MeVfm-3) cs2ad
\end{verbatim}

``vMIT.txt'' is quark EOS data with 13 columns representing lepton fraction, (number desntiy, energy density, pressure) for (quark matter, electron and muon), chemical potential for (lepton, u quark and d or s quark) and adiabatic sound speed.
\begin{verbatim}
# n_lep/nB\\
nB(fm-3) eps_Q(MeVfm-3) p_Q(MeVfm-3)\\
chempo_lep(MeV) chempo_u(MeV) chempo_ds(MeV)\\
n_e(fm-3) eps_e(MeVfm-3) p_e(MeVfm-3)\\
n_mu(fm-3) eps_mu(MeVfm-3) p_mu(MeVfm-3) cs2ad
\end{verbatim}

``MIX\_axx\_Bxxx\_gx'' represent EOS data in mixed phase. The data have 22 columns. The first 12 columns representing ghost lepton ratio factor $g$, hadronic matter ratio $f$, total baryon number density, energy density and pressure; chemical potentials for proton, neutron, quarks, and leptons. The last 10 columns represent the density of all particles without factor in $f$ and $g$.
\begin{verbatim}
# g f(hadronic matter fraction)\\
nB(fm-3) density(MeVfm-3) pressure(MeVfm-3)\\
chempo_p(MeV) chempo_n(MeV)\\
chempo_u(MeV) chempo_d(MeV)\\
chempo_lep_N(MeV) chempo_lep_Q(MeV) chempo_lep_G(MeV)\\
p(fm-3) n(fm-3) u(fm-3) d+s(fm-3)\\
eN(fm-3) muN(fm-3) eQ(fm-3) muN(fm-3) eG(fm-3) muG(fm-3) 
\end{verbatim}
``g100'' is Maxwell which is trivial. ``g0'' is Gibbs and ``g10'' means $g=0.1$. ``a20'' means $a=0.2$fm$^2$, and ``B165'' means bag constant $B^{1/4}=165$ MeV. \emph{\bf In order to get the real densities of all particles, these ten columns need to multiply a factor $[f,f,1-f,1-f,fg,fg,(1-f)g,(1-f)g,1-g,1-g]$ for ten particle species respectively.}

The parameters to compute the EOS data table are listed below:
ZLA parameters:\\
a0,b0,gamma0=[-96.64,58.85,1.40]\\
a1,b1,gamma1=[-26.06,7.34,2.45]\\

vMIT parameters:\\
a=0.2 fm$^2$\\
B$^{1/4}$=165 MeV\\

Masses in MeV:\\
mp=mn=939.5\\
mu=5\\
md=7\\
ms=150\\
me=0.5109989499961642\\
mmu=105.65837549724458\\

Unit conversion:\\
fm-1=197.32698045930243 MeV\\

