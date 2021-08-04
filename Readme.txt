Steady Curved Duct Flow

Brendan Harding, 2019

This repository will be populated with different codes for computing the steady fluid flow through curved ducts.
Primarily these will be implemented in Python, although in some instances C++ implementations may also be provided.

Steady fluid flow is a topic that has been thoroughly studied since the work of W.R. Dean in the late 1920's and yet it is
somewhat difficult to find implementations of the many different computational methods that have been discussed in the literature. 
Partly this may be due to the ubiquity of CFD software that should, in theory, easily be able to approximate this.
Regardless, I think there is value in having a collection of robust and well tested methods that may be used as an alternative resource.
In this repository I intend to clean up and collect various codes I have written in relation to this problem, 
some of which are my own work, others are implementations of methods presented in the literature.
One particular goal is to collect a few different methodologies for approximating the flow through curved ducts having 
an unusual cross-section shape (by unusual I mean non-rectangular and non-circular).
One such method is a Rayleigh-Ritz method which I developed for small flow rates and is discussed in my paper:
B.Harding, A Rayleigh-Ritz method for Navier-Stokes flow through curved ducts, ANZIAMJ, 2019 (doi.org/10.1017/S1446181118000287) 
Many of my implementations for steady curved duct flow use the non-dimensionalisation described therein.

FD_2D:
This folder contains an implementation of a finite difference code (in Python) 
for curved ducts having a rectangular cross-section.
Solutions may be computed either with or without using the so-called Dean approximation.
This code features second order convergence (with respect to grid resolution). 
It is also able to produce solutions over a very large range of Dean numbers 
(possibly even any desired Dean number given sufficient resolution and patience).
The specific equations and non-dimensionalisation employed in the implementation is the same as
that described in my paper on the Rayleigh-Ritz method (doi.org/10.1017/S1446181118000287).
However, it is also possible to use in a manner close to the non-dimensionalisation used 
by Yamamoto et al. (see doi.org/10.1016/j.fluiddyn.2004.04.003).
A modification for solving flow through a two parameter family of trapezoidal cross-sections has
also been added (if you use this please get in touch as it is relatively new as of mid-2021).

FEM_2D:
At some stage (TODO) I will add my Python code for approximating the curved duct flow via 
the finite element method as implemented within the FEniCS framework.
This too is based on the equations and non-dimensionalisation described 
in my paper on the Rayleigh-Ritz method (doi.org/10.1017/S1446181118000287).

Spectral_2D:
At some stage (TODO) I will add various spectral and pseudo-spectral codes for 
approximating curved duct flow. The aim is to eventually include:
 - My Rayleigh-Ritz method which handles a large family of unusual cross-sections 
   (see doi.org/10.1017/S1446181118000287)
 - The implementation described by Yanase et al. for circular cross-sections 
   (see doi.org/10.1016/0169-5983(89)90021-X)
 - The implementation described by Yamamoto et al. for rectangular cross-sections 
   (see doi.org/10.1016/j.fluiddyn.2004.04.003)

Analytic_2D:
At some stage (TODO) I may add various analytic solutions from the literature
(these typically involve just a few terms from a suitable
perturbation expansion or simplification of the equations 
based on a large bend radius and/or slow flow rate).
These are sometimes useful for testing and verification purposes.



