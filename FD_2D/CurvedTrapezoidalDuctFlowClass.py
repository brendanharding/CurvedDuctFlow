import numpy as np
from scipy.sparse import diags,bmat
from scipy.sparse.linalg import spsolve

class CurvedTrapezoidalDuctFlowClass(object):
    """This class provides a finite difference implementation for solving
    steady flow through a curved duct having a trapezoidal cross-section.
    Solutions may be computed both with and without using the Dean
    approximation (solutions in this are often referred to as Dean flow).
    
    The specific equations and non-dimensionalisation used within this class
    are described in my publication here doi.org/10.1017/S1446181118000287
    (noting, however, that this paper describes an alternative numerical method
    to solve these equations for a larger variety of cross-section shapes).
    Specifically, refer to the non-linear PDE given in equations (2.3).
    This class, in fact, solves an appropriate transformation of these equations,
    i.e. one which maps rectangular coordinates to certain trapezoidal cross-sections.
    
    The solver uses Newton's method to solve the fully coupled non-linear problem.
    The linear system for each iteration is contructed in sparse matrix format
    and solved using sparse LU by default (although using bicgstab is an option).
    The discretisation is such that the solutions achieve second order
    convergence with respect to grid resolution.
    """
    def __init__(self,m=65,n=65,W=2.0,H=2.0,epsilon=0.01,K=1.0,G=4.0,\
        alpha=0.0,beta=0.0,delta=None,symmetric=True):
        """
        Initialise the class with the desired parameters.
        Some parameters can be changed later, excepting W,H (I would
        recommend using a separate class instance for different shapes).
        Similarly, changing m,n is loosely supported, but using different
        instances would probably better.
        
        Parameters
        ----------
        m,n : int,int
            The number of points in the finite difference discretisation,
            m being the number of points across the width, and
            n being the number of points across the height.
            Note that array shape/dimensions will be ordered as (n,m).
            Defaults to 65,65
        W,H : float
            The width and height of the rectangular cross-section
            (in dimensionless coordinates).
            To be consistent with the non-dimensionalisation in the referenced
            paper these should generally be scaled such that min(W,H)=2.
            Defaults to 2.0,2.0
        epsilon: float
            The curvature of the duct, that is 1/R given a bend radius R
            in dimensionless coordinates.
            Providing a value of 0 means the solution corresponds to 'Dean flow'.
            Defaults to 0.0
        K : float
            The Dean number used in the equations (specifically K=epsilon*Re^2).
            Observe this can be specified as non-zero even if epsilon=0.
            Defaults to 1.0
        G : float
            The pressure gradient driving the axial flow.
            Ideally it should be chosen such that max(u)=1.0 given K=0.0
            (e.g. 4.0/L^2 would be the appropriate choice for a circular
             pipe and is generally close enough for our purposes).
            You should only change this if you understand the consequences on
            the scaling of both the axial velocity and stream-function.
            Defaults to 4.0
        alpha,beta: float
            Coefficients which describe the cross-section shape.
            Specifically, the rectangular coordinates
            (r,z) in [-W/2,W/2]x[-H/2,H/2]
            are mapped to
            (r,z+alpha*r+beta*r*z)
            which can describe a wide range of trapezoidal/rhombus cross-sections
            having vertical side walls.
            If delta!=None any values passed here are ignored.
        delta: float
            Provides a convenient way to specify trapezoidal cross-sections.
            The delta parameter describes a cross-section whose height is
            H*(1+delta*r/(W/2)) for each r across its width.
            If delta!=None is specified, then the alpha,beta parameters
            are ignored and the symmetric parameter is checked.
        symmetric: bool
            Only checked when delta!=None is specified.
            If True then alpha=0 is set so that the cross-section is
            vertically symmetric, if False then alpha is set such that
            the bottom wall is flat (i.e. perpendicular to the sides).
        """
        self._shape = (n,m)
        self._size  = (H,W)
        self._epsilon = epsilon
        self._K = K
        self._G = G
        #
        self._alpha = alpha
        self._beta  = beta
        if delta is not None:
            self._beta  = delta*2/W
            if symmetric:
                # z -> z*(1+delta*r/(W/2))
                self._alpha = 0.0
            else:
                # z -> (z+H/2)*(1+delta*r/(W/2))-H/2=z+r*z*delta/(W/2)+r*delta*H/W
                self._alpha = delta*H/W
        # Define other useful numbers
        self._L = min(0.5*W,0.5*H) # probably not really needed...
        self._N = n*m              # probably not really needed...
        self._dr = W/(m-1)
        self._dz = H/(n-1)
        # Construct the arrays describing the rectangular coordinate system
        self._r = np.linspace(-0.5*W,0.5*W,m)
        self._z = np.linspace(-0.5*H,0.5*H,n)
        self._R,self._Z = np.meshgrid(self._r,self._z) # Note: use default index order
        # Construct array describing the vertical trapezoidal coordinate (R is same)
        self._Zt = self._Z+self._alpha*self._R+self._beta*self._R*self._Z
        # Define some useful meshgrids depending on physical dimensions
        if self._epsilon==0.0:
            self._Tau = 1.0
        else:
            self._Tau = 1.0+self._epsilon*self._R
        if self._beta==0.0:
            self._Sigma = 1.0
            self._Omega = self._alpha
        else:
            self._Sigma = 1.0+self._beta*self._R
            self._Omega = self._alpha+self._beta*self._Z
        # Pre-allocate array for holding an initial guess (2*8*m*n bytes)
        self._u_old = np.zeros((n,m))
        self._Phi_old = np.zeros((n,m))
        self._u_old_is_zero = True # tracks whether an initial u iteration should be performed
        # Pre-allocate arrays of intermediate results
        self._dudr = np.zeros((n,m))
        self._dudz = np.zeros((n,m))
        self._d2udr2 = np.zeros((n,m))
        self._d2udrdz = np.zeros((n,m))
        self._d2udz2 = np.zeros((n,m))
        self._dPhidr = np.zeros((n,m))
        self._dPhidz = np.zeros((n,m))
        self._d2Phidr2 = np.zeros((n,m))
        self._d2Phidrdz = np.zeros((n,m))
        self._d2Phidz2 = np.zeros((n,m))
        self._d3Phidr3 = np.zeros((n,m))
        self._d3Phidr2dz = np.zeros((n,m))
        self._d3Phidrdz2 = np.zeros((n,m))
        self._d3Phidz3 = np.zeros((n,m))
        self._d4Phidr4 = np.zeros((n,m))
        self._d4Phidr3dz = np.zeros((n,m))
        self._d4Phidr2dz2 = np.zeros((n,m))
        self._d4Phidrdz3 = np.zeros((n,m))
        self._d4Phidz4 = np.zeros((n,m))
        # The following are 'advanced options' I don't expect a typical user to tweak.
        # They first two are related to the extrapolation of ghost cells used to
        # accurately enforce the Neumann boundary conditions.
        # The last relates to the construction of an initial guess if none is provided.
        self._use_first_order_extrap = False # toggle 1st order extrapolation of 2nd derivatives
        self._add_third_deriv_extrap = True  # toggle use of 3rd derivatives in extrapolation
        self._use_initial_u_iterate  = True  # toggle use of initial u iterate (when u_old=0)
        # Done
    def __del__(self):
        """Destructor, deletes all internally stored arrays
        (noting memory may not be freed immediately: google 'python garbage collection')"""
        del self._r
        del self._z
        del self._R
        del self._Z
        del self._Zt
        del self._Tau
        del self._Sigma
        del self._Omega
        del self._u_old
        del self._Phi_old
        del self._dudr
        del self._dudz
        del self._d2udr2
        del self._d2udrdz
        del self._d2udz2
        del self._dPhidr
        del self._dPhidz
        del self._d2Phidr2
        del self._d2Phidrdz
        del self._d2Phidz2
        del self._d3Phidr3
        del self._d3Phidr2dz
        del self._d3Phidrdz2
        del self._d3Phidz3
        del self._d4Phidr4
        del self._d4Phidr3dz
        del self._d4Phidr2dz2
        del self._d4Phidrdz3
        del self._d4Phidz4
        # Done
    def get_meshgrid(self,rect=False):
        """Returns the meshgrid of (r,z) coordinates for the finite
        difference points (in the rectangular coordinates if rect=True,
        otherwise in the trapezoidal coordinates)."""
        if rect:
            return self._R,self._Z
        else:
            return self._R,self._Zt
    def get_secondary_velocities(self,Phi=None,rect=False):
        """Returns the velocity fields associated with a stream-function
        of the secondary flow.
        Note the velocities will be given as exactly zero on the boundary.
        If rect False the horizontal and vertical components will be returned
        with respect to the trapezoidal coordinates, otherwise the transformed
        (rectangular) components are returned."""
        if Phi is None:
            Phi = self._Phi_old
        else:
            assert Phi.shape==self._shape
        if self._epsilon==0.0:
            Tau = 1.0
        else:
            Tau = self._Tau[1:-1,1:-1] # restrict to interior
        if isinstance(self._Sigma,float):
            Sigma = self._Sigma
        else:
            Sigma = self._Sigma[1:-1,1:-1] # restrict to interior
        V = self._zeros_edge()
        W = self._zeros_edge()
        V[1:-1,1:-1] = -self._FD_dz(Phi)/(Tau*Sigma)
        W[1:-1,1:-1] =  self._FD_dr(Phi)/(Tau*Sigma)
        if rect:
            return V,W
        else:
            return V,self._Omega*V+self._Sigma*W
    def set_initial_guess(self,u=None,Phi=None):
        """Provide an initial guess for the solver.
        For example, this may be a previously computed solution with
        similar paramters, e.g. perhaps a nearby Dean number.
        With no arguments it resets initial guesses to zero.
        Note that generally the class will re-use the last computed solution
        as an initial guess unless one is provided/specified.
        """
        if u is None:
            self._u_old.fill(0.0)
            self._u_old_is_zero = True
        else:
            assert u.shape==self._shape
            self._u_old[:,:] = u[:,:]  # copy into pre-allocated arrays
            self._u_old_is_zero = False
        if Phi is None:
            self._Phi_old.fill(0.0)
        else:
            assert Phi.shape==self._shape
            self._Phi_old[:,:] = Phi[:,:]
        return
    def set_K(self,K):
        """Set a new 'Dean' number K
        (Recalling this specific non-dimensionalisation is such that
         K is proportional to epsilon*Re^2)"""
        self._K = K
    def get_K(self):
        """Get the current 'Dean' number K
        (Recalling this specific non-dimensionalisation is such that
         K is proportional to epsilon*Re^2)"""
        return self._K
    def set_G(self,G):
        """Set a (dimensionless) driving pressure G
        (Recall this specific non-dimensionalisation is such that
         G is proportional to Dn=sqrt(K))"""
        self._G = G
    def get_G(self):
        """Get the current (dimensionless) driving pressure G
        (Recall this specific non-dimensionalisation is such that
         G is proportional to Dn=sqrt(K))"""
        return self._G
    def set_epsilon(self,epsilon):
        """Set a new curvature ratio epsilon=X/R
        (R being the bend radius and X the characteristic duct size)
        Recall setting this to zero results in the 'Dean approximation'"""
        self._epsilon = epsilon
        self._Tau = 1.0+self._epsilon*self._R
    def get_epsilon(self):
        """Get the current curvature ratio epsilon=X/R
        (R being the bend radius and X the characteristic duct size)
        Recall setting this to zero results in the 'Dean approximation'"""
        return self._epsilon
    def set_delta(self,delta,symmetric=True):
        """Change the delta parameter describing the trapezoid."""
        self._beta  = delta*2/W
        if symmetric:
            self._alpha = 0.0
        else:
            self._alpha = delta*H/W
        # Need to update the following
        self._Zt = self._Z+alpha*self._R+beta*self._R*self._Z
        if self._beta==0.0:
            self._Sigma = 1.0
            self._Omega = self._alpha
        else:
            self._Sigma = 1.0+self._beta*self._R
            self._Omega = self._alpha+self._beta*self._Z
    def set_alpha_beta(self,alpha,beta):
        """Change the alpha,beta parameters describing the cross-section."""
        self._alpha = alpha
        self._beta  = beta
        # Need to update the following
        self._Zt = self._Z+alpha*self._R+beta*self._R*self._Z
        if self._beta==0.0:
            self._Sigma = 1.0
            self._Omega = self._alpha
        else:
            self._Sigma = 1.0+self._beta*self._R
            self._Omega = self._alpha+self._beta*self._Z
    def get_alpha_beta(self):
        """Fetch the alpha,beta parameters describing the cross-section."""
        return self._alpha,self._beta
    def _zeros_edge(self):
        """Initialise an array of the correct size with zeros on the edge/boundary"""
        # Note: for small array sizes (essentially fitting in cache)
        # using np.zeros is quicker, however the folllowing scales better
        a=np.empty(self._shape);a[[0,-1],:]=0;a[1:-1,[0,-1]]=0;
        return a
    def _ones_edge(self):
        """Initialise an array of the correct size with ones on the edge/boundary"""
        # Note: for small array sizes (essentially fitting in cache)
        # using np.ones is quicker, however the folllowing scales better
        a=np.empty(self._shape);a[[0,-1],:]=1;a[1:-1,[0,-1]]=1;
        return a
    def _initial_u_iterate(self):
        """Construct and solve a Poisson like problem to create an initial
        guess for the axial velocity u (which is stored in u_old)."""
        # We specifically solve the u equation in the case eps=K=0
        # (although potentially using Tau!=1, so not entirely consistent...)
        # TODO: possibly consider including eps!=0 terms here
        shape = self._shape
        m = shape[1]
        dr = self._dr
        dz = self._dz
        eps = self._epsilon
        alpha = self._alpha
        beta = self._beta
        if eps==0.0:
            Tau = 1.0
        else:
            Tau = self._Tau[1:-1,1:-1] # restrict to interior
        if isinstance(self._Sigma,float):
            Sigma = self._Sigma
        else:
            Sigma = self._Sigma[1:-1,1:-1] # restrict to interior
        if isinstance(self._Omega,float):
            Omega = self._Omega
        else:
            Omega = self._Omega[1:-1,1:-1] # restrict to interior
        # Initialise diagonals
        A_p0_p0 = self._ones_edge() # set to 1 to enforce Dirichlet BC
        A_p1_p0 = self._zeros_edge()
        A_m1_p0 = self._zeros_edge()
        A_p0_p1 = self._zeros_edge()
        A_p0_m1 = self._zeros_edge()
        if alpha!=0.0 or beta!=0.0:
            A_p1_p1 = self._zeros_edge()
            A_p1_m1 = self._zeros_edge()
            A_m1_p1 = self._zeros_edge()
            A_m1_m1 = self._zeros_edge()
        # Fill in each entry (note: epsilon=0 terms are stil ignored here, but not Omega terms
        A_p0_p0[1:-1,1:-1] = -2.0/dr**2-2.0*(1+Omega**2)/Sigma**2/dz**2
        A_p1_p0[1:-1,1:-1] = +1.0/dr**2
        A_m1_p0[1:-1,1:-1] = +1.0/dr**2
        A_p0_p1[1:-1,1:-1] = +1.0*(1+Omega**2)/Sigma**2/dz**2+2*beta*Omega/Sigma**2/(2*dz)
        A_p0_m1[1:-1,1:-1] = +1.0*(1+Omega**2)/Sigma**2/dz**2-2*beta*Omega/Sigma**2/(2*dz)
        if alpha!=0.0 or beta!=0.0:
            A_p1_p1[1:-1,1:-1] = -2.0*Omega/(4*dr*dz*Sigma)
            A_m1_p1[1:-1,1:-1] = +2.0*Omega/(4*dr*dz*Sigma)
            A_p1_m1[1:-1,1:-1] = +2.0*Omega/(4*dr*dz*Sigma)
            A_m1_m1[1:-1,1:-1] = -2.0*Omega/(4*dr*dz*Sigma)
            A = diags([A_p0_p0.ravel(),\
                       A_p1_p0.ravel()[:-1]  ,A_m1_p0.ravel()[1:],\
                       A_p0_p1.ravel()[:-m]  ,A_p0_m1.ravel()[m:],\
                       A_p1_p1.ravel()[:-m-1],A_m1_p1.ravel()[:-m+1],\
                       A_p1_m1.ravel()[m-1:] ,A_m1_m1.ravel()[1+m:]],\
                      [0,1,-1,m,-m,m+1,m-1,-m+1,-m-1],format='csr')
        else:
            A = diags([A_p0_p0.ravel(),\
                       A_p1_p0.ravel()[:-1],A_m1_p0.ravel()[1:],\
                       A_p0_p1.ravel()[:-m],A_p0_m1.ravel()[m:]],\
                      [0,1,-1,m,-m],format='csr')
        # Construct the RHS
        b = self._zeros_edge()
        b[1:-1,1:-1] = -self._G/Tau
        # It is difficult to predict how much memory spsolve will use
        self._u_old = spsolve(A,b.ravel()).reshape(shape)
        self._u_old_is_zero = False
        return
    def _FD_dr(self,array):
        """Return centred finite difference estimate of the
        r (or x) derivative on the interior of the given array"""
        return (array[1:-1,2:]-array[1:-1,:-2])/(2.0*self._dr)
    def _FD_dz(self,array):
        """Return centred finite difference estimate of the
        z (or y) derivative on the interior of the given array"""
        return (array[2:,1:-1]-array[:-2,1:-1])/(2.0*self._dz)
    def _FD_dr2(self,array):
        """Return centred finite difference estimate of the second
        r (or x) derivative on the interior of the given array"""
        return (array[1:-1,2:]-2.0*array[1:-1,1:-1]+array[1:-1,:-2])/self._dr**2
    def _FD_drdz(self,array):
        """Return centred finite difference estimate of the mixed second
        r,z (or x,y) derivative on the interior of the given array"""
        return (array[2:,2:]-array[2:,:-2]-array[:-2,2:]+array[:-2,:-2])/(4.0*self._dr*self._dz)
    def _FD_dz2(self,array):
        """Return centred finite difference estimate of the second
        z (or y) derivative on the interior of the given array"""
        return (array[2:,1:-1]-2.0*array[1:-1,1:-1]+array[:-2,1:-1])/self._dz**2
    #def _FD_Laplace(self,array):
    #    """Return centred finite difference estimate of the
    #    Laplacian on the interior of the given array"""
    #    return self._FD_dr2(array)+self._FD_dz2(array)
    def _create_ghost_array(self,array):
        """Extrapolate a given array to provide one layer of ghost points
        around the exterior.
        The specific extrapolation here is such that the Neumann boundary
        conditions for Phi are satisfied (noting that the linear system
        construction must also be consistent with this).
        As such, the given array will generally be self._Phi_old"""
        shape = array.shape
        ghost = np.empty((shape[0]+2,shape[1]+2))
        ghost[1:-1,1:-1] = array
        if self._use_first_order_extrap:
            ghost[ 0,1:-1] = 1.5*ghost[ 1,1:-1]-1.0*ghost[ 2,1:-1]+0.5*ghost[ 3,1:-1]
            ghost[-1,1:-1] = 1.5*ghost[-2,1:-1]-1.0*ghost[-3,1:-1]+0.5*ghost[-4,1:-1]
            ghost[1:-1, 0] = 1.5*ghost[1:-1, 1]-1.0*ghost[1:-1, 2]+0.5*ghost[1:-1, 3]
            ghost[1:-1,-1] = 1.5*ghost[1:-1,-2]-1.0*ghost[1:-1,-3]+0.5*ghost[1:-1,-4]
        else:
            ghost[ 0,1:-1] =  2.0*ghost[ 1,1:-1]-2.5*ghost[ 2,1:-1] \
                             +2.0*ghost[ 3,1:-1]-0.5*ghost[ 4,1:-1]
            ghost[-1,1:-1] =  2.0*ghost[-2,1:-1]-2.5*ghost[-3,1:-1] \
                             +2.0*ghost[-4,1:-1]-0.5*ghost[-5,1:-1]
            ghost[1:-1, 0] =  2.0*ghost[1:-1, 1]-2.5*ghost[1:-1, 2] \
                             +2.0*ghost[1:-1, 3]-0.5*ghost[1:-1, 4]
            ghost[1:-1,-1] =  2.0*ghost[1:-1,-2]-2.5*ghost[1:-1,-3] \
                             +2.0*ghost[1:-1,-4]-0.5*ghost[1:-1,-5]
        if self._add_third_deriv_extrap:
            ghost[ 0,1:-1] -= (-ghost[ 1,1:-1]+3.0*ghost[ 2,1:-1] \
                               -3.0*ghost[ 3,1:-1]+ghost[ 4,1:-1])/6.0
            ghost[-1,1:-1] += ( ghost[-2,1:-1]-3.0*ghost[-3,1:-1] \
                               +3.0*ghost[-4,1:-1]-ghost[-5,1:-1])/6.0
            ghost[1:-1, 0] -= (-ghost[1:-1, 1]+3.0*ghost[1:-1, 2] \
                               -3.0*ghost[1:-1, 3]+ghost[1:-1, 4])/6.0
            ghost[1:-1,-1] += ( ghost[1:-1,-2]-3.0*ghost[1:-1,-3] \
                               +3.0*ghost[1:-1,-4]-ghost[1:-1,-5])/6.0
        # Note that extrapolation of corner points is not needed
        return ghost
    def _update_intermediates(self,u,Phi):
        """Update intermediate arrays containing finite difference
        estimates of the derivatives of solution vectors."""
        # TODO: can make more efficient when one or more of epsilon,alpha,beta are zero by not updating those (if any) which are not used.
        self._dudr[1:-1,1:-1] = self._FD_dr(u)
        self._dudz[1:-1,1:-1] = self._FD_dz(u)
        self._d2udr2[1:-1,1:-1]  = self._FD_dr2(u)
        self._d2udrdz[1:-1,1:-1] = self._FD_drdz(u)
        self._d2udz2[1:-1,1:-1]  = self._FD_dz2(u)
        self._dPhidr[1:-1,1:-1] = self._FD_dr(Phi)
        self._dPhidz[1:-1,1:-1] = self._FD_dz(Phi)
        Phi_ghost = self._create_ghost_array(Phi) # only used for intermediate calculation
        self._d2Phidr2[:,:]  = self._FD_dr2(Phi_ghost)
        self._d2Phidrdz[:,:] = self._FD_drdz(Phi_ghost)
        self._d2Phidz2[:,:]  = self._FD_dz2(Phi_ghost)
        self._d3Phidr3[1:-1,1:-1]   = self._FD_dr(self._d2Phidr2)
        self._d3Phidr2dz[1:-1,1:-1] = self._FD_dz(self._d2Phidr2) # or dr of drdz?
        self._d3Phidrdz2[1:-1,1:-1] = self._FD_dr(self._d2Phidz2) # or dz of drdz?
        self._d3Phidz3[1:-1,1:-1]   = self._FD_dz(self._d2Phidz2)
        self._d4Phidr4[1:-1,1:-1]    = self._FD_dr2(self._d2Phidr2)
        self._d4Phidr3dz[1:-1,1:-1]  = self._FD_drdz(self._d2Phidr2) # or dr2 of drdz?
        #self._d4Phidr2dz2[1:-1,1:-1] = self._FD_drdz(self._d2Phidrdz) # don't use this! (stencil too big)
        self._d4Phidr2dz2[1:-1,1:-1] = self._FD_dz2(self._d2Phidr2) # or dr2 of dz2
        self._d4Phidrdz3[1:-1,1:-1]  = self._FD_drdz(self._d2Phidz2) # or dz2 of drdz
        self._d4Phidz4[1:-1,1:-1]    = self._FD_dz2(self._d2Phidz2)
        return
    def _generate_solver_residual(self,u=None,Phi=None,K=None,G=None):
        """Generates the residual vector for the Newton iteration used by the solver.
        (Note this differs a little from the form written in my ANZIAM paper
         by factors of R=1+epsilon*S)"""
        if u is None:
            u = self._u_old
        else:
            assert u.shape==self._shape
        if Phi is None:
            Phi = self._Phi_old
        else:
            assert Phi.shape==self._shape
        if K is None:
            K = self._K
        if G is None:
            G = self._G
            
        shape = self._shape
        eps = self._epsilon
        alpha = self._alpha
        beta = self._beta
        if eps==0.0:
            Tau = 1.0
        else:
            Tau = self._Tau[1:-1,1:-1] # restrict to interior
        if isinstance(self._Sigma,float):
            Sigma = self._Sigma
        else:
            Sigma = self._Sigma[1:-1,1:-1] # restrict to interior
        if isinstance(self._Omega,float):
            Omega = self._Omega
        else:
            Omega = self._Omega[1:-1,1:-1] # restrict to interior
            
        self._update_intermediates(u,Phi)
        dudr_int = self._dudr[1:-1,1:-1]
        dudz_int = self._dudz[1:-1,1:-1]
        d2udr2_int  = self._d2udr2[1:-1,1:-1]
        d2udrdz_int = self._d2udrdz[1:-1,1:-1]
        d2udz2_int  = self._d2udz2[1:-1,1:-1]
        dPhidr_int = self._dPhidr[1:-1,1:-1]
        dPhidz_int = self._dPhidz[1:-1,1:-1]
        d2Phidr2_int  = self._d2Phidr2[1:-1,1:-1]
        d2Phidrdz_int = self._d2Phidrdz[1:-1,1:-1]
        d2Phidz2_int  = self._d2Phidz2[1:-1,1:-1]
        d3Phidr3_int   = self._d3Phidr3[1:-1,1:-1]
        d3Phidr2dz_int = self._d3Phidr2dz[1:-1,1:-1]
        d3Phidrdz2_int = self._d3Phidrdz2[1:-1,1:-1]
        d3Phidz3_int   = self._d3Phidz3[1:-1,1:-1]
        d4Phidr4_int    = self._d4Phidr4[1:-1,1:-1]
        d4Phidr3dz_int  = self._d4Phidr3dz[1:-1,1:-1]
        d4Phidr2dz2_int = self._d4Phidr2dz2[1:-1,1:-1]
        d4Phidrdz3_int  = self._d4Phidrdz3[1:-1,1:-1]
        d4Phidz4_int    = self._d4Phidz4[1:-1,1:-1]
        
        b = np.empty((2,)+shape)
        # Fill 0's on the boundary (assuming u,Phi are zero on boundary)
        b[:,[0,-1],:] = 0
        b[:,1:-1,[0,-1]] = 0
        # Residual for interior of uhat equation first (epsilon and K free components)
        b[0,1:-1,1:-1] = -G/Tau\
                         -d2udr2_int\
                         -(1+Omega**2)/Sigma**2*d2udz2_int\
                         +2*Omega/Sigma*d2udrdz_int\
                         -2*beta*Omega/Sigma**2*dudz_int
        # Residual for interior of uhat equation first (epsilon free components)
        b[1,1:-1,1:-1] = -(1+Omega**2)**2*d4Phidz4_int\
                         +4*(1+Omega**2)*Omega*Sigma*d4Phidrdz3_int\
                         -(2+6*Omega**2)*Sigma**2*d4Phidr2dz2_int\
                         +4*Omega*Sigma**3*d4Phidr3dz_int\
                         -Sigma**4*d4Phidr4_int\
                         -12*(1+Omega**2)*Omega*beta*d3Phidz3_int\
                         +(8+24*Omega**2)*Sigma*beta*d3Phidrdz2_int\
                         -12*Omega*Sigma**2*beta*d3Phidr2dz_int\
                         \
                         -(12+36*Omega**2)*beta**2*d2Phidz2_int\
                         +24*Omega*Sigma*beta**2*d2Phidrdz_int\
                         \
                         -24*Omega*beta**3*dPhidz_int\
                         \
                         +2*Sigma**3*u[1:-1,1:-1]*dudz_int
        # Add the K!=0 (but eps=0) components for each
        if K!=0.0:
            b[0,1:-1,1:-1] += +K/(Tau*Sigma)*(dPhidr_int*dudz_int-dPhidz_int*dudr_int)
            b[1,1:-1,1:-1] += -K/Tau*(+(1+Omega**2)*Sigma*(d3Phidrdz2_int*dPhidz_int-d3Phidz3_int*dPhidr_int)\
                                      +2*Omega*Sigma**2*(dPhidr_int*d3Phidrdz2_int-dPhidz_int*d3Phidr2dz_int)\
                                      +Sigma**3*(d3Phidr3_int*dPhidz_int-d3Phidr2dz_int*dPhidr_int)\
                                      -2*(1+Omega**2)*beta*dPhidz_int*d2Phidz2_int\
                                      -4*Omega*Sigma*beta*(dPhidr_int*d2Phidz2_int-dPhidz_int*d2Phidrdz_int)\
                                      +2*Sigma**2*beta*dPhidr_int*d2Phidrdz_int\
                                      -2*Sigma*beta**2*dPhidr_int*dPhidz_int\
                                      -4*Omega*beta**2*dPhidz_int**2)
        # Now add the epsilon!=0 components to each (incl. any K!=0 components)
        if eps!=0.0:
            b[0,1:-1,1:-1] += (eps/Tau)*(+Omega/Sigma*dudz_int\
                                         -dudr_int\
                                         +eps/Tau*u[1:-1,1:-1]\
                                         -K/(Tau*Sigma)*u[1:-1,1:-1]*dPhidz_int)
            b[1,1:-1,1:-1] += (eps/Tau)*(-2*(1+Omega**2)*Omega*Sigma*d3Phidz3_int\
                                         +(2+6*Omega**2)*Sigma**2*d3Phidrdz2_int\
                                         -6*Omega*Sigma**3*d3Phidr2dz_int\
                                         +2*Sigma**4*d3Phidr3_int\
                                         -(4+12*Omega**2)*Sigma*beta*d2Phidz2_int\
                                         +12*Omega*Sigma**2*beta*d2Phidrdz_int\
                                         \
                                         -12*Omega*Sigma*beta**2*dPhidz_int\
                                         )\
                              +(eps/Tau)**2*(-3*Omega**2*Sigma**2*d2Phidz2_int\
                                             +6*Omega*Sigma**3*d2Phidrdz_int\
                                             -3*Sigma**4*d2Phidr2_int\
                                             -6*Omega*Sigma**2*beta*dPhidz_int\
                                             )\
                              +(eps/Tau)**3*(-3*Omega*Sigma**3*dPhidz_int\
                                             +3*Sigma**4*dPhidr_int)\
                              -K*eps/Tau**2*(-2*(1+Omega**2)*Sigma*dPhidz_int*d2Phidz2_int\
                                             -Omega*Sigma**2*dPhidr_int*d2Phidz2_int\
                                             +5*Omega*Sigma**2*dPhidz_int*d2Phidrdz_int\
                                             +Sigma**3*dPhidr_int*d2Phidrdz_int\
                                             -3*Sigma**3*dPhidz_int*d2Phidr2_int\
                                             -Sigma**2*beta*dPhidr_int*dPhidz_int\
                                             -5*Omega*Sigma*beta*dPhidz_int**2)\
                              -K*eps**2/Tau**3*(+3*Sigma**3*dPhidr_int*dPhidz_int\
                                                -3*Omega*Sigma**2*dPhidz_int**2)
        
        return b.ravel() # Note: ravel gives a view, flatten creates a copy
    def _construct_Newton_system(self,u=None,Phi=None,K=None,G=None):
        """Constructs the linear system corresponding to a Newton iteration
        of the finite difference discretisation of the problem.
        This is constructed into a sparse matrix format.
        The system is constructed using u_old,Phi_old."""
        if u is None:
            u = self._u_old
        else:
            assert u.shape==self._shape
        if Phi is None:
            Phi = self._Phi_old
        else:
            assert Phi.shape==self._shape
        if K is None:
            K = self._K
        if G is None:
            G = self._G
                
        n = self._shape[0]
        m = self._shape[1]
        dr = self._dr
        dz = self._dz
        eps = self._epsilon
        alpha = self._alpha
        beta = self._beta
        if eps==0.0:
            Tau = 1.0
        else:
            Tau = self._Tau[1:-1,1:-1] # restrict to interior
        if isinstance(self._Sigma,float):
            Sigma = self._Sigma
        else:
            Sigma = self._Sigma[1:-1,1:-1] # restrict to interior
        if isinstance(self._Omega,float):
            Omega = self._Omega
        else:
            Omega = self._Omega[1:-1,1:-1] # restrict to interior
        
        b = self._generate_solver_residual(u,Phi,K,G)
        # Note the preceding line calls self._update_intermediates(u,Phi)
        
        # Possibly don't need reference to all of these here...
        dudr_int = self._dudr[1:-1,1:-1]
        dudz_int = self._dudz[1:-1,1:-1]
        d2udr2_int  = self._d2udr2[1:-1,1:-1]
        d2udrdz_int = self._d2udrdz[1:-1,1:-1]
        d2udz2_int  = self._d2udz2[1:-1,1:-1]
        dPhidr_int = self._dPhidr[1:-1,1:-1]
        dPhidz_int = self._dPhidz[1:-1,1:-1]
        d2Phidr2_int  = self._d2Phidr2[1:-1,1:-1]
        d2Phidrdz_int = self._d2Phidrdz[1:-1,1:-1]
        d2Phidz2_int  = self._d2Phidz2[1:-1,1:-1]
        d3Phidr3_int   = self._d3Phidr3[1:-1,1:-1]
        d3Phidr2dz_int = self._d3Phidr2dz[1:-1,1:-1]
        d3Phidrdz2_int = self._d3Phidrdz2[1:-1,1:-1]
        d3Phidz3_int   = self._d3Phidz3[1:-1,1:-1]
        d4Phidr4_int    = self._d4Phidr4[1:-1,1:-1]
        d4Phidr3dz_int  = self._d4Phidr3dz[1:-1,1:-1]
        d4Phidr2dz2_int = self._d4Phidr2dz2[1:-1,1:-1]
        d4Phidrdz3_int  = self._d4Phidrdz3[1:-1,1:-1]
        d4Phidz4_int    = self._d4Phidz4[1:-1,1:-1]
        
        # Start with the top row, first filling in the left block
        # Initialise the diagonals
        A00_p0_p0 = self._ones_edge() # set to 1 to enforce Dirichlet BC
        A00_p1_p0 = self._zeros_edge()
        A00_m1_p0 = self._zeros_edge()
        A00_p0_p1 = self._zeros_edge()
        A00_p0_m1 = self._zeros_edge()
        if alpha!=0.0 or beta!=0.0:
            A00_p1_p1 = self._zeros_edge()
            A00_p1_m1 = self._zeros_edge()
            A00_m1_p1 = self._zeros_edge()
            A00_m1_m1 = self._zeros_edge()
        # Add stencils for the LHS terms
        # d^2u/dr^2 stencil
        A00_p1_p0[1:-1,1:-1] =  1.0/dr**2 # first update
        A00_p0_p0[1:-1,1:-1] = -2.0/dr**2 # first update
        A00_m1_p0[1:-1,1:-1] =  1.0/dr**2 # first update
        # d^2u/dz^2 stencil
        temp = (1.0/dz**2)*(1+Omega**2)/Sigma**2 # Note: if beta=0 then temp is float so next line fails!
        A00_p0_p1[1:-1,1:-1]  = +  temp # first update
        A00_p0_p0[1:-1,1:-1] += -2*temp
        A00_p0_m1[1:-1,1:-1]  = +  temp # first update
        # d^2u/drdz stencil
        if alpha!=0.0 or beta!=0.0:
            temp = (-2.0/(4*dr*dz))*Omega/Sigma
            A00_p1_p1[1:-1,1:-1] = +temp # first update
            A00_p1_m1[1:-1,1:-1] = -temp # first update
            A00_m1_p1[1:-1,1:-1] = -temp # first update
            A00_m1_m1[1:-1,1:-1] = +temp # first update
        # du/dr stencil
        if eps!=0:
            temp = eps/(2*dr)/Tau
            A00_p1_p0[1:-1,1:-1] += +temp
            A00_m1_p0[1:-1,1:-1] += -temp
        # du/dz stencil
        temp = 2*beta*Omega/(2*dz*Sigma**2)
        if eps!=0:
            temp += -eps/(2*dz)*Omega/(Tau*Sigma)
        A00_p0_p1[1:-1,1:-1] += +temp
        A00_p0_m1[1:-1,1:-1] += -temp
        # u stencil
        if eps!=0.0:
            A00_p0_p0[1:-1,1:-1] += -(eps/Tau)**2
        # Now add stencils for the RHS terms
        if K!=0:
            # u_z*Phi_r stencil
            temp = -K*dPhidr_int/(2*dz*Tau*Sigma)
            A00_p0_p1[1:-1,1:-1] += +temp
            A00_p0_m1[1:-1,1:-1] += -temp
            # u_r*Phi_z stencil
            temp = +K*dPhidz_int/(2*dr*Tau*Sigma)
            A00_p1_p0[1:-1,1:-1] += +temp
            A00_m1_p0[1:-1,1:-1] += -temp
            # u*Phi_z stencil
            if eps!=0:
                A00_p0_p0[1:-1,1:-1] += eps*K*dPhidz_int/(Tau**2*Sigma)
        # Assemble the block
        if alpha==0.0 and beta==0.0:
            A00_block = diags([A00_p0_p0.ravel(),\
                               A00_p1_p0.ravel()[:-1],A00_m1_p0.ravel()[1:],\
                               A00_p0_p1.ravel()[:-m],A00_p0_m1.ravel()[m:]],\
                              [0,1,-1,m,-m],format='csr')
        else:
            A00_block = diags([A00_p0_p0.ravel(),\
                               A00_p1_p0.ravel()[:-1],A00_m1_p0.ravel()[1:],\
                               A00_p0_p1.ravel()[:-m],A00_p0_m1.ravel()[m:],\
                               A00_p1_p1.ravel()[:-m-1],A00_m1_p1.ravel()[:-m+1],\
                               A00_p1_m1.ravel()[m-1:],A00_m1_m1.ravel()[1+m:]],\
                              [0,1,-1,m,-m,m+1,m-1,-m+1,-m-1],format='csr')
        
        # Now the right block (of the top row)
        if K!=0:
            # Initialise the required diagonals
            A01_p1_p0 = self._zeros_edge()
            A01_m1_p0 = self._zeros_edge()
            A01_p0_p1 = self._zeros_edge()
            A01_p0_m1 = self._zeros_edge()
            # u*Phi_z and u_r*Phi_z stencils
            temp = +K*dudr_int/(2*dz*Tau*Sigma)
            if eps!=0:
                temp += eps*K*u[1:-1,1:-1]/(2*dz*Tau**2*Sigma)
            A01_p0_p1[1:-1,1:-1] = +temp # first update
            A01_p0_m1[1:-1,1:-1] = -temp # first update
            # u_z*Phi_r stencil
            temp = -K*dudz_int/(2*dr*Tau*Sigma)
            A01_p1_p0[1:-1,1:-1] = +temp # first update
            A01_m1_p0[1:-1,1:-1] = -temp # first update
            # Assemble the block
            A01_block = diags([A01_p1_p0.ravel()[:-1],A01_m1_p0.ravel()[1:],\
                               A01_p0_p1.ravel()[:-m],A01_p0_m1.ravel()[m:]], \
                              [1,-1,m,-m],format='csr')
        else:
            A01_block = None
        
        # Now move onto the second row, starting with the left block
        # Initialise the required diagonals
        A10_p0_p0 = self._zeros_edge()
        A10_p0_p1 = self._zeros_edge()
        A10_p0_m1 = self._zeros_edge()
        # Now fill in both contributions from u*u_z
        temp = -2*Sigma**3
        A10_p0_p0[1:-1,1:-1] =  temp*dudz_int # first update
        A10_p0_p1[1:-1,1:-1] = +temp/(2*dz)*u[1:-1,1:-1] # first update
        A10_p0_m1[1:-1,1:-1] = -temp/(2*dz)*u[1:-1,1:-1] # first update
        A10_block = diags([A10_p0_p0.ravel(),\
                           A10_p0_p1.ravel()[:-m],A10_p0_m1.ravel()[m:]],\
                          [0,m,-m],format='csr')
        
        # Now the bottom right block (the most difficult/laborious one)
        # Initialise the required diagonals
        A11_p0_p0 = self._ones_edge() # ones here to enforce Dirichlet BCs
        A11_p2_p0 = self._zeros_edge()
        A11_p1_p0 = self._zeros_edge()
        A11_m1_p0 = self._zeros_edge()
        A11_m2_p0 = self._zeros_edge()
        A11_p0_p2 = self._zeros_edge()
        A11_p0_p1 = self._zeros_edge()
        A11_p0_m1 = self._zeros_edge()
        A11_p0_m2 = self._zeros_edge()
        A11_p1_p1 = self._zeros_edge()
        A11_m1_p1 = self._zeros_edge()
        A11_p1_m1 = self._zeros_edge()
        A11_m1_m1 = self._zeros_edge()
        if alpha!=0 or beta!=0:
            A11_p1_p2 = self._zeros_edge()
            A11_m1_p2 = self._zeros_edge()
            A11_p1_m2 = self._zeros_edge()
            A11_m1_m2 = self._zeros_edge()
            A11_p2_p1 = self._zeros_edge()
            A11_m2_p1 = self._zeros_edge()
            A11_p2_m1 = self._zeros_edge()
            A11_m2_m1 = self._zeros_edge()
        # Start with some the RHS terms
        # Phi_zzzz stencil
        temp = (1+Omega**2)**2/dz**4
        if isinstance(temp,float):
            temp = np.full((n-2,m-2),temp) # easy fix, but not particularly efficient
        A11_p0_p2[1:-2,1:-1] = +  temp[:-1,:] # first update
        A11_p0_p1[1:-1,1:-1] = -4*temp        # first update
        A11_p0_p0[1:-1,1:-1] = +6*temp        # first update
        A11_p0_m1[1:-1,1:-1] = -4*temp        # first update
        A11_p0_m2[2:-1,1:-1] = +  temp[ 1:,:] # first update
        A11_p0_p2[-2,1:-1] = 0
        A11_p0_m2[ 1,1:-1] = 0
        A11_p0_p2_edge = temp[-1,:].copy()
        A11_p0_m2_edge = temp[ 0,:].copy()
        # Phi_rrrr stencil (Note: do this next to get 'first update's right)
        temp = Sigma**4/dr**4
        if isinstance(temp,float):
            temp = np.full((n-2,m-2),temp) # easy fix, but not particularly efficient
        A11_p2_p0[1:-1,1:-2]  = +  temp[:,:-1] # first update
        A11_p1_p0[1:-1,1:-1]  = -4*temp        # first update
        A11_p0_p0[1:-1,1:-1] += +6*temp
        A11_m1_p0[1:-1,1:-1]  = -4*temp        # first update
        A11_m2_p0[1:-1,2:-1]  = +  temp[:, 1:] # first update
        A11_p2_p0[1:-1,-2] = 0
        A11_m2_p0[1:-1, 1] = 0
        A11_p2_p0_edge = temp[:,-1].copy()
        A11_m2_p0_edge = temp[:, 0].copy()
        # Phi_rrzz stencil (Note: do this next to get 'first update's right)
        temp = (2+6*Omega**2)*Sigma**2/(dr**2*dz**2)
        A11_p1_p1[1:-1,1:-1]  = +  temp # first update
        A11_p1_p0[1:-1,1:-1] += -2*temp
        A11_p1_m1[1:-1,1:-1]  = +  temp # first update
        A11_p0_p1[1:-1,1:-1] += -2*temp
        A11_p0_p0[1:-1,1:-1] += +4*temp
        A11_p0_m1[1:-1,1:-1] += -2*temp
        A11_m1_p1[1:-1,1:-1]  = +  temp # first update
        A11_m1_p0[1:-1,1:-1] += -2*temp
        A11_m1_m1[1:-1,1:-1]  = +  temp # first update
        # Phi_rzzz stencil
        if alpha!=0 or beta!=0:
            temp = -4/(4*dr*dz**3)*Omega*(1+Omega**2)*Sigma
            if isinstance(temp,float):
                temp = np.full((n-2,m-2),temp) # easy fix, but not particularly efficient
            A11_p1_p2[1:-2,1:-1]  = +  temp[:-1,:] # first update
            A11_p1_p1[1:-1,1:-1] += -2*temp
            A11_p1_m1[1:-1,1:-1] += +2*temp
            A11_p1_m2[2:-1,1:-1]  = -  temp[ 1:,:] # first update
            A11_m1_p2[1:-2,1:-1]  = -  temp[:-1,:] # first update
            A11_m1_p1[1:-1,1:-1] += +2*temp
            A11_m1_m1[1:-1,1:-1] += -2*temp
            A11_m1_m2[2:-1,1:-1]  = +  temp[ 1:,:] # first update
            A11_p1_p2[-2,1:-1] = 0
            A11_p1_m2[ 1,1:-1] = 0
            A11_m1_p2[-2,1:-1] = 0
            A11_m1_m2[ 1,1:-1] = 0
            A11_p1_p2_edge =  temp[-1,:].copy()
            A11_p1_m2_edge = -temp[ 0,:].copy()
            A11_m1_p2_edge = -temp[-1,:].copy()
            A11_m1_m2_edge =  temp[ 0,:].copy()
        # Phi_rrrz stencil
        if alpha!=0 or beta!=0:
            temp = -4/(4*dr**3*dz)*Omega*Sigma**3
            if isinstance(temp,float):
                temp = np.full((n-2,m-2),temp) # easy fix, but not particularly efficient
            A11_p2_p1[1:-1,1:-2]  = +  temp[:,:-1] # first update
            A11_p1_p1[1:-1,1:-1] += -2*temp
            A11_m1_p1[1:-1,1:-1] += +2*temp
            A11_m2_p1[1:-1,2:-1]  = -  temp[:, 1:] # first update
            A11_p2_m1[1:-1,1:-2]  = -  temp[:,:-1] # first update
            A11_p1_m1[1:-1,1:-1] += +2*temp
            A11_m1_m1[1:-1,1:-1] += -2*temp
            A11_m2_m1[1:-1,2:-1]  = +  temp[:, 1:] # first update
            A11_p2_p1[1:-1,-2] = 0
            A11_m2_p1[1:-1, 1] = 0
            A11_p2_m1[1:-1,-2] = 0
            A11_m2_m1[1:-1, 1] = 0
            A11_p2_p1_edge =  temp[:,-1].copy()
            A11_m2_p1_edge = -temp[:, 0].copy()
            A11_p2_m1_edge = -temp[:,-1].copy()
            A11_m2_m1_edge =  temp[:, 0].copy()
        # Phi_zzz stencil
        if alpha!=0 or beta!=0:
            temp = 12*beta/(2*dz**3)*Omega*(1+Omega**2)
            if eps!=0:
                temp += 2*eps/(2*dz**3)*Sigma*Omega*(1+Omega**2)/Tau
            if isinstance(temp,float):
                temp = np.full((n-2,m-2),temp) # easy fix, but not particularly efficient
            A11_p0_p2[1:-2,1:-1] += +  temp[:-1,:]
            A11_p0_p1[1:-1,1:-1] += -2*temp
            A11_p0_m1[1:-1,1:-1] += +2*temp
            A11_p0_m2[2:-1,1:-1] += -  temp[ 1:,:]
            A11_p0_p2_edge +=  temp[-1,:]
            A11_p0_m2_edge += -temp[ 0,:]
        # Phi_rzz stencil
        temp = -beta/(2*dr*dz**2)*(8+24*Omega**2)*Sigma
        if eps!=0:
            temp += -eps/(2*dr*dz**2)*(2+6*Omega**2)*Sigma**2/Tau
        A11_p1_p1[1:-1,1:-1] += +  temp
        A11_p1_p0[1:-1,1:-1] += -2*temp
        A11_p1_m1[1:-1,1:-1] += +  temp
        A11_m1_p1[1:-1,1:-1] += -  temp
        A11_m1_p0[1:-1,1:-1] += +2*temp
        A11_m1_m1[1:-1,1:-1] += -  temp
        # Phi_rrz stencil
        if alpha!=0 or beta!=0:
            temp = 12/(2*dr**2*dz)*beta*Omega*Sigma**2
            if eps!=0:
                temp += 6*eps/(2*dr**2*dz)*Omega*Sigma**3/Tau
            A11_p1_p1[1:-1,1:-1] += +  temp
            A11_p0_p1[1:-1,1:-1] += -2*temp
            A11_m1_p1[1:-1,1:-1] += +  temp
            A11_p1_m1[1:-1,1:-1] += -  temp
            A11_p0_m1[1:-1,1:-1] += +2*temp
            A11_m1_m1[1:-1,1:-1] += -  temp
        # Phi_rrr stencil
        if eps!=0:
            temp = -2*eps/(2*dr**3)*Sigma**4/Tau
            if isinstance(temp,float):
                temp = np.full((n-2,m-2),temp) # easy fix, but not particularly efficient
            A11_p2_p0[1:-1,1:-2] += +  temp[:,:-1]
            A11_p1_p0[1:-1,1:-1] += -2*temp
            A11_m1_p0[1:-1,1:-1] += +2*temp
            A11_m2_p0[1:-1,2:-1] += -  temp[:, 1:]
            A11_p2_p0_edge +=  temp[:,-1]
            A11_m2_p0_edge += -temp[:, 0]
        # Phi_zz stencil
        temp = beta**2/dz**2*(12+36*Omega**2)
        if eps!=0:
            temp += beta*eps/dz**2*Sigma*(4+12*Omega**2)/Tau
            if alpha!=0 or beta!=0:
                temp += 3/dz**2*(eps*Sigma*Omega/Tau)**2
        A11_p0_p1[1:-1,1:-1] += +  temp
        A11_p0_p0[1:-1,1:-1] += -2*temp
        A11_p0_m1[1:-1,1:-1] += +  temp
        # Phi_rz stencil
        if alpha!=0 or beta!=0:
            temp = -24/(4*dr*dz)*beta**2*Sigma*Omega
            if eps!=0:
                temp += -12*eps/(4*dr*dz)*beta*Sigma**2*Omega/Tau
                temp += -6/(4*dr*dz)*Sigma**3*Omega*(eps/Tau)**2
            A11_p1_p1[1:-1,1:-1] += +temp
            A11_p1_m1[1:-1,1:-1] += -temp
            A11_m1_p1[1:-1,1:-1] += -temp
            A11_m1_m1[1:-1,1:-1] += +temp
        # Phi_rr stencil
        if eps!=0:
            temp = 3/dr**2*(eps/Tau)**2*Sigma**4
            A11_p1_p0[1:-1,1:-1] += +  temp
            A11_p0_p0[1:-1,1:-1] += -2*temp
            A11_m1_p0[1:-1,1:-1] += +  temp
        # Phi_z stencil
        if alpha!=0 or beta!=0:
            temp = 24/(2*dz)*beta**3*Omega
            if eps!=0:
                temp += 12/(2*dz)*beta**2*Omega*(eps*Sigma/Tau)
                temp += 6/(2*dz)*beta*Omega*(eps*Sigma/Tau)**2
                temp += 3/(2*dz)*Omega*(eps*Sigma/Tau)**3
            A11_p0_p1[1:-1,1:-1] += +temp
            A11_p0_m1[1:-1,1:-1] += -temp
        # Phi_r stencil
        if eps!=0:
            temp = -3*eps**3/(2*dr)*Sigma**4/Tau**3
            A11_p1_p0[1:-1,1:-1] += +temp
            A11_m1_p0[1:-1,1:-1] += -temp
        # Now proceed with the LHS terms
        if K!=0:
            # [Phi_zz,Phi] term (Phi_rzz*Phi_z-Phi_zzz*Phi_r)
            temp = K*(1+Omega**2)*Sigma/Tau
            if isinstance(temp,float):
                temp = np.full((n-2,m-2),temp) # easy fix, but not particularly efficient
            A11_p0_p1[1:-1,1:-1] += +temp*d3Phidrdz2_int/(2*dz)
            A11_p0_m1[1:-1,1:-1] += -temp*d3Phidrdz2_int/(2*dz)
            A11_p1_p1[1:-1,1:-1] += +  temp*dPhidz_int/(2*dr*dz**2)
            A11_p1_p0[1:-1,1:-1] += -2*temp*dPhidz_int/(2*dr*dz**2)
            A11_p1_m1[1:-1,1:-1] += +  temp*dPhidz_int/(2*dr*dz**2)
            A11_m1_p1[1:-1,1:-1] += -  temp*dPhidz_int/(2*dr*dz**2)
            A11_m1_p0[1:-1,1:-1] += +2*temp*dPhidz_int/(2*dr*dz**2)
            A11_m1_m1[1:-1,1:-1] += -  temp*dPhidz_int/(2*dr*dz**2)
            A11_p1_p0[1:-1,1:-1] += -temp*d3Phidz3_int/(2*dr)
            A11_m1_p0[1:-1,1:-1] += +temp*d3Phidz3_int/(2*dr)
            A11_p0_p2[1:-2,1:-1] += -  temp[:-1,:]*dPhidr_int[:-1,:]/(2*dz**3)
            A11_p0_p1[1:-1,1:-1] += +2*temp*dPhidr_int/(2*dz**3)
            A11_p0_m1[1:-1,1:-1] += -2*temp*dPhidr_int/(2*dz**3)
            A11_p0_m2[2:-1,1:-1] += +  temp[ 1:,:]*dPhidr_int[ 1:,:]/(2*dz**3)
            A11_p0_p2_edge += -temp[-1,:]*dPhidr_int[-1,:]/(2*dz**3)
            A11_p0_m2_edge += +temp[ 0,:]*dPhidr_int[ 0,:]/(2*dz**3)
            # [Phi,Phi_rz] term (Phi_r*Phi_rzz-Phi_z*Phi_rrz)
            if alpha!=0 or beta!=0:
                temp = 2*K*Omega*Sigma**2/Tau
                A11_p1_p0[1:-1,1:-1] += +temp*d3Phidrdz2_int/(2*dr)
                A11_m1_p0[1:-1,1:-1] += -temp*d3Phidrdz2_int/(2*dr)
                A11_p1_p1[1:-1,1:-1] += +  temp*dPhidr_int/(2*dr*dz**2)
                A11_p1_p0[1:-1,1:-1] += -2*temp*dPhidr_int/(2*dr*dz**2)
                A11_p1_m1[1:-1,1:-1] += +  temp*dPhidr_int/(2*dr*dz**2)
                A11_m1_p1[1:-1,1:-1] += -  temp*dPhidr_int/(2*dr*dz**2)
                A11_m1_p0[1:-1,1:-1] += +2*temp*dPhidr_int/(2*dr*dz**2)
                A11_m1_m1[1:-1,1:-1] += -  temp*dPhidr_int/(2*dr*dz**2)
                A11_p0_p1[1:-1,1:-1] += -temp*d3Phidr2dz_int/(2*dz)
                A11_p0_m1[1:-1,1:-1] += +temp*d3Phidr2dz_int/(2*dz)
                A11_p1_p1[1:-1,1:-1] += -  temp*dPhidz_int/(2*dr**2*dz)
                A11_p0_p1[1:-1,1:-1] += +2*temp*dPhidz_int/(2*dr**2*dz)
                A11_m1_p1[1:-1,1:-1] += -  temp*dPhidz_int/(2*dr**2*dz)
                A11_p1_m1[1:-1,1:-1] += +  temp*dPhidz_int/(2*dr**2*dz)
                A11_p0_m1[1:-1,1:-1] += -2*temp*dPhidz_int/(2*dr**2*dz)
                A11_m1_m1[1:-1,1:-1] += +  temp*dPhidz_int/(2*dr**2*dz)
            # [Phi_rr,Phi] term (Phi_rrr*Phi_z-Phi_rrz*Phi_r)
            temp = K*Sigma**3/Tau
            if isinstance(temp,float):
                temp = np.full((n-2,m-2),temp) # easy fix, but not particularly efficient
            A11_p0_p1[1:-1,1:-1] += +temp*d3Phidr3_int/(2*dz)
            A11_p0_m1[1:-1,1:-1] += -temp*d3Phidr3_int/(2*dz)
            A11_p2_p0[1:-1,1:-2] += +  temp[:,:-1]*dPhidz_int[:,:-1]/(2*dr**3)
            A11_p1_p0[1:-1,1:-1] += -2*temp*dPhidz_int/(2*dr**3)
            A11_m1_p0[1:-1,1:-1] += +2*temp*dPhidz_int/(2*dr**3)
            A11_m2_p0[1:-1,2:-1] += -  temp[:, 1:]*dPhidz_int[:, 1:]/(2*dr**3)
            A11_p2_p0_edge += +temp[:,-1]*dPhidz_int[:,-1]/(2*dr**3)
            A11_m2_p0_edge += -temp[:, 0]*dPhidz_int[:, 0]/(2*dr**3)
            A11_p1_p0[1:-1,1:-1] += -temp*d3Phidr2dz_int/(2*dr)
            A11_m1_p0[1:-1,1:-1] += +temp*d3Phidr2dz_int/(2*dr)
            A11_p1_p1[1:-1,1:-1] += -  temp*dPhidr_int/(2*dr**2*dz)
            A11_p0_p1[1:-1,1:-1] += +2*temp*dPhidr_int/(2*dr**2*dz)
            A11_m1_p1[1:-1,1:-1] += -  temp*dPhidr_int/(2*dr**2*dz)
            A11_p1_m1[1:-1,1:-1] += +  temp*dPhidr_int/(2*dr**2*dz)
            A11_p0_m1[1:-1,1:-1] += -2*temp*dPhidr_int/(2*dr**2*dz)
            A11_m1_m1[1:-1,1:-1] += +  temp*dPhidr_int/(2*dr**2*dz)
            # Phi_z*Phi_zz term
            if beta!=0 or eps!=0:
                temp = 0
                if beta!=0:
                    temp += -2*K*beta*(1+Omega**2)/Tau
                if eps!=0:
                    temp += -2*K*eps*(1+Omega**2)*Sigma/Tau**2
                A11_p0_p1[1:-1,1:-1] += +temp*d2Phidz2_int/(2*dz)
                A11_p0_m1[1:-1,1:-1] += -temp*d2Phidz2_int/(2*dz)
                A11_p0_p1[1:-1,1:-1] += +  temp*dPhidz_int/(dz**2)
                A11_p0_p0[1:-1,1:-1] += -2*temp*dPhidz_int/(dz**2)
                A11_p0_m1[1:-1,1:-1] += +  temp*dPhidz_int/(dz**2)
            # Phi_r*Phi_zz term
            if alpha!=0 or beta!=0:
                temp = -4*K*beta*Omega*Sigma/Tau # sign?
                if eps!=0:
                    temp += -K*eps*Omega*(Sigma/Tau)**2
                A11_p1_p0[1:-1,1:-1] += +temp*d2Phidz2_int/(2*dr)
                A11_m1_p0[1:-1,1:-1] += -temp*d2Phidz2_int/(2*dr)
                A11_p0_p1[1:-1,1:-1] += +  temp*dPhidr_int/(dz**2)
                A11_p0_p0[1:-1,1:-1] += -2*temp*dPhidr_int/(dz**2)
                A11_p0_m1[1:-1,1:-1] += +  temp*dPhidr_int/(dz**2)
            # Phi_z*Phi_rz term
            if alpha!=0 or beta!=0:
                temp = +4*K*beta*Omega*Sigma/Tau # sign?
                if eps!=0:
                    temp += 5*K*eps*Omega*(Sigma/Tau)**2
                A11_p0_p1[1:-1,1:-1] += +temp*d2Phidrdz_int/(2*dz)
                A11_p0_m1[1:-1,1:-1] += -temp*d2Phidrdz_int/(2*dz)
                A11_p1_p1[1:-1,1:-1] += +temp*dPhidz_int/(4*dr*dz)
                A11_m1_p1[1:-1,1:-1] += -temp*dPhidz_int/(4*dr*dz)
                A11_p1_m1[1:-1,1:-1] += -temp*dPhidz_int/(4*dr*dz)
                A11_m1_m1[1:-1,1:-1] += +temp*dPhidz_int/(4*dr*dz)
            # Phi_r*Phi_rz term
            temp = 2*K*beta*Sigma**2/Tau
            if eps!=0:
                temp += K*eps*Sigma**3/Tau**2
            A11_p1_p0[1:-1,1:-1] += +temp*d2Phidrdz_int/(2*dr)
            A11_m1_p0[1:-1,1:-1] += -temp*d2Phidrdz_int/(2*dr)
            A11_p1_p1[1:-1,1:-1] += +temp*dPhidr_int/(4*dr*dz)
            A11_m1_p1[1:-1,1:-1] += -temp*dPhidr_int/(4*dr*dz)
            A11_p1_m1[1:-1,1:-1] += -temp*dPhidr_int/(4*dr*dz)
            A11_m1_m1[1:-1,1:-1] += +temp*dPhidr_int/(4*dr*dz)
            # Phi_z*Phi_rr term
            if eps!=0:
                temp = -3*K*eps*Sigma**3/Tau**2
                A11_p0_p1[1:-1,1:-1] += +temp*d2Phidr2_int/(2*dz)
                A11_p0_m1[1:-1,1:-1] += -temp*d2Phidr2_int/(2*dz)
                A11_p1_p0[1:-1,1:-1] += +  temp*dPhidz_int/(dr**2)
                A11_p0_p0[1:-1,1:-1] += -2*temp*dPhidz_int/(dr**2)
                A11_m1_p0[1:-1,1:-1] += +  temp*dPhidz_int/(dr**2)
            # Phi_r*Phi_z term
            temp = -2*K*beta**2*(Sigma/Tau)
            if eps!=0:
                temp += -K*eps*beta*(Sigma/Tau)**2
                temp += +3*K*eps**2*(Sigma/Tau)**3
            A11_p0_p1[1:-1,1:-1] += +temp*dPhidr_int/(2*dz)
            A11_p0_m1[1:-1,1:-1] += -temp*dPhidr_int/(2*dz)
            A11_p1_p0[1:-1,1:-1] += +temp*dPhidz_int/(2*dr)
            A11_m1_p0[1:-1,1:-1] += -temp*dPhidz_int/(2*dr)
            # Phi_z**2 term
            if alpha!=0 or beta!=0:
                temp = -4*K*beta**2/(2*dz)*Omega*dPhidz_int/Tau
                if eps!=0:
                    temp += -5*K*beta*eps/(2*dz)*dPhidz_int*Omega*Sigma/Tau**2
                    temp += -3*K*eps**2/(2*dz)*dPhidz_int*Omega*Sigma**2/Tau**3
                A11_p0_p1[1:-1,1:-1] += +2*temp # (additional factor 2 because two identical contributions)
                A11_p0_m1[1:-1,1:-1] += -2*temp
        # Now, 'fix' the edges in each of the above terms that have a 5 wide stencil
        # The following function makes edge fixes more convenient
        def boundary_adjustment(edge_diags,edge_data,slc):
            if self._use_first_order_extrap: #+3/2,-1,+1/2
                edge_diags[0][slc] += 1.5*edge_data
                edge_diags[1][slc] -= 1.0*edge_data
                edge_diags[2][slc] += 0.5*edge_data
            else: #+2,-5/2,+2,-1/2
                edge_diags[0][slc] += 2.0*edge_data
                edge_diags[1][slc] -= 2.5*edge_data
                edge_diags[2][slc] += 2.0*edge_data
                edge_diags[3][slc] -= 0.5*edge_data
            if self._add_third_deriv_extrap: #+1/6,-1/2,+1/2,-1/6
                edge_diags[0][slc] += 1.0/6.0*edge_data
                edge_diags[1][slc] -= 0.5*edge_data
                edge_diags[2][slc] += 0.5*edge_data
                edge_diags[3][slc] -= 1.0/6.0*edge_data
        # Apply to the centred +-2 stencils
        boundary_adjustment([A11_m1_p0,A11_p0_p0,A11_p1_p0,A11_p2_p0],
                            A11_m2_p0_edge,(slice(1,-1), 1))
        boundary_adjustment([A11_p1_p0,A11_p0_p0,A11_m1_p0,A11_m2_p0],
                            A11_p2_p0_edge,(slice(1,-1),-2))
        boundary_adjustment([A11_p0_m1,A11_p0_p0,A11_p0_p1,A11_p0_p2],
                            A11_p0_m2_edge,( 1,slice(1,-1)))
        boundary_adjustment([A11_p0_p1,A11_p0_p0,A11_p0_m1,A11_p0_m2],
                            A11_p0_p2_edge,(-2,slice(1,-1)))
        # Apply to the off-centre +-2 stencils if required
        if alpha!=0 or beta!=0:
            boundary_adjustment([A11_m1_p1,A11_p0_p1,A11_p1_p1,A11_p2_p1],
                                A11_m2_p1_edge,(slice(1,-1), 1))
            boundary_adjustment([A11_m1_m1,A11_p0_m1,A11_p1_m1,A11_p2_m1],
                                A11_m2_m1_edge,(slice(1,-1), 1))
            boundary_adjustment([A11_p1_p1,A11_p0_p1,A11_m1_p1,A11_m2_p1],
                                A11_p2_p1_edge,(slice(1,-1),-2))
            boundary_adjustment([A11_p1_m1,A11_p0_m1,A11_m1_m1,A11_m2_m1],
                                A11_p2_m1_edge,(slice(1,-1),-2))
            boundary_adjustment([A11_p1_m1,A11_p1_p0,A11_p1_p1,A11_p1_p2],
                                A11_p1_m2_edge,( 1,slice(1,-1)))
            boundary_adjustment([A11_m1_m1,A11_m1_p0,A11_m1_p1,A11_m1_p2],
                                A11_m1_m2_edge,( 1,slice(1,-1)))
            boundary_adjustment([A11_p1_p1,A11_p1_p0,A11_p1_m1,A11_p1_m2],
                                A11_p1_p2_edge,(-2,slice(1,-1)))
            boundary_adjustment([A11_m1_p1,A11_m1_p0,A11_m1_m1,A11_m1_m2],
                                A11_m1_p2_edge,(-2,slice(1,-1)))
        # Now complete construction of the block from the diagonals
        if alpha==0 and beta==0:
            A11_block = diags([A11_p0_p0.ravel(),\
                               A11_p2_p0.ravel()[:-2],A11_p1_p0.ravel()[:-1],\
                               A11_m1_p0.ravel()[1:],A11_m2_p0.ravel()[2:],\
                               A11_p0_p2.ravel()[:-2*m],A11_p0_p1.ravel()[:-m],\
                               A11_p0_m1.ravel()[m:],A11_p0_m2.ravel()[2*m:],\
                               A11_p1_p1.ravel()[:-1-m],A11_m1_p1.ravel()[:-m+1],\
                               A11_p1_m1.ravel()[m-1:],A11_m1_m1.ravel()[m+1:]],\
                              [0,2,1,-1,-2,2*m,m,-m,-2*m,1+m,-1+m,1-m,-1-m],format='csr')
        else:
            A11_block = diags([A11_p0_p0.ravel(),\
                               A11_p2_p0.ravel()[:-2]  ,A11_p1_p0.ravel()[:-1],\
                               A11_m1_p0.ravel()[1:]   ,A11_m2_p0.ravel()[2:],\
                               A11_p0_p2.ravel()[:-2*m],A11_p0_p1.ravel()[:-m],\
                               A11_p0_m1.ravel()[m:]   ,A11_p0_m2.ravel()[2*m:],\
                               A11_p1_p1.ravel()[:-1-m],A11_m1_p1.ravel()[:-m+1],\
                               A11_p1_m1.ravel()[m-1:] ,A11_m1_m1.ravel()[m+1:],\
                               A11_p2_p1.ravel()[:-2-m],A11_m2_p1.ravel()[:-m+2],\
                               A11_p2_m1.ravel()[m-2:] ,A11_m2_m1.ravel()[m+2:],\
                               A11_p1_p2.ravel()[:-1-2*m],A11_m1_p2.ravel()[:-2*m+1],\
                               A11_p1_m2.ravel()[2*m-1:] ,A11_m1_m2.ravel()[2*m+1:]],\
                              [0,2,1,-1,-2,2*m,m,-m,-2*m,1+m,-1+m,1-m,-1-m,\
                               m+2,m-2,-m+2,-m-2,2*m+1,2*m-1,-2*m+1,-2*m-1],format='csr')
        
        # Complete the assembly of the full matrix from the blocks
        A = bmat([[A00_block,A01_block],[A10_block,A11_block]],format='csr')
        
        return A,b
        
    def get_PDE_residual(self,u,Phi,Phi_interior_only=False):
        """Computes a finite difference estimate of the PDE residual given
        an approximation of u,Phi on a regular grid.
        Note that this residual is based on equation 3 in my ANZIAM paper,
        i.e. I have not multiplied or divided by R=1+epsilon*S as has been done for the solver.
        Values of m,n,W,H,epsilon,K,G are taken to be those specified within the class.
        
        Parameters
        ----------
        u,Phi : np.array,np.array
            Approximations of the axial velocity u and stream-function of the secondary flow Phi
            Each must have the shape (n,m) and are assumed to be a regularly spaced full grid
        Phi_interior_only : bool
            If True then the residual for Phi is only calculated on the very interior,
            that is two points away from the boundary.
            (This can be useful for comparing the interior solution with that obtained
             from other codes/implementations that do a poor job near the boundary.)
        
        Returns
        -------
        np.array,np.array
            The residuals of the u and Phi PDE equations respectively
        """
        
        assert u.shape==self._shape
        assert Phi.shape==self._shape
        
        b = self._generate_solver_residual(u,Phi)
        ru   = b[:self._N].reshape(self._shape)
        rPhi = b[self._N:].reshape(self._shape)
        
        # Why bother with this...
        if self._epsilon!=0.0:
            ru   *= self._Tau
            rPhi /= self._Tau
        
        if Phi_interior_only:
            rPhi[[1,-2],1:-1] = 0
            rPhi[2:-2,[1,-2]] = 0
        
        return ru,rPhi
    def _Newton_solver_iterate(self,return_matrix=False):
        """An implementation of a fully coupled Newton iteration in
        sparse matrix format for solving the curved duct flow equations.
        
        Parameters
        ----------
        return_matrix : bool
            If true then the (spares) matrix and RHS vector will be returned
            (rather than the actual solution),
            this can be useful for debugging, or if a custom linear solver is to be used.
        
        Returns
        -------
        np.array,np.array
            An updated approximation of u,Phi respectively
        """
        
        if self._u_old_is_zero and self._use_initial_u_iterate:
            self._initial_u_iterate()
        
        A,b = self._construct_Newton_system()
        
        if return_matrix:
            return A,b
        
        duPhi = spsolve(A,b).reshape((2,)+self._shape)
        
        # Note: we won't update self._u_old,self._Phi_old in this low level function
        
        return duPhi[0]+self._u_old,duPhi[1]+self._Phi_old

    def solve(self,K=None,K_ramp=None,tol=1.0E-10,ramp_tol=1.0E-3,max_its=20,iterative=False,summary=False,verbose=False):
        """
        This automates the full solution of the Dean flow problem.
        The default solversettings are desined to work well for the default
        values of H,G (and W>=H) in the class constructor. In other cases, or
        for very large K, the solver parameters may need tweaking.
        
        Parameters
        ----------
        K : float (>=0)
            Specify the desired 'Dean number' K to solve for.
            If specified it will update the value stored within the class.
            If None then it will update the value stored within the class.
        K_ramp : [float (>=0),float (>=0)]
            If specified then it gives a starting K and a step size to be used
            in relation to solving for large K via progressively increasing K.
            This is particularly relevant if one provides a non-default value of
            any of H,W,G, or if a an initial guess is provided via the solution
            of another (possibly large) K. Otherwise the default is to ramp K from
            500 and increase in steps of 500 until the desired K is reached.
        tol : float
            Specifies a convergence tolerance
        ramp_tol : float
            Specifies a larger tolerance to use during K ramp iterations
        max_its : int
            Specifies a maximum iterations of inner solve (shouldn't need to be high
            when combined with appropriate ramping of K values)
        iterative : bool
            Toggle between iterative and direct solver
            (Not fully supported: I've not had much luck with iterative solvers at high resolution)
        summary : bool
            Print a summary of convergence and iterations on successful completion
            (When ramping the Dean number this prints at the end of the iterations for each K)
            (Note: unsuccessful completion always prints a warning)
        verbose : bool
            Print convergence information after every iteration.
            If this is True then a summary will also be printed at the end
            (i.e. overrides the summary variable)
        
        Returns
        -------
        np.array,np.array
            The flow solution u,Phi respectively
        """
        if K is None:
            K = self._K
        else:
            self._K = K
        if K_ramp is None:
            # These default parameters are based on a default W,H,G
            # and are not guaranteed to work otherwise
            K_ramp = [min(500.0,K),500.0]
        else:
            assert len(K_ramp)==2
            
        # Extra setup in the case of an iterative solver...
        # (although this is not fully tested/supported...)
        if iterative:
            from scipy.sparse.linalg import gmres,bicgstab,spilu,LinearOperator
            iter_solver = bicgstab # bicgstab generally seems to be about twice as fast as gmres
            iter_tol = 1.0E-9
            preconditioner = "ilu" # ILU is the only viable option I've found...
        
        # Perform an initial u iterate if required...
        if self._u_old_is_zero and self._use_initial_u_iterate:
            self._initial_u_iterate()
        
        u   = self._u_old   # we avoid a copy here so that self._u_old is updated
        Phi = self._Phi_old # we avoid a copy here so that self._Phi_old is updated
        
        # Generate list of K values to solve through
        if K<=K_ramp[0]:
            #K_list = [K] # this dis-allows backward ramping
            K_list = list(np.arange(K_ramp[0],K,K_ramp[1]))+[K] # this approach allows 'backward' ramping
        else:
            K_list = list(np.arange(K_ramp[0],K,K_ramp[1]))+[K]
        for k in K_list:
            its = 0
            err = 1.0
            temp_tol = ramp_tol
            if k==K:
                temp_tol = tol
            while err>temp_tol:
                A,b = self._construct_Newton_system(u,Phi,k)
                if iterative:
                    # Note: iterative is untested in this trapezoidal duct version
                    res = 0
                    if preconditioner=="ilu":
                        B = spilu(A.tocsc(),drop_tol=1.0E-9,fill_factor=32.0) # drop_tol and fill_factor may need adjusting for different resolutions and values of K???
                        M = LinearOperator(A.get_shape(),lambda x:B.solve(x))
                        res = iter_solver(A,b.ravel(),M=M,tol=iter_tol)
                    elif preconditioner=="diag":
                        DI = 1.0/A.diagonal(0)
                        M = LinearOperator(A.get_shape(),lambda x:x*DI)
                        res = iter_solver(A,b.ravel(),M=M,tol=iter_tol)
                    elif preconditioner=="jacobi":
                        D = A.diagonal(0)
                        M = LinearOperator(A.get_shape(),lambda x:(x-A.dot(x)+D*x)/D)
                        res = iter_solver(A,b.ravel(),M=M,tol=iter_tol)
                    else:
                        res = iter_solver(A,b.ravel(),tol=iter_tol)
                    if res[1]!=0:
                        print("Error: itertive method was not successful... error code:",res[1])
                    duPhi = res[0].reshape((2,)+self._shape)
                    u   += duPhi[0]
                    Phi += duPhi[1]
                    u_err = np.linalg.norm(duPhi[0])/np.linalg.norm(u)
                    Phi_err = np.linalg.norm(duPhi[1])/np.linalg.norm(Phi)
                    err = max(u_err,Phi_err)
                else:
                    duPhi = spsolve(A,b.ravel()).reshape((2,)+self._shape)
                    u   += duPhi[0]
                    Phi += duPhi[1]
                    u_err = np.linalg.norm(duPhi[0])/np.linalg.norm(u)
                    Phi_err = np.linalg.norm(duPhi[1])/np.linalg.norm(Phi)
                    err = max(u_err,Phi_err)
                its += 1
                if its>=max_its:
                    print("Warning: max_its reached, terminating early...")
                    # I used this to see check that bifurcations are likely to be numerical artefacts
                    # It is too slow/memory intensive to use for more extensive analysis though...
                    #print("conditioning:",np.linalg.cond(A.todense()))
                    break
                if verbose: # Optional next level debugging...
                    print("    Error on iteration {:d} is {:.6E}, {:.6E}".format(its,u_err,Phi_err))
            if summary or verbose:
                print("Convergence for K={:g} reached in {:d} iterations (err,tol={:.6E},{:g})".format(k,its,err,temp_tol))
                
        return u,Phi

    def _solver_test(self):
        """A built-in test routine of the implementation above"""
        import matplotlib.pyplot as plt
        # Setup function for plotting
        S,Z = self.get_meshgrid()
        def plot_pair(u,Phi):
            plt.contourf(S,Z,u,32)
            plt.colorbar()
            plt.gca().set_aspect(1.0)
            plt.show()
            plt.contourf(S,Z,Phi,32)
            plt.colorbar()
            plt.gca().set_aspect(1.0)
            plt.show()
        # Set an initial K, solve the equations (with manual iterations), plot the solution
        self.set_K(500.0)
        print("Solving for K={:.1f}".format(self.get_K()))
        u0_s,Phi0_s = self._Newton_solver_iterate()
        if False: # Optionally view the result of the first iteration...
            plot_pair(u0_s,Phi0_s)
        for k in range(20):
            self.set_initial_guess(u0_s,Phi0_s)
            u1_s,Phi1_s = self._Newton_solver_iterate()
            u_conv = np.linalg.norm(u1_s-u0_s)/np.linalg.norm(u1_s)
            Phi_conv = np.linalg.norm(Phi1_s-Phi0_s)/np.linalg.norm(Phi1_s)
            print("    relative errors:",u_conv,Phi_conv)
            u0_s[:,:] = u1_s[:,:]
            Phi0_s[:,:] = Phi1_s[:,:]
            if max(u_conv,Phi_conv)<1.0E-10:
                print("    converged!")
                break
        print("Plotting solutions:")
        plot_pair(u1_s,Phi1_s)
        # Plot the PDE residual
        print("Plotting PDE residuals:")
        ru,rPhi = self.get_PDE_residual(u1_s,Phi1_s)
        plot_pair(ru,rPhi)
        # Try pushing K higher...
        self.set_initial_guess(u1_s,Phi1_s)
        self.set_K(5000.0)
        print("Solving for K={:.1f} via ramping".format(self.get_K()))
        u,Phi = self.solve(K_ramp=[1000.0,500.0],verbose=True)
        print("Plotting solutions:")
        plot_pair(u,Phi)
        # Plot the PDE residual
        print("Plotting PDE residuals:")
        ru,rPhi = self.get_PDE_residual(u,Phi)
        plot_pair(ru,rPhi)
        # Done
    # End of class
