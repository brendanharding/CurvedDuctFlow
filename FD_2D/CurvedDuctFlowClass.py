import numpy as np
from scipy.sparse import diags,bmat
from scipy.sparse.linalg import spsolve

class CurvedDuctFlowClass(object):
    """This class provides a finite difference implementation for solving
    steady flow through a curved duct having a rectangular cross-section.
    Solutions may be computed both with and without using the Dean
    approximation (solutions in this are often referred to as Dean flow).
    
    The specific equations and non-dimensionalisation used within this class
    are described in my publication here doi.org/10.1017/S1446181118000287
    (noting, however, that this paper describes an alternative numerical method
    to solve these equations for a larger variety of cross-section shapes).
    Specifically, refer to the non-linear PDE given in equations (2.3).
    
    The solver uses Newton's method to solve the fully coupled non-linear problem.
    The linear system for each iteration is contructed in sparse matrix format
    and solved using sparse LU by default (although using bicgstab is an option).
    The discretisation is such that the solutions achieve second order
    convergence with respect to grid resolution.
 
    A note on correct non-dimensional use:
    You are free to choose any characteristic length X and velocity U that you wish.
    You need to pass the appropriate (non-dimensional) W,H along with epsilon=X/R
    and K=epsilon*Re^2 where Re=(rho/mu)*U*X. To obtain the correct flow you must
    now determine G such that the axial velocity of the non-dimensional solution
    respects your chosen velocity scale. For example, if your U is to be the mean axial
    velocity then G needs to be such that the mean of the returned axial velocity is 0.
    Finding the correct G is a non-linear optimisation problem (with non-zero K)
    and will require some form of iteration.
    
    A note on correct dimensional use:
    Pass your dimensional W,H values along with epsilon=1/R and K=epsilon*Re^2
    where Re=(rho/mu) in this instance (with appropriate units). You then need
    to determine G such that the returned axial velocity field meets some desired
    criterion (i.e. you might be aiming for a specifc flow rate). Again, this is a
    non-linear optimisation problem that you, the user, are responsible for solving.
    """
    def __init__(self,m=65,n=65,W=2.0,H=2.0,epsilon=0.01,K=1.0,G=4.0):
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
			The curvature of the duct, that is X/R given a bend radius R
            and length scale X.
            Providing a value of 0 means the solution corresponds to 'Dean flow'.
            Defaults to 0.0
        K : float
            The Dean number used in the equations (specifically K=epsilon*Re^2).
            Observe this can be specified as non-zero even if epsilon=0.
            Defaults to 1.0
        G : float
            The pressure gradient driving the axial flow.
			This needs to be chosen to respect your desired velocity scale U
            (see the notes on correct non-dimensional use).
            You should only change this if you understand the consequences on
            the scaling of both the axial velocity and stream-function.
            Defaults to 4.0
        """
        self._shape = (n,m)
        self._size  = (H,W)
        self._epsilon = epsilon
        self._K = K
        self._G = G
        # Define other useful numbers
        self._N = n*m
        self._ds = W/(m-1)
        self._dz = H/(n-1)
        # Construct arrays/meshgrids for the finite difference node coordinates (8*(2*m+n+3*n*m) bytes)
        self._s = np.linspace(-0.5*W,0.5*W,m)
        self._z = np.linspace(-0.5*H,0.5*H,n)
        self._S,self._Z = np.meshgrid(self._s,self._z) # Note: use default index order
        self._r = 1.0+epsilon*self._s # probably not really needed...
        self._R = 1.0+epsilon*self._S
        # Pre-allocate array for holding an initial guess (2*8*m*n bytes)
        self._u_old = np.zeros((n,m))
        self._Phi_old = np.zeros((n,m))
        self._u_old_is_zero = True # tracks whether an initial u iteration should be performed
        # Pre-allocate arrays of intermediate results (11*8*m*n bytes)
        self._duds = np.zeros((n,m))
        self._dudz = np.zeros((n,m))
        self._Lu = np.zeros((n,m))
        self._dPhids = np.zeros((n,m))
        self._dPhidz = np.zeros((n,m))
        self._d2Phidsdz = np.zeros((n,m))
        self._d2Phids2 = np.zeros((n,m))
        self._d2Phidz2 = np.zeros((n,m))
        #self._LPhi = np.zeros((n,m))   # don't need a persistent copy of this
        self._dLPhids = np.zeros((n,m))
        self._dLPhidz = np.zeros((n,m))
        self._L2Phi = np.zeros((n,m))
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
        del self._s
        del self._z
        del self._r
        del self._S
        del self._Z
        del self._R
        del self._u_old
        del self._Phi_old
        del self._duds
        del self._dudz
        del self._Lu
        del self._dPhids
        del self._dPhidz
        del self._d2Phidsdz
        del self._d2Phids2
        del self._d2Phidz2
        del self._dLPhids
        del self._dLPhidz
        del self._L2Phi
        # Done
    def get_meshgrid(self):
        """Returns the meshgrid of (s,z) coordinates for the finite
        difference points."""
        return self._S,self._Z
    def get_secondary_velocities(self,Phi=None):
        """Returns the velocity fields associated with a stream-function
        of the secondary flow.
        Note the velocities will be given as exactly zero on the boundary."""
        if Phi is None:
            Phi = self._Phi_old
        else:
            assert Phi.shape==self._shape
        V = self._zeros_edge()
        W = self._zeros_edge()
        V[1:-1,1:-1] = -self._FD_dz(Phi)/self._R[1:-1,1:-1]
        W[1:-1,1:-1] =  self._FD_ds(Phi)/self._R[1:-1,1:-1]
        return V,W
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
        """Set a (dimensionless) driving pressure G"""
        self._G = G
    def get_G(self):
        """Get the (dimensionless) driving pressure gradient G"""
        return self._G
    def set_epsilon(self,epsilon):
        """Set a new curvature ratio epsilon=X/R
		(R being the bend radius and X the characteristic length scale)
        Recall setting this to zero results in the 'Dean approximation'"""
        self._epsilon = epsilon
        self._r = 1.0+self._epsilon*self._s # probably not really needed...
        self._R = 1.0+self._epsilon*self._S # important to update this!!!
    def get_epsilon(self):
        """Get the current curvature ratio epsilon=X/R
		(R being the bend radius and X the characteristic length scale)"""
        return self._epsilon
	# Note: One should probably create a new class instance in order to change any other variables
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
        """Construct and solve a Poisson problem to create an initial
        guess for the axial velocity u (which is stored in u_old)."""
        shape = self._shape
        m = shape[1]
        # Total memory for A,b is ~8*8*n*m bytes (including the 3 temporary arrays)
        A_p0_p0 = self._ones_edge()
        A_p0_p0[1:-1,1:-1] = -2.0/self._ds**2-2.0/self._dz**2
        A_pm1_p0 = self._zeros_edge()
        A_pm1_p0[1:-1,1:-1] = 1.0/self._ds**2
        A_p0_pm1 = self._zeros_edge()
        A_p0_pm1[1:-1,1:-1] = 1.0/self._dz**2
        b = self._zeros_edge()
        if self._epsilon==0.0:
            b[1:-1,1:-1] = -self._G
        else:
            b[1:-1,1:-1] = -self._G/self._R[1:-1,1:-1]
            # possibly also modify A_pm1_p0 with duds terms...
        A = diags([A_p0_p0.ravel(),\
                   A_pm1_p0.ravel()[:-1],A_pm1_p0.ravel()[1:],\
                   A_p0_pm1.ravel()[:-m],A_p0_pm1.ravel()[m:]],\
                  [0,1,-1,m,-m],format='csr')
        # It is difficult to predict how much memory spsolve will use
        self._u_old = spsolve(A,b.ravel()).reshape(shape)
        self._u_old_is_zero = False
        return
    def _FD_ds(self,array,include_edges=False):
        """Return centred finite difference estimate of the
        s (or x) derivative on the interior of the given array"""
        if include_edges:
            temp = np.empty(array.shape)
            temp[:,1:-1] = (array[:,2:]-array[:,:-2])/(2.0*self._ds)
            temp[:, 0] = (-3*array[:, 0]+4*array[:, 1]-array[:, 2])/(2.0*self._ds)
            temp[:,-1] = (+3*array[:,-1]-4*array[:,-2]+array[:,-3])/(2.0*self._ds)
            return temp
        else:
            return (array[1:-1,2:]-array[1:-1,:-2])/(2.0*self._ds)
    def _FD_dz(self,array,include_edges=False):
        """Return centred finite difference estimate of the
        z (or y) derivative on the interior of the given array"""
        if include_edges:
            temp = np.empty(array.shape)
            temp[1:-1,:] = (array[2:,:]-array[:-2,:])/(2.0*self._dz)
            temp[ 0,:] = (-3*array[ 0,:]+4*array[ 1,:]-array[ 2,:])/(2.0*self._dz)
            temp[-1,:] = (+3*array[-1,:]-4*array[-2,:]+array[-3,:])/(2.0*self._dz)
            return temp
        else:
            return (array[2:,1:-1]-array[:-2,1:-1])/(2.0*self._dz)
    def _FD_ds2(self,array,include_edges=False):
        """Return centred finite difference estimate of the second
        s (or x) derivative on the interior of the given array"""
        if include_edges:
            temp = np.empty(array.shape)
            temp[:,1:-1] = (array[:,2:]-2.0*array[:,1:-1]+array[:,:-2])/self._ds**2
            temp[:, 0] = (2*array[:, 0]-5*array[:, 1]+4*array[:, 2]-array[:, 3])/self._ds**2
            temp[:,-1] = (2*array[:,-1]-5*array[:,-2]+4*array[:,-3]-array[:,-4])/self._ds**2
            return temp
        else:
            return (array[1:-1,2:]-2.0*array[1:-1,1:-1]+array[1:-1,:-2])/self._ds**2
    def _FD_dsdz(self,array,include_edges=False):
        """Return centred finite difference estimate of the mixed second
        s,z (or x,y) derivative on the interior of the given array"""
        if include_edges:
            return self._FD_dz(self._FD_ds(array,True),True)
        else:
            return (array[2:,2:]-array[2:,:-2]-array[:-2,2:]+array[:-2,:-2])/(4.0*self._ds*self._dz)
    def _FD_dz2(self,array,include_edges=False):
        """Return centred finite difference estimate of the second
        z (or y) derivative on the interior of the given array"""
        if include_edges:
            temp = np.empty(array.shape)
            temp[1:-1,:] = (array[2:,:]-2.0*array[1:-1,:]+array[:-2,:])/self._dz**2
            temp[ 0,:] = (2*array[ 0,:]-5*array[ 1,:]+4*array[ 2,:]-array[ 4,:])/self._dz**2
            temp[-1,:] = (2*array[-1,:]-5*array[-2,:]+4*array[-3,:]-array[-4,:])/self._dz**2
            return temp
        else:
            return (array[2:,1:-1]-2.0*array[1:-1,1:-1]+array[:-2,1:-1])/self._dz**2
    def _FD_Laplace(self,array,include_edges=False):
        """Return centred finite difference estimate of the
        Laplacian on the interior of the given array"""
        return self._FD_ds2(array,include_edges)+self._FD_dz2(array,include_edges)
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
        # TODO: determine which (if any) of the following can be avoided when epsilon==0
        self._duds[1:-1,1:-1] = self._FD_ds(u)
        self._dudz[1:-1,1:-1] = self._FD_dz(u)
        self._Lu[1:-1,1:-1] = self._FD_Laplace(u)
        self._dPhids[1:-1,1:-1] = self._FD_ds(Phi)
        self._dPhidz[1:-1,1:-1] = self._FD_dz(Phi)
        self._d2Phidsdz[1:-1,1:-1] = self._FD_dsdz(Phi)
        Phi_ghost = self._create_ghost_array(Phi)     # this is only used for intermediate calculation
        self._d2Phids2[:,:] = self._FD_ds2(Phi_ghost)
        self._d2Phidz2[:,:] = self._FD_dz2(Phi_ghost)
        LPhi = self._d2Phids2 + self._d2Phidz2        # this is only used for intermediate calculation
        self._dLPhids[1:-1,1:-1] = self._FD_ds(LPhi)
        self._dLPhidz[1:-1,1:-1] = self._FD_dz(LPhi)
        self._L2Phi[1:-1,1:-1] = self._FD_Laplace(LPhi)
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
        C_int = 1.0
        if eps!=0.0:
            C_int = 1.0/self._R[1:-1,1:-1]
            
        self._update_intermediates(u,Phi)
        duds_int = self._duds[1:-1,1:-1]
        dudz_int = self._dudz[1:-1,1:-1]
        Lu_int = self._Lu[1:-1,1:-1]
        dPhids_int = self._dPhids[1:-1,1:-1]
        dPhidz_int = self._dPhidz[1:-1,1:-1]
        d2Phids2_int = self._d2Phids2[1:-1,1:-1]
        d2Phidz2_int = self._d2Phidz2[1:-1,1:-1]
        L2Phi_int = self._L2Phi[1:-1,1:-1]
        d2Phidsdz_int = self._d2Phidsdz[1:-1,1:-1]
        dLPhids_int = self._dLPhids[1:-1,1:-1]
        dLPhidz_int = self._dLPhidz[1:-1,1:-1]
        
        b = np.empty((2,)+shape)
        b[:,[0,-1],:] = 0    # 0's on the edge/boundary
        b[:,1:-1,[0,-1]] = 0
        b[0,1:-1,1:-1] = -G*C_int-Lu_int \
                         -K*C_int*(dPhidz_int*duds_int-dPhids_int*dudz_int)
        b[1,1:-1,1:-1] =  2.0*u[1:-1,1:-1]*dudz_int-L2Phi_int \
                         +K*C_int*(dPhids_int*dLPhidz_int-dPhidz_int*dLPhids_int)
        if eps!=0.0:
            b[0,1:-1,1:-1] += eps*((eps-K*dPhidz_int)*u[1:-1,1:-1]*C_int-duds_int)*C_int
            b[1,1:-1,1:-1] +=  2.0*eps*dLPhids_int*C_int \
                              -3.0*eps**2*d2Phids2_int*C_int**2 \
                              +3.0*eps**3*dPhids_int*C_int**3 \
                              +eps*K*( dPhidz_int*(2.0*d2Phidz2_int+3.0*d2Phids2_int) \
                                      -dPhids_int*d2Phidsdz_int)*C_int**2 \
                              -3.0*eps**2*K*dPhidz_int*dPhids_int*C_int**3
        
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
        
        m = self._shape[1]
        ds = self._ds
        dz = self._dz
        eps = self._epsilon
        C_int = 1.0
        if eps!=0.0:
            C_int = 1.0/self._R[1:-1,1:-1]
        
        b = self._generate_solver_residual(u,Phi,K,G)
        # Note the preceding line calls self._update_intermediates(u,Phi)
        
        duds_int = self._duds[1:-1,1:-1]
        dudz_int = self._dudz[1:-1,1:-1]
        Lu_int = self._Lu[1:-1,1:-1]
        dPhids_int = self._dPhids[1:-1,1:-1]
        dPhidz_int = self._dPhidz[1:-1,1:-1]
        d2Phids2_int = self._d2Phids2[1:-1,1:-1]
        d2Phidz2_int = self._d2Phidz2[1:-1,1:-1]
        L2Phi_int = self._L2Phi[1:-1,1:-1]
        d2Phidsdz_int = self._d2Phidsdz[1:-1,1:-1]
        dLPhids_int = self._dLPhids[1:-1,1:-1]
        dLPhidz_int = self._dLPhidz[1:-1,1:-1]
        
        # TODO: consider re-using 'temporary' arrays within here...
        
        A00_p0_p0 = self._ones_edge()
        A00_p0_p0[1:-1,1:-1] = -2.0/ds**2-2.0/dz**2
        A00_p1_p0 = self._zeros_edge()
        A00_p1_p0[1:-1,1:-1] = ( K/(2.0*ds))*C_int*dPhidz_int
        if eps!=0.0:
            A00_p0_p0[1:-1,1:-1] += (eps*K*dPhidz_int-eps**2)*C_int**2
            A00_p1_p0[1:-1,1:-1] += (eps/(2.0*ds))*C_int
        A00_m1_p0 = -A00_p1_p0
        A00_p1_p0[1:-1,1:-1] += 1.0/ds**2
        A00_m1_p0[1:-1,1:-1] += 1.0/ds**2
        A00_p0_p1 = self._zeros_edge()
        A00_p0_p1[1:-1,1:-1] = (-K/(2.0*dz))*C_int*dPhids_int
        A00_p0_m1 = -A00_p0_p1
        A00_p0_p1[1:-1,1:-1] += 1.0/dz**2
        A00_p0_m1[1:-1,1:-1] += 1.0/dz**2
        A00_block = diags([A00_p0_p0.ravel(),\
                           A00_p1_p0.ravel()[:-1],A00_m1_p0.ravel()[1:],\
                           A00_p0_p1.ravel()[:-m],A00_p0_m1.ravel()[m:]],\
                          [0,1,-1,m,-m],format='csr')
        
        A01_p1_p0 = self._zeros_edge()
        A01_p1_p0[1:-1,1:-1] = (-K/(2.0*ds))*C_int*dudz_int
        #A01_m1_p0 = -A01_p1_p0
        A01_p0_p1 = self._zeros_edge()
        A01_p0_p1[1:-1,1:-1] = ( K/(2.0*dz))*C_int*duds_int
        if eps!=0.0:
            A01_p0_p1[1:-1,1:-1] += (eps*K/(2.0*dz))*u[1:-1,1:-1]*C_int**2
        #A01_p0_m1 = -A01_p0_p1
        #A01_block = diags([A01_p1_p0.ravel()[:-1],A01_m1_p0.ravel()[1:],\
        #                   A01_p0_p1.ravel()[:-m],A01_p0_m1.ravel()[m:]], \
        #                  [1,-1,m,-m],format='csr')
        A01_block = diags([A01_p1_p0.ravel()[:-1],-A01_p1_p0.ravel()[1:],\
                           A01_p0_p1.ravel()[:-m],-A01_p0_p1.ravel()[m:]], \
                          [1,-1,m,-m],format='csr')
        
        A10_p0_p0 = self._zeros_edge()
        A10_p0_p1 = self._zeros_edge()
        A10_p0_p0[1:-1,1:-1] = -2.0*dudz_int
        A10_p0_p1[1:-1,1:-1] = (-1.0/dz)*u[1:-1,1:-1] # factor 2 cancels from 2*dz
        #A10_p0_m1 = -A10_p0_p1
        #A10_block = diags([A10_p0_p0.ravel(),\
        #                   A10_p0_p1.ravel()[:-m],A10_p0_m1.ravel()[m:]],\
        #                  [0,m,-m],format='csr')
        A10_block = diags([A10_p0_p0.ravel(),\
                           A10_p0_p1.ravel()[:-m],-A10_p0_p1.ravel()[m:]],\
                          [0,m,-m],format='csr')
        
        A11_p0_p0 = self._ones_edge()
        A11_p0_p0[1:-1,1:-1] =  6.0/ds**4+6.0/dz**4+8.0/(ds*dz)**2
        A11_p2_p0 = self._zeros_edge()
        A11_p1_p0 = self._zeros_edge()
        A11_p1_p0[1:-1,1:-1] = -4.0/ds**4-4.0/(ds*dz)**2 \
                               -K/(2.0*ds)*C_int*dLPhidz_int \
                               -K*(1.0/ds**3+1.0/(ds*dz**2))*C_int*dPhidz_int
        A11_m1_p0 = self._zeros_edge()
        A11_m1_p0[1:-1,1:-1] = -4.0/ds**4-4.0/(ds*dz)**2 \
                               +K/(2.0*ds)*C_int*dLPhidz_int \
                               +K*(1.0/ds**3+1.0/(ds*dz**2))*C_int*dPhidz_int
        A11_m2_p0 = self._zeros_edge()
        A11_p0_p2 = self._zeros_edge()
        A11_p0_p1 = self._zeros_edge()
        A11_p0_p1[1:-1,1:-1] = -4.0/dz**4-4.0/(ds*dz)**2 \
                               +K/(2.0*dz)*C_int*dLPhids_int \
                               +K*(1.0/dz**3+1.0/(dz*ds**2))*C_int*dPhids_int
        A11_p0_m1 = self._zeros_edge()
        A11_p0_m1[1:-1,1:-1] = -4.0/dz**4-4.0/(ds*dz)**2 \
                               -K/(2.0*dz)*C_int*dLPhids_int \
                               -K*(1.0/dz**3+1.0/(dz*ds**2))*C_int*dPhids_int
        A11_p0_m2 = self._zeros_edge()
        A11_p1_p1 = self._zeros_edge()
        A11_p1_p1[1:-1,1:-1] = 2.0/(ds*dz)**2+K/(2.0*ds*dz**2)*C_int*dPhidz_int \
                                             -K/(2.0*dz*ds**2)*C_int*dPhids_int
        A11_m1_p1 = self._zeros_edge()
        A11_m1_p1[1:-1,1:-1] = 2.0/(ds*dz)**2-K/(2.0*ds*dz**2)*C_int*dPhidz_int \
                                             -K/(2.0*dz*ds**2)*C_int*dPhids_int
        A11_p1_m1 = self._zeros_edge()
        A11_p1_m1[1:-1,1:-1] = 2.0/(ds*dz)**2+K/(2.0*ds*dz**2)*C_int*dPhidz_int \
                                             +K/(2.0*dz*ds**2)*C_int*dPhids_int
        A11_m1_m1 = self._zeros_edge()
        A11_m1_m1[1:-1,1:-1] = 2.0/(ds*dz)**2-K/(2.0*ds*dz**2)*C_int*dPhidz_int \
                                             +K/(2.0*dz*ds**2)*C_int*dPhids_int
        
        if eps!=0.0:
            A11_p0_p0[1:-1,1:-1] -= ( 6.0*(eps/ds)**2 \
                                     -(4.0*K*eps/dz**2+6.0*K*eps/ds**2)*dPhidz_int)*C_int**2
            A11_p1_p0[1:-1,1:-1] +=  2.0*(eps/(ds*dz**2)+eps/ds**3)*C_int \
                                    +( 3.0*(eps/ds)**2-3.0*K*eps/ds**2*dPhidz_int \
                                      +K*eps/(2.0*ds)*d2Phidsdz_int)*C_int**2 \
                                    +1.5*(-eps**3/ds+K*eps**2/ds*dPhidz_int)*C_int**3
            A11_m1_p0[1:-1,1:-1] += -2.0*(eps/(ds*dz**2)+eps/ds**3)*C_int \
                                    +( 3.0*(eps/ds)**2-3.0*K*eps/ds**2*dPhidz_int \
                                      -K*eps/(2.0*ds)*d2Phidsdz_int)*C_int**2 \
                                    +1.5*( eps**3/ds-K*eps**2/ds*dPhidz_int)*C_int**3
            A11_p0_p1[1:-1,1:-1] -=  2.0*K*eps*C_int**2*( 0.5/dz*d2Phidz2_int+1.0/dz**2*dPhidz_int) \
                                    +1.5/dz*K*eps*C_int**2*d2Phids2_int \
                                    -1.5*K*eps**2/dz*C_int**3*dPhids_int
            A11_p0_m1[1:-1,1:-1] -=  2.0*K*eps*C_int**2*(-0.5/dz*d2Phidz2_int+1.0/dz**2*dPhidz_int) \
                                    -1.5/dz*K*eps*C_int**2*d2Phids2_int \
                                    +1.5*K*eps**2/dz*C_int**3*dPhids_int
            A11_p1_p1[1:-1,1:-1] -=  eps/(ds*dz**2)*C_int \
                                    -K*eps/(4.0*ds*dz)*C_int**2*dPhids_int
            A11_p1_m1[1:-1,1:-1] -=  eps/(ds*dz**2)*C_int \
                                    +K*eps/(4.0*ds*dz)*C_int**2*dPhids_int
            A11_m1_p1[1:-1,1:-1] +=  eps/(ds*dz**2)*C_int \
                                    -K*eps/(4.0*ds*dz)*C_int**2*dPhids_int
            A11_m1_m1[1:-1,1:-1] +=  eps/(ds*dz**2)*C_int \
                                    +K*eps/(4.0*ds*dz)*C_int**2*dPhids_int

        if eps==0.0:
            temp_p2_p0 = 1.0/ds**4+0.5*K/ds**3*dPhidz_int
            temp_m2_p0 = 1.0/ds**4-0.5*K/ds**3*dPhidz_int
        else:
            temp_p2_p0 = 1.0/ds**4+(0.5*K/ds**3*dPhidz_int-eps/ds**3)*C_int
            temp_m2_p0 = 1.0/ds**4-(0.5*K/ds**3*dPhidz_int-eps/ds**3)*C_int
        A11_p2_p0[1:-1,1:-2] = temp_p2_p0[:, :-1] # p2 part remains as is for left edge
        A11_m2_p0[1:-1,2:-1] = temp_m2_p0[:,1:  ] # m2 part remains as is for right edge
        A11_p2_p0[1:-1,-2] = 0.0
        A11_m2_p0[1:-1, 1] = 0.0
        # adjust left side appropriately
        if self._use_first_order_extrap: #+3/2,-1,+1/2
            A11_m1_p0[1:-1, 1] += 1.5*temp_m2_p0[:,0]
            A11_p0_p0[1:-1, 1] -= 1.0*temp_m2_p0[:,0]
            A11_p1_p0[1:-1, 1] += 0.5*temp_m2_p0[:,0]
        else: #+2,-5/2,+2,-1/2
            A11_m1_p0[1:-1, 1] += 2.0*temp_m2_p0[:,0]
            A11_p0_p0[1:-1, 1] -= 2.5*temp_m2_p0[:,0]
            A11_p1_p0[1:-1, 1] += 2.0*temp_m2_p0[:,0]
            A11_p2_p0[1:-1, 1] -= 0.5*temp_m2_p0[:,0]
        if self._add_third_deriv_extrap: #+1/6,-1/2,+1/2,-1/6
            A11_m1_p0[1:-1, 1] += 1.0/6.0*temp_m2_p0[:,0]
            A11_p0_p0[1:-1, 1] -= 0.5*temp_m2_p0[:,0]
            A11_p1_p0[1:-1, 1] += 0.5*temp_m2_p0[:,0]
            A11_p2_p0[1:-1, 1] -= 1.0/6.0*temp_m2_p0[:,0]
        # adjust right side appropriately
        if self._use_first_order_extrap: #+3/2,-1,+1/2
            A11_p1_p0[1:-1,-2] += 1.5*temp_p2_p0[:,-1]
            A11_p0_p0[1:-1,-2] -= 1.0*temp_p2_p0[:,-1]
            A11_m1_p0[1:-1,-2] += 0.5*temp_p2_p0[:,-1]
        else: #+2,-5/2,+2,-1/2
            A11_p1_p0[1:-1,-2] += 2.0*temp_p2_p0[:,-1]
            A11_p0_p0[1:-1,-2] -= 2.5*temp_p2_p0[:,-1]
            A11_m1_p0[1:-1,-2] += 2.0*temp_p2_p0[:,-1]
            A11_m2_p0[1:-1,-2] -= 0.5*temp_p2_p0[:,-1]
        if self._add_third_deriv_extrap: #+1/6,-1/2,+1/2,-1/6
            A11_p1_p0[1:-1,-2] += 1.0/6.0*temp_p2_p0[:,-1]
            A11_p0_p0[1:-1,-2] -= 0.5*temp_p2_p0[:,-1]
            A11_m1_p0[1:-1,-2] += 0.5*temp_p2_p0[:,-1]
            A11_m2_p0[1:-1,-2] -= 1.0/6.0*temp_p2_p0[:,-1]
        #
        temp_p0_p2 = 1.0/dz**4-K/(2.0*dz**3)*C_int*dPhids_int
        temp_p0_m2 = 1.0/dz**4+K/(2.0*dz**3)*C_int*dPhids_int
        A11_p0_p2[1:-2,1:-1] = temp_p0_p2[ :-1,:] # p2 part remains as is for bottom edge
        A11_p0_m2[2:-1,1:-1] = temp_p0_m2[1:  ,:] # m2 part remains as is for top edge
        A11_p0_p2[-2,1:-1] = 0.0
        A11_p0_m2[ 1,1:-1] = 0.0
        # adjust bottom side appropriately
        if self._use_first_order_extrap: #+3/2,-1,+1/2
            A11_p0_m1[ 1,1:-1] += 1.5*temp_p0_m2[0,:]
            A11_p0_p0[ 1,1:-1] -= 1.0*temp_p0_m2[0,:]
            A11_p0_p1[ 1,1:-1] += 0.5*temp_p0_m2[0,:]
        else: #+2,-5/2,+2,-1/2
            A11_p0_m1[ 1,1:-1] += 2.0*temp_p0_m2[0,:]
            A11_p0_p0[ 1,1:-1] -= 2.5*temp_p0_m2[0,:]
            A11_p0_p1[ 1,1:-1] += 2.0*temp_p0_m2[0,:]
            A11_p0_p2[ 1,1:-1] -= 0.5*temp_p0_m2[0,:]
        if self._add_third_deriv_extrap: #+1/6,-1/2,+1/2,-1/6
            A11_p0_m1[ 1,1:-1] += 1.0/6.0*temp_p0_m2[0,:]
            A11_p0_p0[ 1,1:-1] -= 0.5*temp_p0_m2[0,:]
            A11_p0_p1[ 1,1:-1] += 0.5*temp_p0_m2[0,:]
            A11_p0_p2[ 1,1:-1] -= 1.0/6.0*temp_p0_m2[0,:]
        # adjust top side appropriately
        if self._use_first_order_extrap: #+3/2,-1,+1/2
            A11_p0_p1[-2,1:-1] += 1.5*temp_p0_p2[-1,:]
            A11_p0_p0[-2,1:-1] -= 1.0*temp_p0_p2[-1,:]
            A11_p0_m1[-2,1:-1] += 0.5*temp_p0_p2[-1,:]
        else: #+2,-5/2,+2,-1/2
            A11_p0_p1[-2,1:-1] += 2.0*temp_p0_p2[-1,:]
            A11_p0_p0[-2,1:-1] -= 2.5*temp_p0_p2[-1,:]
            A11_p0_m1[-2,1:-1] += 2.0*temp_p0_p2[-1,:]
            A11_p0_m2[-2,1:-1] -= 0.5*temp_p0_p2[-1,:]
        if self._add_third_deriv_extrap: #+1/6,-1/2,+1/2,-1/6
            A11_p0_p1[-2,1:-1] += 1.0/6.0*temp_p0_p2[-1,:]
            A11_p0_p0[-2,1:-1] -= 0.5*temp_p0_p2[-1,:]
            A11_p0_m1[-2,1:-1] += 0.5*temp_p0_p2[-1,:]
            A11_p0_m2[-2,1:-1] -= 1.0/6.0*temp_p0_p2[-1,:]
        
        A11_block = diags([A11_p0_p0.ravel(),\
                           A11_p2_p0.ravel()[:-2],A11_p1_p0.ravel()[:-1],\
                           A11_m1_p0.ravel()[1:],A11_m2_p0.ravel()[2:],\
                           A11_p0_p2.ravel()[:-2*m],A11_p0_p1.ravel()[:-m],\
                           A11_p0_m1.ravel()[m:],A11_p0_m2.ravel()[2*m:],\
                           A11_p1_p1.ravel()[:-1-m],A11_m1_p1.ravel()[:-m+1],\
                           A11_p1_m1.ravel()[m-1:],A11_m1_m1.ravel()[m+1:]],\
                          [0,2,1,-1,-2,2*m,m,-m,-2*m,1+m,-1+m,1-m,-1-m],format='csr')
        
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
        
        if self._epsilon!=0.0:
            ru   *= self._R
            rPhi /= self._R
        
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

    def solve(self,K=None,K_ramp=None,tol=1.0E-10,ramp_tol=1.0E-3,max_its=20,iterative=False,verbose=False):
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
        verbose : bool
            Toggle the verbosity of the solver (mostly just prints convergence info)
        
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
        
        # TODO: consider if we want to implement more generally, e.g. for G,H,W,epsilon which are not the current class values (although I don't think it makes sense to do so...)
        
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
                # TODO: Consider modifying the following to avoid re-allocating temporary arrays every iteration
                A,b = self._construct_Newton_system(u,Phi,k)
                if iterative:
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
                    err = max(np.linalg.norm(duPhi[0])/np.linalg.norm(u),\
                              np.linalg.norm(duPhi[1])/np.linalg.norm(Phi))
                else:
                    duPhi = spsolve(A,b.ravel()).reshape((2,)+self._shape)
                    u   += duPhi[0]
                    Phi += duPhi[1]
                    err = max(np.linalg.norm(duPhi[0])/np.linalg.norm(u),\
                              np.linalg.norm(duPhi[1])/np.linalg.norm(Phi))
                its += 1
                if its>=max_its:
                    print("Warning: max_its reached, terminating early...")
                    # I used this to see check that bifurcations are likely to be numerical artefacts
                    # It is too slow/memory intensive to use for more extensive analysis though...
                    #print("conditioning:",np.linalg.cond(A.todense()))
                    break
                if verbose and False: # Optional next level debugging...
                    print("Error on iteration ",its," is ",err)
            if verbose:
                print("Convergence for K=",k," reached in ",its," iterations (err,tol=",err,temp_tol,")")
                
        return u,Phi
        
    def _pressure_solve(self,u=None,Phi=None):
        """A routine to solve for the residual pressure.
        (Note: we use a finite element discretisation for this, it works
        much better on account of the Neumann boundary conditions.)"""
        # If u and/or Phi are not supplied, use the internal ones
        if u is None:
            u = self._u_old
        if Phi is None:
            Phi = self._Phi_old
        v,w = self.get_secondary_velocities(Phi)
        
        # Fetch constants
        m = self._shape[1]
        n = self._shape[0]
        ds = self._ds
        dz = self._dz
        eps = self._epsilon
        K = self._K
        _R = self._R
        
        # Compute some intermediate derivatives
        dvds = self._FD_ds(v,True)
        dvdz = self._FD_dz(v,True)
        dwds = self._FD_ds(w,True)
        dwdz = self._FD_dz(w,True)
        
        # Setup the pressure Laplacian (with Neumann BC's)
        # -\iiint grad(p).grad(phi) dV = \iiint div(grad(p))*phi dV - \iint (grad(p).n)*phi dS
        Ap_p0_p0 = np.zeros(u.shape)
        Ap_p1_p0 = np.zeros(u.shape)
        Ap_m1_p0 = np.zeros(u.shape)
        Ap_p0_p1 = np.zeros(u.shape)
        Ap_p1_p1 = np.zeros(u.shape)
        Ap_m1_p1 = np.zeros(u.shape)
        Ap_p0_m1 = np.zeros(u.shape)
        Ap_p1_m1 = np.zeros(u.shape)
        Ap_m1_m1 = np.zeros(u.shape)
        # Interior integrals (LHS)
        if eps==0:
            # test (i,j) -> trial +( 0, 0), integrate over each quadrant (as applicable)
            Ap_p0_p0[1:  ,1:  ] -= (+ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/3)
            Ap_p0_p0[1:  , :-1] -= (+ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/3)
            Ap_p0_p0[ :-1,1:  ] -= (+ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/3)
            Ap_p0_p0[ :-1, :-1] -= (+ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/3)
            # test (i,j) -> trial +(+1, 0), integrate over each quadrant (as applicable)
            Ap_p1_p0[1:  , :-1] -= (-ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/6)
            Ap_p1_p0[ :-1, :-1] -= (-ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/6)
            # test (i,j) -> trial +(-1, 0), integrate over each quadrant (as applicable)
            Ap_m1_p0[1:  ,1:  ] -= (-ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/6)
            Ap_m1_p0[ :-1,1:  ] -= (-ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/6)
            # test (i,j) -> trial +( 0,+1), integrate over each quadrant (as applicable)
            Ap_p0_p1[ :-1,1:  ] -= (+ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/3)
            Ap_p0_p1[ :-1, :-1] -= (+ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/3)
            # test (i,j) -> trial +( 0,-1), integrate over each quadrant (as applicable)
            Ap_p0_m1[1:  ,1:  ] -= (+ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/3)
            Ap_p0_m1[1:  , :-1] -= (+ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/3)
            # test (i,j) -> trial +(+1,+1) and other corners, integrate over relevant quadrant
            Ap_p1_p1[ :-1, :-1] -= (-ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/6)
            Ap_m1_p1[ :-1,1:  ] -= (-ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/6)
            Ap_p1_m1[1:  , :-1] -= (-ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/6)
            Ap_m1_m1[1:  ,1:  ] -= (-ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/6)
        else:
            # Need to account for the integrating factor
            # test (i,j) -> trial +( 0, 0), integrate over each quadrant (as applicable)
            Ap_p0_p0[1:  ,1:  ] -= _R[1:  ,1:  ]*((+ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/3))
            Ap_p0_p0[1:  , :-1] -= _R[1:  , :-1]*((+ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/3))
            Ap_p0_p0[ :-1,1:  ] -= _R[ :-1,1:  ]*((+ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/3))
            Ap_p0_p0[ :-1, :-1] -= _R[ :-1, :-1]*((+ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/3))
            # test (i,j) -> trial +(+1, 0), integrate over each quadrant (as applicable)
            Ap_p1_p0[1:  , :-1] -= _R[1:  ,1:  ]*((-ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/6))
            Ap_p1_p0[ :-1, :-1] -= _R[ :-1,1:  ]*((-ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/6))
            # test (i,j) -> trial +(-1, 0), integrate over each quadrant (as applicable)
            Ap_m1_p0[1:  ,1:  ] -= _R[1:  , :-1]*((-ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/6))
            Ap_m1_p0[ :-1,1:  ] -= _R[ :-1, :-1]*((-ds/ds**2)*(dz/3)+(+dz/dz**2)*(ds/6))
            # test (i,j) -> trial +( 0,+1), integrate over each quadrant (as applicable)
            Ap_p0_p1[ :-1,1:  ] -= _R[1:  ,1:  ]*((+ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/3))
            Ap_p0_p1[ :-1, :-1] -= _R[1:  , :-1]*((+ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/3))
            # test (i,j) -> trial +( 0,-1), integrate over each quadrant (as applicable)
            Ap_p0_m1[1:  ,1:  ] -= _R[ :-1,1:  ]*((+ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/3))
            Ap_p0_m1[1:  , :-1] -= _R[ :-1, :-1]*((+ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/3))
            # test (i,j) -> trial +(+1,+1) and other corners, integrate over relevant quadrant
            Ap_p1_p1[ :-1, :-1] -= _R[1:  ,1:  ]*((-ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/6))
            Ap_m1_p1[ :-1,1:  ] -= _R[1:  , :-1]*((-ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/6))
            Ap_p1_m1[1:  , :-1] -= _R[ :-1,1:  ]*((-ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/6))
            Ap_m1_m1[1:  ,1:  ] -= _R[ :-1, :-1]*((-ds/ds**2)*(dz/6)+(-dz/dz**2)*(ds/6))
        # Interior integrals (RHS)
        bp = np.zeros(u.shape)
        rhs_form = 0
        temp =  2.0/_R*u*self._FD_ds(u,True)\
               -K*(+dvds**2\
                   +2.0*dvdz*dwds\
                   +dwdz**2)
        if eps!=0:
            temp += -K*(eps*v/_R)**2
            temp *= _R # integrating factor
        bp[1:  ,1:  ] += temp[1:  ,1:  ]*(ds/3)*(dz/3) # +( 0, 0)
        bp[1:  , :-1] += temp[1:  , :-1]*(ds/3)*(dz/3)
        bp[ :-1,1:  ] += temp[ :-1,1:  ]*(ds/3)*(dz/3)
        bp[ :-1, :-1] += temp[ :-1, :-1]*(ds/3)*(dz/3)
        bp[1:  , :-1] += temp[1:  ,1:  ]*(ds/6)*(dz/3) # +(+1, 0)
        bp[ :-1, :-1] += temp[ :-1,1:  ]*(ds/6)*(dz/3)
        bp[1:  ,1:  ] += temp[1:  , :-1]*(ds/6)*(dz/3) # +(-1, 0)
        bp[ :-1,1:  ] += temp[ :-1, :-1]*(ds/6)*(dz/3)
        bp[ :-1,1:  ] += temp[1:  ,1:  ]*(ds/3)*(dz/6) # +( 0,+1)
        bp[ :-1, :-1] += temp[1:  , :-1]*(ds/3)*(dz/6)
        bp[1:  ,1:  ] += temp[ :-1,1:  ]*(ds/3)*(dz/6) # +( 0,-1)
        bp[1:  , :-1] += temp[ :-1, :-1]*(ds/3)*(dz/6)
        bp[ :-1, :-1] += temp[1:  ,1:  ]*(ds/6)*(dz/6) # +(+1,+1)
        bp[ :-1,1:  ] += temp[1:  , :-1]*(ds/6)*(dz/6) # +(-1,+1)
        bp[1:  , :-1] += temp[ :-1,1:  ]*(ds/6)*(dz/6) # +(+1,-1)
        bp[1:  ,1:  ] += temp[ :-1, :-1]*(ds/6)*(dz/6) # +(-1,-1)
        # Boundary integrals (RHS)
        # z=0, n=(0,-1), grad(p).n = -dp/dz
        temp = -(2*w[ 0,:]-5*w[ 1,:]+4*w[ 2,:]-w[ 3,:])/dz**2
        if eps!=0:
            temp *= _R[ 0,:] # integrating factor
        bp[ 0,1:  ] -= temp[1:  ]*(ds/3)
        bp[ 0, :-1] -= temp[ :-1]*(ds/3)
        bp[ 0, :-1] -= temp[1:  ]*(ds/6)
        bp[ 0,1:  ] -= temp[ :-1]*(ds/6)
        # z=1, n=(0,+1), grad(p).n = +dp/dz
        temp = +(2*w[-1,:]-5*w[-2,:]+4*w[-3,:]-w[-4,:])/dz**2
        if eps!=0:
            temp *= _R[-1,:] # integrating factor
        bp[-1,1:  ] -= temp[1:  ]*(ds/3)
        bp[-1, :-1] -= temp[ :-1]*(ds/3)
        bp[-1, :-1] -= temp[1:  ]*(ds/6)
        bp[-1,1:  ] -= temp[ :-1]*(ds/6)
        # s=0, n=(-1,0), grad(p).n = -dp/ds
        temp = -(2*v[:, 0]-5*v[:, 1]+4*v[:, 2]-v[:, 3])/ds**2
        if eps!=0:
            temp += -eps/_R[:, 0]*dvds[:, 0]
            temp *= _R[:, 0] # integrating factor
        bp[1:  , 0] -= temp[1:  ]*(dz/3)
        bp[ :-1, 0] -= temp[ :-1]*(dz/3)
        bp[ :-1, 0] -= temp[1:  ]*(dz/6)
        bp[1:  , 0] -= temp[ :-1]*(dz/6)
        # s=1, n=(+1,0), grad(p).n = +dp/ds
        temp = +(2*v[:,-1]-5*v[:,-2]+4*v[:,-3]-v[:,-4])/ds**2
        if eps!=0:
            temp += +eps/_R[:,-1]*dvds[:,-1]
            temp *= _R[:,-1] # integrating factor
        bp[1:  ,-1] -= temp[1:  ]*(dz/3)
        bp[ :-1,-1] -= temp[ :-1]*(dz/3)
        bp[ :-1,-1] -= temp[1:  ]*(dz/6)
        bp[1:  ,-1] -= temp[ :-1]*(dz/6)

        # Construct the sparse matrix
        Ap = diags([Ap_p0_p0.ravel(),\
                    Ap_p1_p0.ravel()[:-1],Ap_m1_p0.ravel()[1:],\
                    Ap_p0_p1.ravel()[:-m],Ap_p0_m1.ravel()[m:],\
                    Ap_p1_p1.ravel()[:-1-m],Ap_m1_p1.ravel()[:-m+1],\
                    Ap_p1_m1.ravel()[m-1:],Ap_m1_m1.ravel()[m+1:]],\
                   [0,1,-1,m,-m,1+m,-1+m,1-m,-1-m],format='lil')
                   
        # Modify one row to force the mean (via the integral) to be zero
        k = m//2+m*(n//2) # We pick a point in the centre
        temp = np.zeros(u.shape)
        temp[1:  ,1:  ] += 0.25*ds*dz
        temp[ :-1,1:  ] += 0.25*ds*dz
        temp[1:  , :-1] += 0.25*ds*dz
        temp[ :-1, :-1] += 0.25*ds*dz
        if True:
            # Replace existing row
            Ap[k,:] = temp.ravel()
            bp.ravel()[k] = 0
        else:
            # Add to existing row
            Ap[k,:] += temp.ravel()
                          
        # Solve the pressure Laplacian
        p = spsolve(Ap.tocsr(),bp.ravel()).reshape(u.shape)
        
        return p
    
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
