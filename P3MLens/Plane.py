import numpy as np
from numba import jit,prange,set_num_threads
from scipy.special import j0,j1
from scipy.spatial import cKDTree
from astropy.cosmology import Planck18 as cosmo
from multiprocessing import Pool
from itertools import repeat

class Plane:
    """ Lens Plane construct from input particles
    
    This class constructs a lens plane from 2D positions of particals 
    and calculates deflection angles and gravitational parameters for 
    any positions in this plane using P3M algorithm with optimized 
    Green function and adaptive soften length. 
    
    Parameters:
    -----------
    coor: ndarray of shape (n_particles, 2)
        [x,y] coordinates of particles in the unit of kpc/h. x and y 
        should be in the range of 0 < x,y < box.
    box: even int
        Physical length of the Plane in kpc/h. Should be even for FFT.
    m_p: float or ndarray of shape (n_particles,)
        Mass of each particle in 10^6 Msun/h. If float, mass is the 
        same for all particles.
    H: float, default=1.
        Physical length for each grid in kpc/h. The number of grids is 
        simply (box/H)^2.
    p: int, default=2
        Mass assignment and force intepolation scheme. 1 for CIC, 2 for 
        TSC and 3 for PCS.
    a: float, default=6.
        The soften length in PM: a_pm = a*H.
    fftw: bool, default=True
        If True, using pyfftw for FFT, which can be paralleled. If False,
        using numpy for FFT.
    green: ndarray of shape (box/H, box/H), default=None
        Green function used to solve Poisson's equation. If None, 
        optimized Green function is calculated automatically. If you're 
        building a lot of Plane with the same parameters (box, H, p, a), 
        you're recommanded to calculate and save the optimized Green func-
        tion using Plane.Green function and input it directly.
    core: int, default=5
        Core number used for parallel.
        
    Attributes:
    ------------
    density_map: ndarray of shape (box/H, box/H)
        Surface density for each grid after mass assignment with the 
        unit 10^6 h Msun/kpc^2.
    PM_field_grid: ndarray of shape (2, box/H, box/H)
        PM force grid used for force intepolation with the unit (km/s)^2.
        PM_field_grid[0] for the force of x direction and PM_field_grid[1]
        for the y direction.
    """
    def __init__(self,coor,box,m_p,H=1,p=2,a=6,fftw=True,green=None,core=5):
        self._box = box
        m_p = np.atleast_1d(m_p)
        if len(m_p) == 1:
            self._m_p = np.ones(len(coor))*m_p
        else:
            self._m_p = m_p
        self._H = H
        self._p = p
        self._a = a
        self._core = core
        self._set_numba_threads(core)
        self._coor = coor
        self._fftw = fftw
        self._tree = cKDTree(self._coor,leafsize=400,boxsize=self._box)
        self._green = green
        self.density_map = self._paint(self._coor,self._box,self._m_p,self._H,self._p)
        self.PM_field_grid = self._PM_grid()
    
    def __del__(self):
        pass
    
    def _set_numba_threads(self,core):
        set_num_threads(core)
        
    def _paint(self,coor,box,m_p,H,p):
        coor = coor / H
        box = int(round(box / H))
        x = coor[:,0]
        y = coor[:,1]
        if p == 1:
            number = self._paint_cic(box,x,y,m_p)
        if p == 2:
            number = self._paint_tsc(box,x,y,m_p)
        if p == 3:
            number = self._paint_PCS(box,x,y,m_p)
        return number / H**2
    
    @staticmethod
    @jit(nopython=True)#, parallel=True)
    def _paint_cic(box,x,y,m_p):
        lense = box
        xgrid = np.zeros((box,box))
        for i in prange(len(x)):
            cx = np.int64(np.ceil(x[i]))
            cy = np.int64(np.ceil(y[i]))
            fx = cx - 1
            fy = cy - 1
            cx_w = 1 - (cx - x[i])
            cy_w = 1 - (cy - y[i])
            fx_w = 1 - (x[i] - fx)
            fy_w = 1 - (y[i] - fy)
            xgrid[cy%lense,cx%lense] += cy_w*cx_w*m_p[i]
            xgrid[cy%lense,fx%lense] += cy_w*fx_w*m_p[i]
            xgrid[fy%lense,cx%lense] += fy_w*cx_w*m_p[i]
            xgrid[fy%lense,fx%lense] += fy_w*fx_w*m_p[i]
        return xgrid

    @staticmethod
    @jit(nopython=True)#, parallel=True)
    def _paint_tsc(box,x,y,m_p):
        lense = box 
        xgrid = np.zeros((lense,lense))
        for i in prange(len(x)):
            cx = np.int64(np.ceil(x[i]))
            cy = np.int64(np.ceil(y[i]))
            fx = cx - 1
            fy = cy - 1
            if cx - x[i] < 0.5:
                ax = cx + 1
                cx_w = 0.75 - (cx - x[i])**2
                ax_w = 0.5 * (1.5 - ax + x[i])**2
                fx_w = 0.5 * (1.5 - x[i] + fx)**2
            else:
                ax = fx - 1
                cx_w = 0.5 * (1.5 - cx + x[i])**2
                ax_w = 0.5 * (1.5 - x[i] + ax)**2
                fx_w = 0.75 - (x[i] - fx)**2
            if cy - y[i] < 0.5:
                ay = cy + 1
                cy_w = 0.75 - (cy - y[i])**2
                ay_w = 0.5 * (1.5 - ay + y[i])**2
                fy_w = 0.5 * (1.5 - y[i] + fy)**2
            else:
                ay = fy - 1
                cy_w = 0.5 * (1.5 - cy + y[i])**2
                ay_w = 0.5 * (1.5 - y[i] + ay)**2
                fy_w = 0.75 - (y[i] - fy)**2
            xgrid[cy%lense,cx%lense] += cy_w*cx_w*m_p[i]
            xgrid[cy%lense,fx%lense] += cy_w*fx_w*m_p[i]
            xgrid[fy%lense,cx%lense] += fy_w*cx_w*m_p[i]
            xgrid[fy%lense,fx%lense] += fy_w*fx_w*m_p[i]
            xgrid[cy%lense,ax%lense] += cy_w*ax_w*m_p[i]
            xgrid[fy%lense,ax%lense] += fy_w*ax_w*m_p[i]
            xgrid[ay%lense,cx%lense] += ay_w*cx_w*m_p[i]
            xgrid[ay%lense,fx%lense] += ay_w*fx_w*m_p[i]     
            xgrid[ay%lense,ax%lense] += ay_w*ax_w*m_p[i]
        return xgrid
    
    @staticmethod
    @jit(nopython=True)
    def _paint_PCS(box,x,y):
        lense = box 
        xgrid = np.zeros((lense,lense))
        for i in prange(len(x)):
            cx = np.int64(np.ceil(x[i]))
            cy = np.int64(np.ceil(y[i]))
            fx = cx - 1
            fy = cy - 1
            acx = cx + 1
            acy = cy + 1
            afx = fx - 1
            afy = fy - 1
            cx_w = 1./6*(4.-6*(cx-x[i])**2+3.*(cx-x[i])**3)
            cy_w = 1./6*(4.-6*(cy-y[i])**2+3.*(cy-y[i])**3)
            fx_w = 1./6*(4.-6*(fx-x[i])**2+3.*(x[i]-fx)**3)
            fy_w = 1./6*(4.-6*(fy-y[i])**2+3.*(y[i]-fy)**3)
            acx_w = 1./6*(2-(acx-x[i]))**3
            acy_w = 1./6*(2-(acy-y[i]))**3
            afx_w = 1./6*(2-(x[i]-afx))**3
            afy_w = 1./6*(2-(y[i]-afy))**3
            xgrid[cy%lense,cx%lense] += cy_w*cx_w*m_p[i]
            xgrid[cy%lense,fx%lense] += cy_w*fx_w*m_p[i]
            xgrid[cy%lense,acx%lense] += cy_w*acx_w*m_p[i]
            xgrid[cy%lense,afx%lense] += cy_w*afx_w*m_p[i]
            xgrid[fy%lense,cx%lense] += fy_w*cx_w*m_p[i]
            xgrid[fy%lense,fx%lense] += fy_w*fx_w*m_p[i]
            xgrid[fy%lense,acx%lense] += fy_w*acx_w*m_p[i]
            xgrid[fy%lense,afx%lense] += fy_w*afx_w*m_p[i]
            xgrid[acy%lense,cx%lense] += acy_w*cx_w*m_p[i]
            xgrid[acy%lense,fx%lense] += acy_w*fx_w*m_p[i]
            xgrid[acy%lense,acx%lense] += acy_w*acx_w*m_p[i]
            xgrid[acy%lense,afx%lense] += acy_w*afx_w*m_p[i]
            xgrid[afy%lense,cx%lense] += afy_w*cx_w*m_p[i]
            xgrid[afy%lense,fx%lense] += afy_w*fx_w*m_p[i]
            xgrid[afy%lense,acx%lense] += afy_w*acx_w*m_p[i]
            xgrid[afy%lense,afx%lense] += afy_w*afx_w*m_p[i]
        return xgrid
    
    @staticmethod
    @jit(nopython=True)#,parallel=True)
    def _differece(potential,alpha,H): #alpha prefer 4/3
        
        # difference
        
        f1y = np.zeros(potential.shape)
        f1y[1:-1] = (potential[2:] - potential[:-2]) / (2. * H)
        f1y[0] = (potential[1] - potential[0]) / H
        f1y[-1] = (potential[-2] - potential[-1]) / H
        f1x = np.zeros(potential.shape)
        f1x[:,1:-1] = (potential[:,2:] - potential[:,:-2]) / (2. * H)
        f1x[:,0] = (potential[:,1] - potential[:,0]) / H
        f1x[:,-1] = (potential[:,-2] - potential[:,-1]) / H
        f2y = np.zeros(potential.shape)
        f2y[2:-2] = (potential[4:] - potential[:-4]) / (4. * H)
        f2y[0] = (potential[2] - potential[0]) / (2. * H)
        f2y[1] = (potential[3] - potential[0]) / (3. * H)
        f2y[-1] = (potential[-3] - potential[-1]) / (2. * H)
        f2y[-2] = (potential[-4] - potential[-1]) / (3. * H)
        f2x = np.zeros(potential.shape)
        f2x[:,2:-2] = (potential[:,4:] - potential[:,:-4]) / (4. * H)
        f2x[:,0] = (potential[:,2] - potential[:,0]) / (2. * H)
        f2x[:,1] = (potential[:,3] - potential[:,0]) / (3. * H)
        f2x[:,-1] = (potential[:,-3] - potential[:,-1]) / (2. * H)
        f2x[:,-2] = (potential[:,-4] - potential[:,-1]) / (3. * H)
        return alpha * np.stack((f1x,f1y)) + (1. - alpha) * np.stack((f2x,f2y))
    
    def _PM_grid(self): 
        
        # calculate force on grid
        
        if self._green is None:
            gk, kx, ky = Green(self._box, self._H, self._p, self._a, self._core)
        else:
            gk = self._green
        if self._fftw == False:
            sigmak = np.fft.fft2(self.density_map)
            phik = sigmak * gk
            phik[0,0] = 0
            phi = np.fft.ifft2(phik)
            phi = phi.real
            field = -1.*self._differece(phi,4./3.,self._H) # (km/s)^ 2
        else:
            import pyfftw 
            density_pfw = pyfftw.empty_aligned(gk.shape, dtype='complex128', n=16)
            density_pfw = self.density_map + 1j*0.0
            sigmak = pyfftw.interfaces.numpy_fft.fft2(density_pfw, threads=self._core)
            phik = sigmak * gk
            phik[0,0] = 0
            phi = pyfftw.interfaces.numpy_fft.ifft2(phik, threads=self._core)
            phi = phi.real
            field = -1.*self._differece(phi,4./3.,self._H) # (km/s)^ 2
        return field
    
    def PM_field(self,x,y):
        """
        
        PM force field for required positions
        
        Parameters:
        -----------
        x: ndarray of any shape
            x coordinates of required positions. 
        y: ndarray of any shape
            y coordinates of required positions.
            
        Returns:
        -----------
        f: ndarray of shape (2, x.shape[0], x.shape[1])
            x and y direction PM force field for required 
            positions in (km/s)^2.
        """
        return self.__interpolate_PM_field(self.PM_field_grid,x,y,self._p,self._H)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def __interpolate_PM_field(PM_field_grid, x, y, p, H):
        
        #interpolate grid force to whole space
        
        xt = x / H
        yt = y / H
        forcex = PM_field_grid[0]
        lense = forcex.shape[0]
        forcey = PM_field_grid[1]
        xp = xt.reshape(xt.size)
        yp = yt.reshape(yt.size)
        force_interx = np.zeros(xp.shape)
        force_intery = np.zeros(xp.shape)
        for i in prange(len(force_interx)):
            cx = np.int64(np.ceil(xp[i]))
            cy = np.int64(np.ceil(yp[i]))
            fx = cx - 1
            fy = cy - 1
            if p == 1:
                cx_w = 1 - (cx - xp[i])
                cy_w = 1 - (cy - yp[i])
                fx_w = 1 - (xp[i] - fx)
                fy_w = 1 - (yp[i] - fy)
                force_interx[i] = forcex[cy%lense,cx%lense]*cy_w*cx_w + forcex[cy%lense,fx%lense]*cy_w*fx_w + forcex[fy%lense,cx%lense]*fy_w*cx_w + forcex[fy%lense,fx%lense]*fy_w*fx_w
                force_intery[i] = forcey[cy%lense,cx%lense]*cy_w*cx_w + forcey[cy%lense,fx%lense]*cy_w*fx_w + forcey[fy%lense,cx%lense]*fy_w*cx_w + forcey[fy%lense,fx%lense]*fy_w*fx_w
            if p == 2:
                if cx - xp[i] < 0.5:
                    ax = cx + 1
                    cx_w = 0.75 - (cx - xp[i])**2
                    ax_w = 0.5 * (1.5 - ax + xp[i])**2
                    fx_w = 0.5 * (1.5 - xp[i] + fx)**2
                else:
                    ax = fx - 1
                    cx_w = 0.5 * (1.5 - cx + xp[i])**2
                    ax_w = 0.5 * (1.5 - xp[i] + ax)**2
                    fx_w = 0.75 - (xp[i] - fx)**2
                if cy - yp[i] < 0.5:
                    ay = cy + 1
                    cy_w = 0.75 - (cy - yp[i])**2
                    ay_w = 0.5 * (1.5 - ay + yp[i])**2
                    fy_w = 0.5 * (1.5 - yp[i] + fy)**2
                else:
                    ay = fy - 1
                    cy_w = 0.5 * (1.5 - cy + yp[i])**2
                    ay_w = 0.5 * (1.5 - yp[i] + ay)**2
                    fy_w = 0.75 - (yp[i] - fy)**2
                force_interx[i] = forcex[cy%lense,cx%lense]*cy_w*cx_w + forcex[cy%lense,fx%lense]*cy_w*fx_w +\
forcex[fy%lense,cx%lense]*fy_w*cx_w + forcex[fy%lense,fx%lense]*fy_w*fx_w + forcex[cy%lense,ax%lense]*cy_w*ax_w +\
    forcex[fy%lense,ax%lense]*fy_w*ax_w + forcex[ay%lense,cx%lense]*ay_w*cx_w + forcex[ay%lense,fx%lense]*ay_w*fx_w +\
    forcex[ay%lense,ax%lense]*ay_w*ax_w
                force_intery[i] = forcey[cy%lense,cx%lense]*cy_w*cx_w + forcey[cy%lense,fx%lense]*cy_w*fx_w +\
    forcey[fy%lense,cx%lense]*fy_w*cx_w + forcey[fy%lense,fx%lense]*fy_w*fx_w + forcey[cy%lense,ax%lense]*cy_w*ax_w +\
    forcey[fy%lense,ax%lense]*fy_w*ax_w + forcey[ay%lense,cx%lense]*ay_w*cx_w + forcey[ay%lense,fx%lense]*ay_w*fx_w +\
    forcey[ay%lense,ax%lense]*ay_w*ax_w
            if p == 3:
                acx = cx + 1
                acy = cy + 1
                afx = fx - 1
                afy = fy - 1
                cx_w = 1./6*(4.-6*(cx-xp[i])**2+3.*(cx-xp[i])**3)
                cy_w = 1./6*(4.-6*(cy-yp[i])**2+3.*(cy-yp[i])**3)
                fx_w = 1./6*(4.-6*(fx-xp[i])**2+3.*(xp[i]-fx)**3)
                fy_w = 1./6*(4.-6*(fy-yp[i])**2+3.*(yp[i]-fy)**3)
                acx_w = 1./6*(2-(acx-xp[i]))**3
                acy_w = 1./6*(2-(acy-yp[i]))**3
                afx_w = 1./6*(2-(xp[i]-afx))**3
                afy_w = 1./6*(2-(yp[i]-afy))**3
                force_interx[i] = forcex[cy%lense,cx%lense]*cy_w*cx_w + forcex[cy%lense,fx%lense]*cy_w*fx_w +\
    forcex[cy%lense,acx%lense]*cy_w*acx_w + forcex[cy%lense,afx%lense]*cy_w*afx_w + forcex[fy%lense,cx%lense]*fy_w*cx_w + forcex[fy%lense,fx%lense]*fy_w*fx_w +\
    forcex[fy%lense,acx%lense]*fy_w*acx_w + forcex[fy%lense,afx%lense]*fy_w*afx_w + forcex[acy%lense,cx%lense]*acy_w*cx_w + forcex[acy%lense,fx%lense]*acy_w*fx_w +\
    forcex[acy%lense,acx%lense]*acy_w*acx_w + forcex[acy%lense,afx%lense]*acy_w*afx_w + forcex[afy%lense,cx%lense]*afy_w*cx_w + forcex[afy%lense,fx%lense]*afy_w*fx_w +\
    forcex[afy%lense,acx%lense]*afy_w*acx_w + forcex[afy%lense,afx%lense]*afy_w*afx_w
                force_intery[i] = forcey[cy%lense,cx%lense]*cy_w*cx_w + forcey[cy%lense,fx%lense]*cy_w*fx_w +\
    forcey[cy%lense,acx%lense]*cy_w*acx_w + forcey[cy%lense,afx%lense]*cy_w*afx_w + forcey[fy%lense,cx%lense]*fy_w*cx_w + forcey[fy%lense,fx%lense]*fy_w*fx_w +\
    forcey[fy%lense,acx%lense]*fy_w*acx_w + forcey[fy%lense,afx%lense]*fy_w*afx_w + forcey[acy%lense,cx%lense]*acy_w*cx_w + forcey[acy%lense,fx%lense]*acy_w*fx_w +\
    forcey[acy%lense,acx%lense]*acy_w*acx_w + forcey[acy%lense,afx%lense]*acy_w*afx_w + forcey[afy%lense,cx%lense]*afy_w*cx_w + forcey[afy%lense,fx%lense]*afy_w*fx_w +\
    forcey[afy%lense,acx%lense]*afy_w*acx_w + forcey[afy%lense,afx%lense]*afy_w*afx_w
        return np.stack((force_interx.reshape(x.shape),force_intery.reshape(y.shape)))
    
    def PP_field(self,x,y,N=800):
        """
        
        PP force field for required positions
        
        Parameters:
        -----------
        x: ndarray of any shape
            x coordinates of required positions. 
        y: ndarray of any shape
            y coordinates of required positions.
        N: int, default=800
            Number of particles used in adaptive soften length.
        Returns:
        -----------
        f: ndarray of shape (2, x.shape[0], x.shape[1])
            x and y direction PP force field for required positions 
            in (km/s)^2.
        """
        
        @jit(nopython=True)
        def get_index(count):
            index = np.zeros(count.size + 1,dtype=np.int64)
            index[0] = 0
            for i in range(len(count)):
                index[i+1] = index[i] + count[i]
            return index
        
        @jit(nopython=True)
        def PM_f1(x,a):
                ep = 2.*x/a
                return 1./a*(7.43080530e-01*ep**4-1.83299236e+00*ep**3-5.71160351e-02*ep**2+2.67270709e+00*ep-8.24463263e-05)
            
        @jit(nopython=True)  
        def PM_f2(x,a):
                ep = 2.*x/a
                return 1./a*(1.53996716/ep-6.8231916+15.10702097*ep-11.85624512*ep**2+4.08123043*ep**3-0.52410421*ep**4)
            
        @jit(nopython=True)
        def f_pm(x,a):
            f = np.zeros(x.shape)
            f = np.where(x<a/2.,PM_f1(x,a),PM_f2(x,a))
            f = np.where(x>a,1./x,f)
            return f
        
        
        @jit(nopython=True, parallel=True)
        def PP(coor_inter1,coor_inter2,coor_part,ind1,ind2,index,m_p,am,ap1,ap2,box):
            l1 = len(coor_inter1)
            l2 = len(coor_inter2)
            PP_fx = np.zeros(l1+l2)
            PP_fy = np.zeros(l1+l2)
            for i in prange(l1+l2):
                if i < l2:
                    coor_p = coor_part[ind2[index[i]:index[i+1]]]
                    m = m_p[ind2[index[i]:index[i+1]]]
                    displace = coor_p - coor_inter2[i]
                    distance = np.sqrt(np.sum(displace**2,axis=1))
                    displace = np.transpose(displace)
                    part = displace / distance
                    f = 8.60183454013995*m*(f_pm(distance,ap2[i]) - f_pm(distance,am))*part
                    fi = np.sum(f,axis=1)
                    PP_fx[i] = fi[0]
                    PP_fy[i] = fi[1]
                else:
                    coor_p = coor_part[ind1[i-l2]]
                    m = m_p[ind1[i-l2]]
                    displace = coor_p - coor_inter1[i-l2]
                    displace = np.where(displace>box/2.,displace-box,displace)
                    displace = np.where(displace<-1*box/2,displace+box,displace)
                    distance = np.sqrt(np.sum(displace**2,axis=1))
                    displace = np.transpose(displace)
                    part = displace / distance
                    f = 8.60183454013995*m*(f_pm(distance,ap1[i-l2]) - f_pm(distance,am))*part
                    fi = np.sum(f,axis=1)
                    PP_fx[i] = fi[0]
                    PP_fy[i] = fi[1]
            return PP_fx,PP_fy
        
        @jit(nopython=True, parallel=True)
        def PP_point(coor_inter,coor_part,ind,index,m_p,a,count):
            PP_fx = np.zeros(len(index)-1)
            PP_fy = np.zeros(len(index)-1)
            for i in prange(len(index)-1):
                if index[i]==index[i+1]:
                    continue
                else:
                    coor_p = coor_part[ind[index[i]:index[i+1]]]
                    m = m_p[ind[index[i]:index[i+1]]]
                    displace = coor_p - coor_inter[i]
                    distance = np.sqrt(np.sum(displace**2,axis=1))
                    displace = np.transpose(displace)
                    part = displace / distance
                    f = 8.60183454013995*m*(1/distance - f_pm(distance,a))*part
                    fi = np.sum(f,axis=1)
                    PP_fx[i] = fi[0]
                    PP_fy[i] = fi[1]
            return PP_fx,PP_fy
        
        xp = x.reshape(x.size)
        yp = y.reshape(y.size)
        xp = xp%self._box
        yp = yp%self._box
        coor_inter = np.array([xp,yp]).T
        if N != 0:
            dis_neigh,neigh = self._tree.query(coor_inter, k=N, workers=self._core)
            dis_neigh = dis_neigh[:,-1]
            j = dis_neigh<(self._a*self._H)
            nj = ~j
            coor_inter1 = coor_inter[nj]
            coor_inter2 = coor_inter[j]
            dis_neigh1 = dis_neigh[nj]
            dis_neigh2 = dis_neigh[j]
            ind1 = neigh[nj]
            if len(coor_inter2) != 0:
                ind2 = self._tree.query_ball_point(coor_inter2,r=self._a*self._H,workers=self._core)
                arr_len = np.frompyfunc(len,1,1)
                count2 = arr_len(ind2).astype(int)
                ind2 = np.hstack(ind2)
            else:
                count2 = np.zeros(0,dtype=int)
                ind2 = np.zeros(0,dtype=int)
            index = get_index(count2)
            ind1 = ind1.astype(int)
            ind2 = ind2.astype(int)
            PP_fx_t, PP_fy_t = PP(coor_inter1,coor_inter2,self._coor,ind1,ind2,index,self._m_p,self._a*self._H,dis_neigh1,dis_neigh2,float(self._box))
            PP_fx = np.zeros(PP_fx_t.shape)
            PP_fx[j] = PP_fx_t[0:len(dis_neigh2)]
            PP_fx[nj] = PP_fx_t[len(dis_neigh2):]
            PP_fy = np.zeros(PP_fy_t.shape)
            PP_fy[j] = PP_fy_t[0:len(dis_neigh2)]
            PP_fy[nj] = PP_fy_t[len(dis_neigh2):]
        else:
            ind = self._tree.query_ball_point(coor_inter,r=self._a*self._H,workers=self._core)
            arr_len = np.frompyfunc(len,1,1)
            count = arr_len(ind).astype(int)
            ind = np.hstack(ind)
            ind = ind.astype(int)
            index = get_index(count)
            PP_fx, PP_fy = PP_point(coor_inter,self._coor,ind,index,self._m_p,self._a*self._H,count)
        return np.stack((PP_fx.reshape(x.shape),PP_fy.reshape(y.shape)))
    
    def total_field(self,x,y,PP=True,N=800):
        """
        
        Total force field for required positions.
        
        Parameters:
        -----------
        x: ndarray of any shape
            x coordinates of required positions. 
        y: ndarray of any shape
            y coordinates of required positions.
        PP: bool, default=True
            If False, only performing PM.
        N: int, default=800
            Number of particles used in adaptive soften length of PP.
        Returns:
        -----------
        f: ndarray of shape (2, x.shape[0], x.shape[1])
            x and y direction total force field for required positions 
            in (km/s)^2.
        """
        if PP==True:
            return self.PM_field(x, y) + self.PP_field(x,y,N)
        else:
            return self.PM_field(x, y)
    
    def deflection_angle(self,x,y,PP=True,N=800):
        """
        Deflection angles for required positions.
        
        Parameters:
        -----------
        x: ndarray of any shape
            x coordinates of required positions. 
        y: ndarray of any shape
            y coordinates of required positions.
        PP: bool, default=True
            If False, only performing PM.
        N: int, default=800
            Number of particles used in adaptive soften length of PP.
        Returns:
        -----------
        f: ndarray of shape (2, x.shape[0], x.shape[1])
            x and y direction deflection angles for required positions 
            in radian.
        """
        return self.total_field(x,y,PP,N)*(-2)/(3e5)**2 # rad
    
    @staticmethod
    @jit(nopython=True,parallel=True)
    def _lens(angle_mx,angle_px,angle_my,angle_py,d,H,zl,zs,offset,Ds,Dl,Dls):
        
        # for Function lense_parameter
        
        angle_dx = (angle_px-angle_mx)/(2.*d*H)
        angle_dy = (angle_py-angle_my)/(2.*d*H)
        convergence = 0.5*(angle_dx[0]+angle_dy[1])
        convergence += offset
        shear1 = 0.5*(angle_dx[0]-angle_dy[1])
        shear2 = 0.5*(angle_dx[1]+angle_dy[0])
        scale = Dls*Dl/Ds
        convergence *= scale
        shear1 *= scale
        shear2 *= scale
        magnification = 1./((1.-convergence)**2-shear1**2-shear2**2)
        return np.stack((convergence,shear1,shear2,magnification))
    
    
    def lense_parameter(self,x,y,d=0.05,PP=True,N=800,zl=0.5,zs=1.0,cosmo=cosmo):
        """
        Lensing parameters for required positions. Should be used only 
        for single plane problems.
        
        Parameters:
        -----------
        x: ndarray of any shape
            x coordinates of required positions. 
        y: ndarray of any shape
            y coordinates of required positions.
        d: float, default=0.05
            Difference step d*H used to calculate lensing parameters. Defle-
            ction angles at x+d*H, x-d*H, y+d*H and y-d*H are calculated
            to derive lensing parameters at (x, y).
        PP: bool, default=True
            If False, only performing PM.
        N: int, default=800
            Number of particles used in adaptive soften length of PP.
        zl: float, default=0.5
            Redshift of the lens plane.
        zs: float, default=1.0
            Redshift of the source plane.
        cosmo: astropy.cosmology, default=Planck18
            Cosmology used to calculate angular diameter distances.
            
        Returns:
        -----------
        parameters: ndarray of shape (4, x.shape[0], x.shape[1])
            [convergence,shear1,shear2,magnification] for required 
            positions.
        """
        Ds = cosmo.angular_diameter_distance(zs).value*1000.*cosmo.h
        Dl = cosmo.angular_diameter_distance(zl).value*1000.*cosmo.h
        Dls = cosmo.angular_diameter_distance_z1z2(zl, zs).value*1000.*cosmo.h
        angle_mx = self.deflection_angle((x-d*self._H),y,PP,N)
        angle_px = self.deflection_angle((x+d*self._H),y,PP,N)
        angle_my = self.deflection_angle(x,(y-d*self._H),PP,N)
        angle_py = self.deflection_angle(x,(y+d*self._H),PP,N)
        offset = np.sum(self._m_p)/self._box**2*4.*np.pi*4.300917270069975/(3e5)**2
        return self._lens(angle_mx,angle_px,angle_my,angle_py,d,self._H,zl,zs,offset,Ds,Dl,Dls)



#Green function   
    
def green(kx,ky,H=1,p=2,a=6.,alpha=4./3.,n=1):
    def sr(k,a):
            result = np.where(k==0,1.,128./(k**3*a**3)*j1(k*a/2.)-32./(k**2*a**2)*j0(k*a/2.))
            return result

    def R(kx,ky,a):
        k = np.sqrt(kx**2+ky**2)
        if a != 0:
            s = sr(k,a)
        else:
            s = 1.
        return np.stack((-1j*kx*s**2/k**2,-1j*ky*s**2/k**2))*4.3009173*4*np.pi  #kpc^2 * (km/s)**2 / 1e6 M_sun

    @jit(nopython=True)
    def u(kx,ky,H,p):
        result = (4.*np.sin(kx*H/2.)*np.sin(ky*H/2.)/(kx*ky*H**2))**(p+1)
        result = np.where(kx==0,(2.*np.sin(ky*H/2.)/(ky*H))**(p+1),result)
        result = np.where(ky==0,(2.*np.sin(kx*H/2.)/(kx*H))**(p+1),result)
        result = np.where((kx==0)&(ky==0),1.,result)
        return result

    @jit(nopython=True)
    def u2n_2(kx,ky,H):#only for p=2
        return (1-np.sin(kx*H/2.)**2+2./15.*np.sin(kx*H/2.)**4)*(1-np.sin(ky*H/2.)**2+2./15.*np.sin(ky*H/2.)**4)

    @jit(nopython=True)
    def u2n(kx,ky,H,p,n):
        result = np.zeros(kx.shape)
        kg = 2.*np.pi/H
        for ix in range(-n,n+1):
            for iy in range(-n,n+1):
                result += u(kx-ix*kg,ky-iy*kg,H,p)**2
        return result

    @jit(nopython=True)
    def d(kx,ky,alpha,H):
        dx = 1j*alpha*np.sin(kx*H)/H+1j*(1-alpha)*np.sin(2.*kx*H)/(2.*H)
        dy = 1j*alpha*np.sin(ky*H)/H+1j*(1-alpha)*np.sin(2.*ky*H)/(2.*H)
        return np.stack((dx,dy))

    def ur_cn(n,kx,ky,H,p,a):
        result = 0.0
        kg = 2.*np.pi/H
        for ix in range(-n,n+1):
            for iy in range(-n,n+1):
                result += (u(kx-ix*kg,ky-iy*kg,H,p)**2*np.conj(R(kx-ix*kg,ky-iy*kg,a)))
        return result

    if p == 2:
        u2_n = u2n_2(kx,ky,H)
    else:
        u2_n = u2n(kx,ky,H,p,n)
    D = d(kx,ky,alpha,H)
    result = np.sum(D*ur_cn(n,kx,ky,H,p,a),axis=0)/(np.sum(np.abs(D)**2,axis=0)*u2_n**2)
    return result


def Green(box,H,p=2,a=6.,core=5):
    """2D optimized green function in Fourier space.
    
    Parameters:
    -----------
    box: even int
        Physical length of the Plane in kpc/h. Should be even for FFT.
    H: float, default=1.
        Physical length for each grid in kpc/h. The number of grids is 
        simply (box/H)^2.
    p: int, default=2
        Mass assignment and force intepolation scheme. 1 for CIC, 2 for 
        TSC and 3 for PCS.
    a: float, default=6.
        The soften length in PM: a_pm = a*H.
    core: int, default=5
        Core number used for parallel.
    
    Returns:
    ------------
    G: ndarray of shape (box/H, box/H)
        Optimized green function in Fourier space G(kx,ky). The sequence 
        of kx and ky is the same as FFT, which is returned by numpy.fft.
        fftfreq.
    kx: ndarray of shape (box/H, box/H)
        kx for each G.
    ky: ndarray of shape (box/H, box/H)
        ky for each G.
    """
    a *= H
    length = int(round(box / H))
    kxy = np.fft.fftfreq(length,H)*2*np.pi
    gky,gkx = np.meshgrid(kxy,kxy)
    kx = np.array_split(gkx,core)
    ky = np.array_split(gky,core)
    with Pool(core) as pool:
        result = pool.starmap(green,zip(kx,ky,repeat(H),repeat(p),repeat(a)))
    return np.vstack(result), gkx, gky
