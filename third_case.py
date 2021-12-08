import sys
scr_path = '/home/hcwang96/foreground_project'
sys.path.insert(1, scr_path)

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import scipy.signal as signal
import vis
import h5py
from numpy.linalg import inv
import scipy as sp
import PS_estimator #IMPORT THE PS ESTIMATOR CODE 
import os
import time
import pickle
from scipy import linalg as LA
import telescope
import estimation

if sys.argv[1] == "1e-3":
    gf = 10
    mg = sys.argv[1]
elif sys.argv[1] == "1e-4":
    gf = 1
    mg = sys.argv[1]
elif sys.argv[1] == "1e-5":
    gf = 0.1
    mg = sys.argv[1]
else:
    print("Wrong argument. An argument is necessary and can be only one of the following: 1e-3, 1e-4, or 1e-5!")


# We need this function to set up the baseline for an Nant by Nant dish array
def config_baseline(Nant, dx, lamb):
    #Nant: number of antenna per side
    #dx: antenna separation
    #lamb:wavelength
    
    u = np.arange(-(Nant-1), Nant)[np.newaxis, :]*dx/lamb[:, np.newaxis]
    v = np.arange(-(Nant-1), Nant)[np.newaxis, :]*dx/lamb[:, np.newaxis]
    u_vec = []
    v_vec = []
    for i in range(len(lamb)):
        U, V = np.meshgrid(u[i], v[i])
        u_vec.append(U.ravel())
        v_vec.append(V.ravel())
    u_vec = np.vstack(u_vec)
    v_vec = np.vstack(v_vec)

    #Mask to cutoff redundant baseline + autocorrelation
    mask = np.logical_or(v_vec[0]>0., np.logical_and(v_vec[0]==0., u_vec[0]>0.)) 
    
    return u_vec, v_vec, mask

fmin = 400. #min freq in MHz
fmax = 500. #max freq in MHz
nfreq = 50 #No. of freq channels
freq = np.linspace(fmin, fmax, nfreq)
wavelength =3.e8/(freq*1e6) #wavelength in m
xreso = 0.2*np.pi/180. #T map pixel reso in radian
nxpix = 150
dx = np.deg2rad(0.2)  #radian/pix
Nant = 5 # No. of antennas per side #Cautions when changing baseline parameters!
db = 7. #baseline separation in m

u_vec, v_vec, mask = config_baseline(Nant, db, wavelength)
ngood = len(np.where(mask == True)[0])
ngood_unst = int((Nant*Nant)*(Nant*Nant-1)/2) # how many elements there are in a half unstacked visibility
R_vec = np.sqrt(u_vec[:, mask]**2+v_vec[:, mask]**2)
vis_instance = vis.vis(u_vec[:, mask], v_vec[:, mask], nxpix, nxpix, xreso, xreso)

# Function to cutoff redundant baseline + autocorrelation --> get the half visibility from the full visibility
def cutoff(v, mask):
    return v[:, mask]

# need this to plot visibilities
def mat(v, nfreq):
    if np.ndim(v) !=1:
        v = v.flatten()
    ngood = int(len(v)/nfreq)
    nside = int( np.sqrt(2*ngood+1))
    m = np.zeros([nfreq, nside, nside], dtype=np.complex_)
    v = np.reshape(v, (nfreq, ngood))
    for f in range(nfreq):
        m[f, ...] = np.reshape(np.append(np.append(np.flip(v[f, ...].conj()), np.NaN), v[f, ...]), (nside, nside))    
    return m

# Convert a stacked half visibility to an unstack half visibility
# arguments: Nant: number of antenna on each side, nfreq: number of frequencies, vis: stacked half visibility
def st_to_unst(Nant, nfreq, vis):
    if nfreq != vis.shape[0]:
        print("The number of frequencies is wrong!")
    if ((2*Nant-1)*(2*Nant-1)-1)/2 != vis.shape[1]:
        print("The format of the visibilities is wrong!")
    Vij = np.zeros((nfreq, Nant*Nant, Nant*Nant), dtype = complex)
    for f in range(0, nfreq):
        for ind in range(0, vis.shape[1]):
            # find the uv coordinate of each visibility
            if ind < Nant-1:
                v = 0 # v is the row difference
                u = ind + 1 # u is the column difference
            else:
                v = 1 + int((ind-(Nant-1))/(2*Nant-1))
                m = np.mod(ind-(Nant-1), 2*Nant-1)
                u = -(Nant-1) + m
            # x and y are the indices of two antennae (coordinates of the unstacked space)
            for x in range(0, Nant*Nant):
                z = np.mod(x, Nant)
                if (x + v*Nant < Nant*Nant) and (z + u >= 0) and (z + u < Nant):
                    y = x + v*Nant + u
                    Vij[f, x, y] = vis[f, ind]
    # flattern the upper half of Vij
    vis_ij = np.zeros((nfreq, int((Nant*Nant)*(Nant*Nant-1)/2)), dtype = complex)
    for f in range(0, nfreq):
        ind = 0
        for x in range(0, Nant*Nant-1):
            for y in range(x+1, Nant*Nant):
                vis_ij[f, ind] = Vij[f, x, y]
                ind = ind+1
    return vis_ij

# Convert an unstacked half visibility to a stacked half visibility
# arguments: Nant: number of antenna on each side, nfreq: number of frequencies, unst_vis: unstacked half visibility
def unst_to_st(Nant, nfreq, unst_vis):
    st_vis = np.zeros((nfreq, int(((Nant*2-1)*(Nant*2-1)-1)/2)), dtype = complex)
    n = np.zeros((nfreq, int(((Nant*2-1)*(Nant*2-1)-1)/2)))
    for f in range(0, nfreq):
        ind = 0
        for x in range(0, Nant*Nant-1):
            for y in range(x+1, Nant*Nant):
                i = int(x/Nant)
                j = np.mod(x, Nant)
                l = int(y/Nant)
                m = np.mod(y, Nant)
                v = l - i
                u = m - j
                ind_st = (Nant*2-1)*v + u - 1
                st_vis[f, ind_st] += unst_vis[f, ind]
                n[f, ind_st] += 1
                ind = ind + 1
    st_vis = st_vis / n
    return st_vis

# Convert a half unstacked visibility into a full unstacked visibility (square array) for plotting

def full_Vij(Nant, nfreq, vis_ij):
    V = np.zeros((nfreq, Nant*Nant, Nant*Nant), dtype = complex)
    for f in range(0, nfreq):
        ind = 0 
        for x in range(0, Nant*Nant):
            for y in range(x, Nant*Nant):
                if x == y:
                    V[f, x, y] = np.NaN
                else:
                    V[f, x, y] = vis_ij[f, ind]
                    V[f, y, x] = np.conjugate(V[f, x, y])
                    ind = ind+1
    return V

sigma1 = 1e-4*gf # gain
sigma2 = 1e-6*gf # delay
sigma3 = 1e-4*gf # gain statistical
sigma4 = 1e-5*gf # delay statistical

class Get_err:
    def __init__(self, Nant, freq, sigma1, sigma2, sigma3, sigma4, mask):
        """ Assign a frequency-dependent error to each antenna. The error of each antenna is a complex number. 
            The magnitude of the error is frequency-independent. It is 1 plus a small number drawn from a Gaussin 
            centered at 0 with std = sigma1. The phase of the error is frequency-dependent. It is a small number
            drawn from a Gaussian centered at 0 with std = sigma2 then multipled by 2*pi*frequency. The errors are 
            stored in an Nfreq*Nant*Nant array called gerr. For example, the error of the antenna on the nth row, mth
            column at the kth frequency is gerr[k, n, m]."""
        self.freq = freq # vector that carries all frequencies
        self.Nf = len(self.freq) # number of frequencies
        self.Nant = Nant # number of antennae on each side
        g = 1 + np.random.normal(0, sigma1, (self.Nant, self.Nant)) # magnitude of error
        t = np.random.normal(0, sigma2, (self.Nant, self.Nant)) # later will be multiplied by frequencies to give phase errors
        g_freq = np.ones((self.Nf, Nant, Nant)) * g # make a copy with each frequency
        g_freq = g_freq + np.random.normal(0, sigma3, (self.Nf, Nant, Nant)) # get the statistical gain error
        t_freq = np.ones((self.Nf, Nant, Nant)) * t # make a copy with each frequency
        t_freq = freq[:, np.newaxis, np.newaxis]*t_freq # get frequency dependence in the phase error
        t_freq = t_freq + np.random.normal(0, sigma4, (self.Nf, Nant, Nant)) # get the statistical phase error
        gerr = np.zeros((self.Nf, Nant, Nant), dtype = complex)
        gerr = g_freq * np.exp(t_freq*2*np.pi*1j) # combine the amplitude and phase to get the conplex error
        self.g = gerr # we will call gerr g
        self.mask = mask
        
        """ This function assigns error to half visibilities in the stacked space. It considers each antenna pair and 
            sums the error from each pair to the total error of each baseline. n counts the redundancy of each 
            baseline. err stores the summed error of each baseline at each frequency. There are 
            (2*Nant-1) * (2*Nant-1) baselines.
        """
        n = np.zeros((2*self.Nant-1, 2*self.Nant-1))
        err = np.zeros((self.Nf, 2*self.Nant-1, 2*self.Nant-1), dtype = complex)
        for i in range(self.Nant):
            for k in range(self.Nant):
                for l in range(self.Nant):
                    for m in range(self.Nant): # considering the antenna pair (i,k) and (l,m)
                        v = l - i # row coordinates of the baseline
                        u = m - k # column coordinate of the baseline
                        v_i = v + self.Nant -1 # to avoid negative indices
                        u_i = u + self.Nant -1 # to avoid negative indices 
                        n[v_i, u_i] += 1
                        err[:, v_i, u_i] += self.g[:, i, k]*np.conjugate(self.g[:, l, m])
        err = err/n # calculate the average error
        err_vec = [] # err_vec stores errors in the same format of the input visibilities 
        for f_i in range(self.Nf):
            err_vec.append(err[f_i].ravel())
        err_vec = np.vstack(err_vec)
        self.err_vec = cutoff(err_vec, self.mask)
     
    def return_err(self):
        return self.err_vec_unst, self.g
    
    def gain_err_uv(self, vis): 
        return self.err_vec*vis

    def gain_err_ij(self, vis):
        """ This function assigns the same error to half visibilities in the unstacked space.
        """
        vis_err = np.zeros((self.Nf, int((self.Nant*self.Nant)*(self.Nant*self.Nant-1)/2)), dtype = complex)
        self.err_vec_unst = np.zeros_like(vis_err)
        for f in range(0, self.Nf):
            ind = 0
            for x in range(0, (self.Nant)*(self.Nant)-1):
                for y in range(x+1, (self.Nant)*(self.Nant)):
                    i = int(x/self.Nant)
                    j = np.mod(x, self.Nant)
                    l = int(y/self.Nant)
                    m = np.mod(y, self.Nant)
                    vis_err[f, ind] = vis[f, ind]*self.g[f, i, j]*np.conjugate(self.g[f, l, m])
                    self.err_vec_unst[f, ind] =  self.g[f, i, j]*np.conjugate(self.g[f, l, m])
                    ind = ind + 1
        return vis_err  
    
sigmabp = 1e-4*gf

class Get_bp_err:
    def __init__(self, freq, sigma):
        self.freq = freq # vector that carries all frequencies
        self.Nf = len(self.freq) # number of frequencies
        h = np.random.normal(0, sigma, self.Nf) # band-pass error assigned to all antennae at each frequency
        self.g_plus_one = (1 + h)**2 # the err that will be multiplied to visibilities comes from the product of two antennae
        
    def bp_err(self, vis):
        # input visibilities without error; returns visibilties with error
        return self.g_plus_one[:, np.newaxis]*vis # multiply the band-pass error with visibilities
    
    def plot_err(self):
        plt.figure(figsize = (30, 16))

        plt.subplot(121)
        plt.plot(self.g_plus_one - 1)
        plt.title("True error")
        plt.xlabel("Frequency channel")
        plt.ylabel("Value")
        
nreal = 10000
read_cov = True #if directly read cov file in vis space, otherwise generating
covfile = '/home/juan/foregrounds/maps/visibility_data_and_cov.hdf5' #The sky component cube in vis space if exists
f = h5py.File(covfile, 'r')

t_reduce = 6 # 6 times less integration time

S = f['HI_covariance'][...]
F = f['FOREGROUND_covariance'][...]
N = f['NOISE_covariance'][...]*t_reduce

C_deriv = f['HI_deriv_covariance'][...]
bc = f['bin_center'][...]
nkbin = len(bc)

f.close()

C_tot = S + F + N
C_tot_inv = inv(C_tot)

def KLT_1step(S, F, threshold):
    
    R_val, R= sp.linalg.eigh(F, S) 
    Rd = R.T.conj()

    #Rejecting bad modes
    Rd_inv = np.linalg.inv(Rd) #inverse of Rd
    i_threshold = np.searchsorted(R_val, threshold) #found threshold index that have eigenvalue < threshold
    #print('The threshold is ', threshold)
    #print('Rejecting n>', i_threshold, ' modes')
    Rd_inv_good = Rd_inv[:, :i_threshold] #select colums has eigenvalue < threshold
    Rd_good = Rd[:i_threshold] #select rows has eigenvalue < threshold
    filter_matrix = Rd_inv_good.dot(Rd_good)

    #plt.semilogy(np.abs(R_val))
        
    return filter_matrix, R, R_val, Rd_good

S1 = S
F1 = F
threshold = 0.01

filter1, R1, R1_val, Rd_good = KLT_1step(S, F, threshold) #Recover S from S+F

K = filter1

with open('/home/hcwang96/foreground_project/K_unst.pickle', 'rb') as f:
    filter2 = pickle.load(f)
    
filepath_test = '/home/hcwang96/foreground_project/Realistic_sims_50freq_150x150pix_30x30deg_10patches.hdf5'
f = h5py.File(filepath_test, 'r')

# get the sky map components
HI_map = f['HI'][...]
FREE_map = f['FREE'][...]
PS_map = f['PS'][...]
SYN_map = f['SYN'][...]
FG_map = FREE_map + PS_map + SYN_map

data_handler = h5py.File('/home/juan/foregrounds/maps/archive/visibilties_1000_realizations.hdf5', 'r')
Npoi = data_handler.attrs['nxpix'] #Number of points in x and y
delta_x = data_handler.attrs['xreso'] # pixel resolution in radians (x and y)
f_MHz = data_handler.attrs['nu']
Nant_side = data_handler.attrs['Nant'] #Number of antennas in x and y
delta_b_m = data_handler.attrs['db'] # Baseline in meters
D = data_handler.attrs['Dape'] # Telescope aperture size in meters
window = data_handler.attrs['window'] # Window to use for the telescope aperture
t_int = data_handler.attrs['tint']/t_reduce # Integration time in seconds
T_sys = data_handler.attrs['Tsys'] # System temperature in Kelvin
tel = telescope.telescope(Nant_side, delta_b_m, Npoi, delta_x, f_MHz, t_int, T_sys, D, window)

Nant = Nant_side
nfreq = len(f_MHz)
freq = f_MHz
delta_f_MHz = f_MHz[1] - f_MHz[0]
ndata = 10

vis_noise_rms = T_sys/np.sqrt(t_int * (delta_f_MHz*1e6))
N_vis = (vis_noise_rms/np.sqrt(2.)) * (
                np.random.randn(nfreq, ngood_unst, ndata) + 1j*np.random.randn(nfreq, ngood_unst, ndata))
N_vis2 =(vis_noise_rms/np.sqrt(2.)) * (
                np.random.randn(nfreq, ngood_unst, ndata) + 1j*np.random.randn(nfreq, ngood_unst, ndata))

V_HI = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128') # true HI visibilities
V_N = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128') # noise
V_F = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128') # true foreground visibilities

errorbp = Get_bp_err(freq, sigmabp) #call a band-pass error instance
error = Get_err(Nant, freq, sigma1, sigma2, sigma3, sigma4, mask)
V_HI_err = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128') # HI visibilities with error
V_FG_err = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128') # foreground visibilities with error

for i in range(ndata):
    #Read in data
    V_HI[..., i] = st_to_unst(Nant, nfreq, tel.visibilities.get_visibilities(HI_map[..., i])) #real HI data in unstacked vis space
    V_HI_err[..., i] = errorbp.bp_err(error.gain_err_ij(V_HI[..., i])) # HI data in vis space with error
    V_F[..., i] = st_to_unst(Nant, nfreq, tel.visibilities.get_visibilities(FG_map[..., i]))
    V_FG_err[..., i] = errorbp.bp_err(error.gain_err_ij(V_F[..., i]))
    V_N[..., i] = N_vis[..., i]

V_tot = V_HI + V_F + V_N
V_d = V_HI_err + V_FG_err + V_N


V_HI_hat = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128') 
V_FG_hat = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128') 

for i in range(ndata):
    V_HI_hat[..., i] = np.reshape(filter2.dot(V_d[..., i].flatten()), (nfreq, ngood_unst)) # estimated signal which is mostly FG residuals
    V_FG_hat[..., i] = V_d[..., i] - V_HI_hat[..., i] # estimated foreground
    
def generate_E1_one_freq(i):
    E1 = np.zeros((ngood_unst, ngood_unst))
    start = ((Nant**2-1)+(Nant**2-i))*i/2
    for j in range(i+1, Nant*Nant):
        ind = int(start + j - i - 1)
        E1[ind, ind] = 1
    return E1

def generate_E2_one_freq(i):
    E2 = np.zeros((ngood_unst, ngood_unst))
    for j in range(0, i):
        start = ((Nant**2-1)+(Nant**2-j))*j/2
        ind = int(start + i - j - 1)
        E2[ind, ind] = 1
    return E2

y_hat = np.zeros((nfreq, Nant*Nant, ndata), dtype = 'complex')
for d in range(ndata):
    for nu_ind in range(nfreq):
        for i in range(Nant*Nant):
            E1 = generate_E1_one_freq(i)
            E2 = generate_E2_one_freq(i)
        
            numerator = 0
            norm = 0
        
            #if i < Nant*Nant - 1:
            norm1 = np.vdot(V_FG_hat[nu_ind, :, d], np.dot(E1, V_FG_hat[nu_ind, :, d])) # normalization in the denominator
            numerator1 = np.vdot(V_HI_hat[nu_ind, :, d], np.dot(E1, V_FG_hat[nu_ind, :, d])) # cross correlate the estimated signal with estimated FG
            numerator += numerator1
            norm += norm1
            
            #if i > 0:
            norm2 = np.vdot(V_FG_hat[nu_ind, :, d], np.dot(E2, V_FG_hat[nu_ind, :, d]))
            numerator2 = np.vdot(V_FG_hat[nu_ind, :, d], np.dot(E2, V_HI_hat[nu_ind, :, d]))
            numerator += numerator2
            norm += norm2
            
            y_hat[nu_ind, i, d] = numerator/norm
            
with open('/home/hcwang96/foreground_project/W1_ind.pickle', 'rb') as f:
    W1 = pickle.load(f)
with open('/home/hcwang96/foreground_project/W2_ind.pickle', 'rb') as f:
    W2 = pickle.load(f)
    
W_combined = np.zeros((2500, 2500), dtype = np.complex_)
for i in range(0, 1250):
    W_combined[i,0:1250] = W1[i, :]
    W_combined[i, 1250:] = W2[i, :]
for i in range(1250, 2500):
    W_combined[i,0:1250] = np.conjugate(W2[i - 1250, :])
    W_combined[i, 1250:] = np.conjugate(W1[i - 1250, :])
    
U, s, Vh = LA.svd(W_combined)

with open('/home/hcwang96/foreground_project/pinv_cutoff_017.pickle', 'rb') as f:
    pinv = pickle.load(f)
    
y_hat_combined = np.zeros((nfreq*Nant*Nant*2, ndata), dtype = 'complex')

for d in range(ndata):
    y_hat_combined[0:nfreq*Nant*Nant, d] = y_hat[..., d].flatten()
    y_hat_combined[nfreq*Nant*Nant:nfreq*Nant*Nant*2, d] = np.conjugate(y_hat[..., d].flatten())
    
y_recovered = np.zeros((nfreq*Nant*Nant*2, ndata), dtype = 'complex')
y_use = np.zeros((nfreq, Nant*Nant, ndata), dtype = 'complex')

for d in range(ndata):
    y_recovered[:, d] = np.dot(pinv, y_hat_combined[:, d])
    y_use[..., d] = np.reshape(y_recovered[1250:, d], (50, 25))
    
y_hat_vis = np.zeros((nfreq, ngood_unst, ndata), dtype = 'complex')
for d in range(ndata):
    for nu in range(nfreq):
        ind = 0
        for i in range(0, Nant**2 - 1):
            for j in range(i+1, Nant**2):
                y_hat_vis[nu, ind, d] = y_use[nu, i, d] + np.conjugate(y_use[nu, j, d])
                ind += 1
                
V_HI_tilde = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128')
V_HI_tilde_st = np.zeros([nfreq, ngood, ndata], dtype = 'complex128')

for d in range(ndata):
    gV_FG_hat = y_hat_vis[..., d]*V_FG_hat[..., d]
    V_HI_tilde[..., d] = V_HI_hat[..., d] - np.reshape(filter2.dot(gV_FG_hat.flatten()), (nfreq, ngood_unst))
    V_HI_tilde_st[..., d] = unst_to_st(Nant, nfreq, V_HI_tilde[..., d])
    
V_d2 = V_d - N_vis + N_vis2
V_HI_hat2 = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128')
V_FG_hat2 = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128')

'''
for i in range(ndata):
    #Note here that i change N each realisation so that data2 has different N that data, but HI and F are the same
    # simply use (i+1)th noise for ith data
    if i == ndata-1:
        V_d2[..., i] = (V_HI_err[..., i] + V_FG_err[..., i] + V_N[..., 0])
    else:
        V_d2[..., i] = (V_HI_err[..., i] + V_FG_err[..., i] + V_N[..., i+1])
'''

for i in range(ndata):
    V_HI_hat2[..., i] = np.reshape(filter2.dot(V_d2[..., i].flatten()), (nfreq, ngood_unst)) # estimated signal which is mostly FG residuals
    V_FG_hat2[..., i] = V_d2[..., i] - V_HI_hat2[..., i] # estimated foreground

y_hat2 = np.zeros((nfreq, Nant*Nant, ndata), dtype = 'complex')
for d in range(ndata):
    for nu_ind in range(nfreq):
        for i in range(Nant*Nant):
            E1 = generate_E1_one_freq(i)
            E2 = generate_E2_one_freq(i)
        
            numerator = 0
            norm = 0
        
            #if i < Nant*Nant - 1:
            norm1 = np.vdot(V_FG_hat2[nu_ind, :, d], np.dot(E1, V_FG_hat2[nu_ind, :, d])) # normalization in the denominator
            numerator1 = np.vdot(V_HI_hat2[nu_ind, :, d], np.dot(E1, V_FG_hat2[nu_ind, :, d])) # cross correlate the estimated signal with estimated FG
            numerator += numerator1
            norm += norm1
            
            #if i > 0:
            norm2 = np.vdot(V_FG_hat2[nu_ind, :, d], np.dot(E2, V_FG_hat2[nu_ind, :, d]))
            numerator2 = np.vdot(V_FG_hat2[nu_ind, :, d], np.dot(E2, V_HI_hat2[nu_ind, :, d]))
            numerator += numerator2
            norm += norm2
            
            y_hat2[nu_ind, i, d] = numerator/norm
            
            

y_hat_combined2 = np.zeros((nfreq*Nant*Nant*2, ndata), dtype = 'complex')
for d in range(ndata):
    y_hat_combined2[0:nfreq*Nant*Nant, d] = y_hat2[..., d].flatten()
    y_hat_combined2[nfreq*Nant*Nant:nfreq*Nant*Nant*2, d] = np.conjugate(y_hat2[..., d].flatten())
    
y_recovered2 = np.zeros((nfreq*Nant*Nant*2, ndata), dtype = 'complex')
y_use2 = np.zeros((nfreq, Nant*Nant, ndata), dtype = 'complex')
for d in range(ndata):
    y_recovered2[:, d] = np.dot(pinv, y_hat_combined2[:, d])
    y_use2[..., d] = np.reshape(y_recovered2[1250:, d], (50, 25))

y_hat_vis2 = np.zeros((nfreq, ngood_unst, ndata), dtype = 'complex')
for d in range(ndata):
    for nu in range(nfreq):
        ind = 0
        for i in range(0, Nant**2 - 1):
            for j in range(i+1, Nant**2):
                y_hat_vis2[nu, ind, d] = y_use2[nu, i, d] + np.conjugate(y_use2[nu, j, d])
                ind += 1

V_HI_tilde2 = np.zeros([nfreq, ngood_unst, ndata], dtype = 'complex128')
V_HI_tilde_st2 = np.zeros([nfreq, ngood, ndata], dtype = 'complex128')

for d in range(ndata):
    gV_FG_hat2 = y_hat_vis2[..., d]*V_FG_hat2[..., d]
    V_HI_tilde2[..., d] = V_HI_hat2[..., d] - np.reshape(filter2.dot(gV_FG_hat2.flatten()), (nfreq, ngood_unst))
    V_HI_tilde_st2[..., d] = unst_to_st(Nant, nfreq, V_HI_tilde2[..., d])
    
#APPLY KLT
klt_C_tot = Rd_good.dot(C_tot).dot(Rd_good.T.conj()) #apply klt to total covariance matrix
inv_klt_C_tot = inv(klt_C_tot) #inverse covariance matrix

klt_C_deriv = [] #apply klt to derivative matrix
for k in range(nkbin):
    klt_C_deriv.append(Rd_good.dot(C_deriv[k, ...]).dot(Rd_good.T.conj()))
klt_C_deriv = np.array(klt_C_deriv)

truecl = '/home/hcwang96/foreground_project/HI_Cl.hdf5'# file name for the realistic power spectrum
Fcl = h5py.File(truecl, 'r')
ell = Fcl['ell'][...]
cl = np.mean(Fcl['cl'][...], axis = 0)

est_ps = estimation.quadratic_estimator(klt_C_tot, klt_C_deriv)

data1 = np.zeros((Rd_good.shape[0], ndata), dtype = 'complex128')
data2 = np.zeros((Rd_good.shape[0], ndata), dtype = 'complex128')
for i in range(ndata):
    data1[..., i] = Rd_good.dot(V_HI_tilde_st[..., i].flatten()) #apply klt to data
    data2[..., i] = Rd_good.dot(V_HI_tilde_st2[..., i].flatten()) #apply klt to data

C_ell_hat = est_ps.estimate_params(data1, data2)
C_ell_hat_mean = np.mean(C_ell_hat, axis=1)
C_ell_hat_std = np.std(C_ell_hat, axis=1)
C_ell_std = est_ps.expected_params_std()

V_d_st = np.zeros([nfreq, ngood, ndata], dtype = 'complex128')
V_d_st2 = np.zeros([nfreq, ngood, ndata], dtype = 'complex128')

for d in range(ndata):
    V_d_st[..., d] = unst_to_st(Nant, nfreq, V_d[..., d])
    V_d_st2[..., d] = unst_to_st(Nant, nfreq, V_d2[..., d])
    
data1_unc = np.zeros((Rd_good.shape[0], ndata), dtype = 'complex128')
data2_unc = np.zeros((Rd_good.shape[0], ndata), dtype = 'complex128')
for i in range(ndata):
    data1_unc[..., i] = np.dot(Rd_good, V_d_st[..., i].flatten()) #apply klt to data
    data2_unc[..., i] = np.dot(Rd_good, V_d_st2[..., i].flatten()) #apply klt to data


C_ell_hat_unc = est_ps.estimate_params(data1_unc, data2_unc)
C_ell_hat_mean_unc = np.mean(C_ell_hat_unc, axis=1)
C_ell_hat_std_unc = np.std(C_ell_hat_unc, axis=1)
C_ell_std_unc = est_ps.expected_params_std()

file1 = 'test_third_cleaned_mean_%s' % sys.argv[1]
file2 = 'test_third_cleaned_std_%s' % sys.argv[1]
file3 = 'test_third_uncleaned_mean_%s' % sys.argv[1]
file4 = 'test_third_uncleaned_std_%s' % sys.argv[1]

with open(file1, 'wb') as f:
    pickle.dump(C_ell_hat_mean.real, f)
with open(file2, 'wb') as f:
    pickle.dump(C_ell_hat_std, f)
with open(file3, 'wb') as f:
    pickle.dump(C_ell_hat_mean_unc.real, f)
with open(file4, 'wb') as f:
    pickle.dump(C_ell_hat_std_unc, f)