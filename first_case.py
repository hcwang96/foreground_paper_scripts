import sys
scr_path = '/home/hcwang96/foreground_project'
sys.path.insert(1, scr_path)
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import scipy.signal as signal
import h5py
from numpy.linalg import inv
import scipy as sp
import PS_estimator #IMPORT THE PS ESTIMATOR CODE 
import os
import time
import pickle
import vis, beam
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

sigma = 1e-4*gf

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
covfile = '/home/juan/foregrounds/maps/visibility_data_and_cov.hdf5'#The sky component cube in vis space if exists
f = h5py.File(covfile, 'r')

t_reduce = 6 # 6 times less integration time; we reduced integration time from 2 years to 120 days

S = f['HI_covariance'][...]
F = f['FOREGROUND_covariance'][...]
N = f['NOISE_covariance'][...]*t_reduce

C_deriv = f['HI_deriv_covariance'][...]
bc = f['bin_center'][...]
nkbin = len(bc)

f.close()

C_tot = S + F + N
C_tot_inv = inv(C_tot)
bc = np.array(bc)

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

nfreq = 50
ng = 40 # number of baselines in the stacked half visibility, namely ngood_st
W = np.zeros([nfreq, nfreq], dtype = 'complex')

I = np.eye(nfreq*ng)
ImK = I - K
Kda = np.matrix.getH(K)
ImKda = I - Kda
Ctot = S + F
norm = np.dot(ImK, np.dot(Ctot, ImKda)) # the denominator before multiplying E_\nu

for nu in range(0, nfreq):
    denominator = np.trace(norm[nu*ng:(nu+1)*ng, nu*ng:(nu+1)*ng]) # the denominator multiplied with E_\nu
    for nup in range(0, nfreq):
        num1 = 0 # num1 is small and can be neglected

        #A = np.dot(ImK[nu*ng:(nu+1)*ng, nup*ng: (nup+1)*ng], Ctot[nup*ng:(nup+1)*ng, :])
        #for i in range(0, ng):
        #    num1 += np.dot(A[i,:], Kda[:, nu*ng + i])

        num2 = 0
        B = np.dot(ImK[nu*ng: (nu+1)*ng, :], Ctot[:, nup*ng:(nup+1)*ng])
        C = Kda[nup*ng:(nup+1)*ng, nu*ng: (nu+1)*ng]
        #num2 = np.trace(np.dot(B, C))
        for i in range(0, ng):
            num2 += np.dot(B[i,:], C[:, i])
        
        W[nu, nup] = (num1 + num2)/denominator
        
testlist = '/home/hcwang96/foreground_project/Realistic_sims_50freq_150x150pix_30x30deg_10patches.hdf5'
f = h5py.File(testlist, 'r')

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


N_vis= data_handler['Thermal_cube'][...]*(t_reduce**0.5)

ndata = 10

V_HI = np.zeros([nfreq, ngood, ndata], dtype = 'complex128') # true HI visibilities
V_N = np.zeros([nfreq, ngood, ndata], dtype = 'complex128') # noise
V_F = np.zeros([nfreq, ngood, ndata], dtype = 'complex128') # true foreground visibilities

error = Get_bp_err(freq, sigma) #call an error instance
V_HI_err = np.zeros([nfreq, ngood, ndata], dtype = 'complex128') # HI visibilities with error
V_FG_err = np.zeros([nfreq, ngood, ndata], dtype = 'complex128') # foreground visibilities with error

for i in range(ndata):
    #Read in data
    V_HI[..., i] = tel.visibilities.get_visibilities(HI_map[..., i]) #real HI data in vis space
    V_HI_err[..., i] = error.bp_err(V_HI[..., i]) # HI data in vis space with error
    V_F[..., i] = tel.visibilities.get_visibilities(FG_map[..., i])
    V_FG_err[..., i] = error.bp_err(V_F[..., i])
    V_N[..., i] = N_vis[..., i]

V_tot = V_HI + V_F + V_N
V_d = V_HI_err + V_FG_err + V_N

V_HI_hat = np.zeros([nfreq, ngood, ndata], dtype = 'complex128') 
V_FG_hat = np.zeros([nfreq, ngood, ndata], dtype = 'complex128') 

for i in range(ndata):
    V_HI_hat[..., i] = np.reshape(filter1.dot(V_d[..., i].flatten()), (nfreq, ngood)) # estimated signal which is mostly FG residuals
    V_FG_hat[..., i] = V_d[..., i] - V_HI_hat[..., i] # estimated foreground

y_hat = np.zeros([nfreq, ndata], dtype = 'complex128')
for i in range(ndata):
    for nu_ind in range(nfreq):
        norm = np.vdot(V_FG_hat[nu_ind, :, i], V_FG_hat[nu_ind, :, i]) # normalization in the denominator
        numerator = np.vdot(V_HI_hat[nu_ind, :, i], V_FG_hat[nu_ind, :, i]) # cross correlate the estimated signal with estimated FG
        y_hat[nu_ind, i] = numerator/norm
        
from scipy import linalg as LA
U, s, Vh = LA.svd(W)
#plt.figure(figsize = (18,5))
#plt.plot(s, ".")
#plt.title("Singular value spectrum of W")

pseudo_inv, rank = LA.pinv(W, rcond = 0.2, return_rank = True)

g_hat = np.zeros([nfreq, ndata], dtype = 'complex128')

for i in range(ndata):
    g_hat[..., i] = np.dot(pseudo_inv, y_hat[..., i]).real
    
g_actual = error.g_plus_one - 1

if gf == 10:
    with open('g_actual', 'wb') as f:
        pickle.dump(g_actual, f)
    with open('y_hat', 'wb') as f:
        pickle.dump(y_hat[..., 0].real, f)
    with open('g_hat', 'wb') as f:
        pickle.dump(np.dot(W, g_actual).real, f)

V_HI_tilde = np.zeros([nfreq, ngood, ndata], dtype = 'complex128')

for i in range(ndata):
    g_hat_i = g_hat[..., i]
    gV_FG_hat = g_hat_i[..., np.newaxis].real*V_FG_hat[..., i]
    V_HI_tilde[..., i] = V_HI_hat[..., i] - np.reshape(filter1.dot(gV_FG_hat.flatten()), (nfreq, ngood))
    
V_d2 = np.zeros([nfreq, ngood, ndata], dtype = 'complex128')
V_HI_hat2 = np.zeros([nfreq, ngood, ndata], dtype = 'complex128')
V_FG_hat2 = np.zeros([nfreq, ngood, ndata], dtype = 'complex128')
y_hat2 = np.zeros([nfreq, ndata], dtype = 'complex128')
g_hat2 = np.zeros([nfreq, ndata], dtype = 'complex128')
V_HI_tilde2 = np.zeros([nfreq, ngood, ndata], dtype = 'complex128')

for i in range(ndata):
    #Note here that i change N each realisation so that data2 has different N that data, but HI and F are the same
    # simply use (i+1)th noise for ith data
    if i == ndata-1:
        V_d2[..., i] = (V_HI_err[..., i] + V_FG_err[..., i] + V_N[..., 0])
    else:
        V_d2[..., i] = (V_HI_err[..., i] + V_FG_err[..., i] + V_N[..., i+1])
    
    V_HI_hat2[..., i] = np.reshape(filter1.dot(V_d2[..., i].flatten()), (nfreq, ngood)) # estimated signal which is mostly FG residuals
    V_FG_hat2[..., i] = V_d2[..., i] - V_HI_hat2[..., i] # estimated foreground
    
    for nu_ind in range(nfreq):
        norm = np.vdot(V_FG_hat2[nu_ind, :, i], V_FG_hat2[nu_ind, :, i]) # normalization in the denominator
        numerator = np.vdot(V_HI_hat2[nu_ind, :, i], V_FG_hat2[nu_ind, :, i]) # cross correlate the estimated signal with estimated FG
        y_hat2[nu_ind, i] = numerator/norm
    
    g_hat2[..., i] = np.dot(pseudo_inv, y_hat2[..., i]).real
    g_hat_i2 = g_hat2[..., i]
    gV_FG_hat2 = g_hat_i2[..., np.newaxis].real*V_FG_hat2[..., i]
    V_HI_tilde2[..., i] = V_HI_hat2[..., i] - np.reshape(filter1.dot(gV_FG_hat2.flatten()), (nfreq, ngood))    
    
freq_plot = 10
dpick = 2

V_HI_plot = np.reshape(filter1.dot(V_HI[..., dpick].flatten()), (nfreq, ngood))
V_KL_only = np.reshape(filter1.dot(V_d[..., dpick].flatten()), (nfreq, ngood))

#V_HI_stacked = unst_to_st(Nant, nfreq, np.reshape(filter2.dot(V_HI[..., dpick].flatten()), (nfreq, ngood_unst)))
#V_d_stacked = unst_to_st(Nant, nfreq, np.reshape(filter2.dot(V_d[..., dpick].flatten()), (nfreq, ngood_unst)))


V_mat1 = mat(V_HI_plot, nfreq)
V_mat2 = mat(V_HI_tilde[..., dpick], nfreq)
V_mat3 = mat(V_KL_only[...], nfreq) #mat(V_d[..., dpick], nfreq)
V_fg = mat(V_F[..., dpick], nfreq)


if gf == 10:
    with open('true_HI', 'wb') as f:
        pickle.dump(V_mat1[freq_plot].real, f)
    with open('true_FG', 'wb') as f:
        pickle.dump(V_fg[freq_plot].real, f)
    with open('cleaned_HI', 'wb') as f:
        pickle.dump(V_mat2[freq_plot].real, f)
    with open('uncleaned_HI', 'wb') as f:
        pickle.dump(V_mat3[freq_plot].real, f)

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
    data1[..., i] = Rd_good.dot(V_HI_tilde[..., i].flatten()) #apply klt to data
    data2[..., i] = Rd_good.dot(V_HI_tilde2[..., i].flatten()) #apply klt to data

C_ell_hat = est_ps.estimate_params(data1, data2)
C_ell_hat_mean = np.mean(C_ell_hat, axis=1)
C_ell_hat_std = np.std(C_ell_hat, axis=1)
C_ell_std = est_ps.expected_params_std()

data1_unc = np.zeros((Rd_good.shape[0], ndata), dtype = 'complex128')
data2_unc = np.zeros((Rd_good.shape[0], ndata), dtype = 'complex128')
for i in range(ndata):
    data1_unc[..., i] = np.dot(Rd_good, V_d[..., i].flatten()) #apply klt to data
    data2_unc[..., i] = np.dot(Rd_good, V_d2[..., i].flatten()) #apply klt to data


C_ell_hat_unc = est_ps.estimate_params(data1_unc, data2_unc)
C_ell_hat_mean_unc = np.mean(C_ell_hat_unc, axis=1)
C_ell_hat_std_unc = np.std(C_ell_hat_unc, axis=1)
C_ell_std_unc = est_ps.expected_params_std()

file1 = 'test_first_cleaned_mean_%s' % sys.argv[1]
file2 = 'test_first_cleaned_std_%s' % sys.argv[1]
file3 = 'test_first_uncleaned_mean_%s' % sys.argv[1]
file4 = 'test_first_uncleaned_std_%s' % sys.argv[1]

with open(file1, 'wb') as f:
    pickle.dump(C_ell_hat_mean.real, f)
with open(file2, 'wb') as f:
    pickle.dump(C_ell_hat_std, f)
with open(file3, 'wb') as f:
    pickle.dump(C_ell_hat_mean_unc.real, f)
with open(file4, 'wb') as f:
    pickle.dump(C_ell_hat_std_unc, f)
with open('test_bc_new', 'wb') as f:
    pickle.dump(np.array(bc), f)





