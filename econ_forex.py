import torch
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

dtype = torch.float32
device = "mps" if torch.backends.mps.is_available() else "cpu"

# %% Classes for the forex economies
class baseline():
    """
    Classes for forex economies without frictions
    """
    def __init__(self, betas, cQ, theta_H, theta_F, theta_psi_H, theta_psi_F):
        self.cQ                    = cQ
        self.n_state               = cQ.size()[0]
        self.mu_H, self.mu_F       = torch.tensor([0.03, 0.03], device=device, dtype=dtype)
        self.delta_H, self.delta_F = torch.tensor([0.03, 0.03], device=device, dtype=dtype)
        self.beta_H, self.beta_F   = betas#torch.tensor([0.9, 0.9], device=device, dtype=dtype)

        self.theta_H       = theta_H
        self.theta_F       = theta_F
        self.theta_psi_H   = theta_psi_H
        self.theta_psi_F   = theta_psi_F

        self.pi_H = (self.beta_F/(1-self.beta_F) - self.cQ)/(self.beta_F/(1-self.beta_F) - (1-self.beta_H)/self.beta_H)
        self.pi_F = (self.beta_H/(1-self.beta_H) - 1/self.cQ)/(self.beta_H/(1-self.beta_H) - (1-self.beta_F)/self.beta_F)

        self.gs2_H  = torch.sum(self.theta_H*self.theta_H, dim=1)
        self.gs2_F  = torch.sum(self.theta_F*self.theta_F, dim=1)
      
    def set_diffusion_premia(self):
        self.eta_H = self.theta_H - self.theta_psi_H + (1-self.pi_H).unsqueeze(-1) *\
                    (self.theta_psi_H-self.theta_psi_F)
        self.eta_F = self.theta_F - self.theta_psi_F + (1-self.pi_F).unsqueeze(-1) *\
                    (self.theta_psi_F-self.theta_psi_H)

        self.eta_I_H = self.eta_H
        self.eta_I_F = self.eta_F

    def set_r(self):
        theta_c_H = self.eta_H+self.theta_psi_H
        theta_c_F = self.eta_F+self.theta_psi_F
        
        self.gs2_c_H = torch.sum(theta_c_H*theta_c_H, dim=1)
        self.gs2_c_F = torch.sum(theta_c_F*theta_c_F, dim=1)
        
        self.r_H = self.pi_H*self.delta_H + (1-self.pi_H)*self.delta_F + self.mu_H +\
                   (1-self.pi_H)*(torch.sum(self.eta_H*theta_c_H, dim=1)-torch.sum(self.eta_F*theta_c_F, dim=1)) -\
                   (1-self.pi_H)*(torch.sum((self.eta_I_H-self.eta_I_F)*(theta_c_F+self.eta_I_H), dim=1)) -\
                   torch.sum( self.eta_H*theta_c_H, dim=1 )
        self.r_F = self.pi_F*self.delta_F + (1-self.pi_F)*self.delta_H + self.mu_F +\
                   (1-self.pi_F)*(torch.sum(self.eta_F*theta_c_F, dim=1)-torch.sum(self.eta_H*theta_c_H, dim=1)) -\
                   (1-self.pi_F)*(torch.sum((self.eta_I_F-self.eta_I_H)*(theta_c_H+self.eta_I_F), dim=1)) -\
                   torch.sum( self.eta_F*theta_c_F, dim=1 )

        self.rc_H = self.r_H + torch.sum((self.eta_I_F-self.eta_I_H)*(self.eta_F-self.eta_I_F), dim=1)
        self.rc_F = self.r_F + torch.sum((self.eta_I_H-self.eta_I_F)*(self.eta_H-self.eta_I_H), dim=1)

    def set_exchange_rates(self):
        gs2_M_I_H = torch.sum(self.eta_I_H*self.eta_I_H, dim=1)
        gs2_M_I_F = torch.sum(self.eta_I_F*self.eta_I_F, dim=1)
        self.mu_e = self.r_H - self.r_F + 0.5*(gs2_M_I_H - gs2_M_I_F)
        self.mu_q = self.mu_e + 0.5*(self.gs2_H - self.gs2_F) - (self.mu_H-self.mu_F)

        self.theta_e = self.eta_I_H - self.eta_I_F
        self.theta_q = self.theta_e - (self.theta_H - self.theta_F)
        self.CRP_H   = 0.5*(gs2_M_I_F - gs2_M_I_H)
        self.gs2_e   = torch.sum(self.theta_e*self.theta_e, dim=1)
        self.SR_H    = self.CRP_H/torch.sqrt(self.gs2_e)
    
    def set_equil(self):
        self.set_diffusion_premia()
        self.set_r()
        self.set_exchange_rates()
        
    def set_moments(self):
        theta_c_H = self.eta_H+self.theta_psi_H
        theta_c_F = self.eta_F+self.theta_psi_F
        
        # diffusion conditional instantaneous variance
        ve = self.gs2_e  
        vcH, vcF = self.gs2_c_H, self.gs2_c_F 
        vgH, vgF = self.gs2_H, self.gs2_c_F 
                        
        moments_dict = {}
        moments_dict['std(Δc̄H)']           = torch.sqrt( vcH )
        moments_dict['std(Δc̄F)']           = torch.sqrt( vcF )
        moments_dict['std(ΔcMH)']          = torch.sqrt( vgH )
        moments_dict['std(ΔcMF)']          = torch.sqrt( vgF )
        moments_dict['std(Δe)']            = torch.sqrt( ve )
        moments_dict['std(Δc̄H-Δc̄F)']       = torch.sqrt( torch.sum((theta_c_H-theta_c_F)*(theta_c_H-theta_c_F), dim=1) )
        moments_dict['corr(Δc̄H-Δc̄F, Δe)']  = ( torch.sum((theta_c_H-theta_c_F)*self.theta_e, dim=1) )/moments_dict['std(Δe)']/moments_dict['std(Δc̄H-Δc̄F)']
        moments_dict['std(Δc̄H)/std(ΔcMH)'] = moments_dict['std(Δc̄H)']/moments_dict['std(ΔcMH)']
        moments_dict['std(Δc̄F)/std(ΔcMF)'] = moments_dict['std(Δc̄F)']/moments_dict['std(ΔcMF)']
        moments_dict['corr(ΔcMH, ΔcMF)']   = ( torch.sum(self.theta_H*self.theta_F, dim=1) )/moments_dict['std(ΔcMH)']/moments_dict['std(ΔcMF)']
        moments_dict['corr(Δc̄H, Δc̄F)']     = ( torch.sum(theta_c_H*theta_c_F, dim=1) )/moments_dict['std(Δc̄H)']/moments_dict['std(Δc̄F)']
        
        moments_dict['corr(Δc̄H, ΔcMH)']    = ( torch.sum(self.theta_H*theta_c_H, dim=1) )/moments_dict['std(ΔcMH)']/moments_dict['std(Δc̄H)']
        moments_dict['corr(Δc̄F, ΔcMF)']    = ( torch.sum(self.theta_F*theta_c_F, dim=1) )/moments_dict['std(ΔcMF)']/moments_dict['std(Δc̄F)']
        moments_dict['std(Δe)/std(ΔcMH)']  = moments_dict['std(Δe)']/moments_dict['std(ΔcMH)']
        moments_dict['std(Δe)/std(ΔcMF)']  = moments_dict['std(Δe)']/moments_dict['std(ΔcMF)']
        #del moments_dict['std(Δc̄H-Δc̄F)']
        self.moments_dict = moments_dict
        
        self.corr_ddc_de                   = ( torch.sum((theta_c_H-theta_c_F)*self.theta_e, dim=1) )/moments_dict['std(Δe)']/moments_dict['std(Δc̄H-Δc̄F)']
        self.corr_dcH_dgH                  = ( torch.sum(self.theta_H*theta_c_H, dim=1) )/moments_dict['std(ΔcMH)']/moments_dict['std(Δc̄H)']
        self.corr_dcF_dgF                  = ( torch.sum(self.theta_F*theta_c_F, dim=1) )/moments_dict['std(ΔcMF)']/moments_dict['std(Δc̄F)']
        self.corr_dgH_dgF                  = ( torch.sum(self.theta_H*self.theta_F, dim=1) )/moments_dict['std(ΔcMH)']/moments_dict['std(ΔcMF)']
        self.std_de_dgH                    = moments_dict['std(Δe)']/moments_dict['std(ΔcMH)']
        self.std_de_dgF                    = moments_dict['std(Δe)']/moments_dict['std(ΔcMF)']
        self.std_dcH_dgH                   = moments_dict['std(Δc̄H)']/moments_dict['std(ΔcMH)']
        self.std_dcF_dgF                   = moments_dict['std(Δc̄F)']/moments_dict['std(ΔcMF)']
        
        self.gs2_M_H                       = torch.sum(self.eta_H*self.eta_H, dim=1)   
        self.gs2_M_F                       = torch.sum(self.eta_F*self.eta_F, dim=1)   
        self.gs2_MI_H                      = torch.sum(self.eta_I_H*self.eta_I_H, dim=1)   
        self.gs2_MI_F                      = torch.sum(self.eta_I_F*self.eta_I_F, dim=1)   
        
        self.corr_dmH_dmF                  = torch.sum(self.eta_H*self.eta_F, dim=1)/torch.sqrt(self.gs2_M_H*self.gs2_M_F)
        self.corr_dmIH_dmIF                = torch.sum(self.eta_I_H*self.eta_I_F, dim=1)/torch.sqrt(self.gs2_MI_H*self.gs2_MI_F)

class frictional(baseline):
    """
    Classes for forex economies with frictions
    """
    def __init__(self, betas, cQ, theta_H, theta_F, theta_psi_H, theta_psi_F):
        super().__init__(betas, cQ, theta_H, theta_F, theta_psi_H, theta_psi_F)
      
    def set_diffusion_premia(self):
        pi = 0.5 + 0.5*(1 - self.pi_H + 1 - self.pi_F)
        cov_cM = torch.sum(self.theta_H*self.theta_F, dim=1)

        a00 = (2-self.pi_F)/pi*self.gs2_H
        a01 = (1-self.pi_H)/pi*cov_cM
        a10 = (1-self.pi_F)/pi*cov_cM
        a11 = (2-self.pi_H)/pi*self.gs2_F

        pi_A = torch.reshape(torch.cat([torch.reshape(torch.cat([a00, a01]), (2, self.n_state)),
                                        torch.reshape(torch.cat([a10, a11]), (2, self.n_state))], dim=0).T, (self.n_state, 2, 2))

        pi_A_inv = torch.linalg.inv(pi_A)

        cov_psi_H = torch.sum((self.theta_psi_H-self.theta_psi_F)*self.theta_H, dim=1)
        cov_psi_F = torch.sum((self.theta_psi_F-self.theta_psi_H)*self.theta_F, dim=1)
        pi_b = torch.reshape(torch.reshape(torch.cat([(1-self.pi_H)*cov_psi_H,
                                                      (1-self.pi_F)*cov_psi_F]), (2, self.n_state)).T, (self.n_state, 2, 1))

        gl    = torch.matmul(pi_A_inv,pi_b)+1

        self.eta_H = self.theta_H - self.theta_psi_H + torch.reshape((1-self.pi_H)/(2*pi), (self.n_state, 1)) *\
                    (self.theta_psi_H-self.theta_psi_F + gl[:,0,:]*self.theta_H - gl[:,1,:]*self.theta_F - (self.theta_H-self.theta_F))
        self.eta_F = self.theta_F - self.theta_psi_F + torch.reshape((1-self.pi_F)/(2*pi), (self.n_state, 1)) *\
                    (self.theta_psi_F-self.theta_psi_H + gl[:,1,:]*self.theta_F - gl[:,0,:]*self.theta_H - (self.theta_F-self.theta_H))

        self.eta_I_H = 2*self.eta_H + self.theta_psi_H - gl[:,0,:]*self.theta_H
        self.eta_I_F = 2*self.eta_F + self.theta_psi_F - gl[:,1,:]*self.theta_F
        
        self.gl = gl

# create the grid of states (cQ, x)
def get_states(betas):
    beta_H,  beta_F  = betas #torch.tensor([0.9, 0.9], device=device, dtype=dtype)
    cQ_array         = torch.exp(torch.linspace( torch.log((1-beta_H)/beta_H), torch.log(beta_F/(1-beta_F)), 65, device=device, dtype=dtype))
    x_array          = torch.linspace( -1.0, 1.0, 65, device=device, dtype=dtype )
    
    return cQ_array, x_array

# setting the drift/diffusion coefficients given (cQ, x) for the economic model
def set_inputs(paras):
    betas, gs_bar_cM, rho_cM, gs_bar_Psi, s_psi, rho_psi, gs_carry, gs_ERP, is_xt_const = paras
    betas = torch.tensor(betas, device=device, dtype=dtype)
    cQ_array, x_array = get_states(betas)
    
    cQ, x   = torch.meshgrid(cQ_array, x_array, indexing='ij')
    n_cQ    = cQ_array.size()[0]
    n_x     = x_array.size()[0]
    n_state = n_cQ*n_x
    
    cQ = torch.reshape(cQ, (n_state,))
    x  = torch.reshape(x, (n_state,))

    theta_H = torch.tensor([[gs_bar_cM]*n_state,
                            [.0]*n_state,
                            [gs_bar_cM]*n_state,
                            [.0]*n_state,
                            [.0]*n_state,
                            [.0]*n_state,
                            ], device=device, dtype=dtype).T
    
    theta_F = torch.tensor([[.0]*n_state,
                            [gs_bar_cM]*n_state,
                            [gs_bar_cM*rho_cM]*n_state,
                            [.0]*n_state,
                            [.0]*n_state,
                            [gs_bar_cM*np.sqrt(1-rho_cM**2)]*n_state,                  
                            ], device=device, dtype=dtype).T
    
    gs_t_Psi = (-1.0+torch.exp(4*x.pow(2)))*(1-is_xt_const)+is_xt_const*0.12

    theta_psi_H = torch.tensor([[gs_bar_Psi*s_psi]*n_state,
                                [.0]*n_state,
                                [gs_carry]*n_state, 
                                list(gs_t_Psi.cpu().numpy()*np.sqrt(1-(s_psi)**2)),
                                [.0]*n_state,
                                [gs_ERP]*n_state, 
                                ], device=device, dtype=dtype).T

    theta_psi_F = torch.tensor([[.0]*n_state,
                                [gs_bar_Psi*s_psi]*n_state,
                                [gs_carry]*n_state,
                                list(rho_psi*(gs_t_Psi).cpu().numpy()*np.sqrt(1-s_psi**2)),
                                list(np.sqrt(1-rho_psi**2)*(gs_t_Psi).cpu().numpy()*np.sqrt(1-s_psi**2)),
                                [gs_ERP]*n_state,
                                ], device=device, dtype=dtype).T
    
    return cQ, x, n_cQ, n_x, n_state, theta_H, theta_F, theta_psi_H, theta_psi_F

# routines to create the economic model
def get_econ_wrapper(is_frictional, paras):
    betas = torch.tensor(paras[0], device=device, dtype=dtype)
    cQ, x, n_cQ, n_x, n_state, theta_H, theta_F, theta_psi_H, theta_psi_F = set_inputs(paras)
    if is_frictional:
        econ = frictional(betas, cQ, theta_H, theta_F, theta_psi_H, theta_psi_F)
    else:
        econ = baseline(betas, cQ, theta_H, theta_F, theta_psi_H, theta_psi_F)
    econ.set_equil()
    econ.set_moments()
    
    return econ


# helper function for matrix multiplication
class torch_bilinear(object):
    def __init__(self, data, z):
        self.x, self.y = data
        self.z = z
        self.grid_points = torch.meshgrid(self.x, self.y, indexing='ij')
        
    def interp(self, query):
        xs, ys = query
        n_state = xs.size()[0]
        idxs = torch.searchsorted( self.x[1:], xs )
        idys = torch.searchsorted( self.y[1:], ys )
        dx = self.x[idxs+1]-self.x[idxs]
        dy = self.y[idys+1]-self.y[idys]
        
        f_A = torch.reshape(torch.cat((torch.reshape(torch.cat( (self.z[idxs, idys],   self.z[idxs, idys+1])   ), (2, n_state)),
                                       torch.reshape(torch.cat( (self.z[idxs+1, idys], self.z[idxs+1, idys+1]) ), (2, n_state))), dim=0).T, (n_state, 2,2) )
        
        lf = torch.reshape(torch.cat(((self.x[idxs+1]-xs).unsqueeze(-1), (xs-self.x[idxs]).unsqueeze(-1)), dim=1), (n_state,1,2))
        rf = torch.reshape(torch.cat(((self.y[idys+1]-ys).unsqueeze(-1), (ys-self.y[idys]).unsqueeze(-1)), dim=1), (n_state,2,1))
        
        return torch.reshape(lf@f_A@rf, (n_state,))/dx/dy

# approximating the endogenous variables given the state variables (cQ, x)
def torch_set_approx(is_frictional, paras):
    betas = torch.tensor(paras[0], device=device, dtype=dtype)    
    cQ, x, n_cQ, n_x, n_state, theta_H, theta_F, theta_psi_H, theta_psi_F = set_inputs(paras)
    
    if is_frictional:
        econ = frictional(betas, cQ, theta_H, theta_F, theta_psi_H, theta_psi_F)
    else:
        econ = baseline(betas, cQ, theta_H, theta_F, theta_psi_H, theta_psi_F)
    econ.set_equil()
    econ.set_moments()
    
    approx_scalar_list = ['r_H', 'r_F', 'rc_H', 'rc_F', 'pi_H', 'pi_F', 
                          'mu_q', 'mu_e', 'mu_H', 'mu_F', 'mu_psi_H', 'mu_psi_F']
                          #'μc_H', 'μc_F']
    approx_vector_list = ['theta_q', 'theta_e', 'eta_H', 'eta_I_H', 'eta_F', 'eta_I_F', 
                          'theta_psi_H', 'theta_psi_F', 'theta_H', 'theta_F']#, 'θc_H', 'θc_F']

    approx_out  = {} # save the approx values


    for item in approx_vector_list:
        approx_out[item] = {}

                        
    # approximating the scalar variables (drifts/shares/rates etc.)
    cQ_array, x_array = get_states(betas)
    approx_out['r_H'] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.r_H, (n_cQ,n_x)))
    approx_out['r_F'] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.r_F, (n_cQ,n_x)))
    approx_out['rc_H'] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.rc_H, (n_cQ,n_x)))
    approx_out['rc_F'] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.rc_F, (n_cQ,n_x)))
    approx_out['pi_H'] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.pi_H, (n_cQ,n_x)))
    approx_out['pi_F'] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.pi_F, (n_cQ,n_x)))
    approx_out['mu_q'] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.mu_q, (n_cQ,n_x)))
    approx_out['mu_e'] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.mu_e, (n_cQ,n_x)))
    approx_out['mu_H'] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.mu_H - 0.5*econ.gs2_H, (n_cQ,n_x)))
    approx_out['mu_F'] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.mu_F - 0.5*econ.gs2_F, (n_cQ,n_x)))
    approx_out['mu_psi_H'] = torch_bilinear((cQ_array, x_array), torch.reshape(-econ.delta_H - 0.5*torch.sum(econ.theta_psi_H*econ.theta_psi_H, dim=1), (n_cQ,n_x)))
    approx_out['mu_psi_F'] = torch_bilinear((cQ_array, x_array), torch.reshape(-econ.delta_F - 0.5*torch.sum(econ.theta_psi_F*econ.theta_psi_F, dim=1), (n_cQ,n_x)))


    # approximating the vector variables for shock loadings
    for item_dim in range(econ.theta_q.size()[1]):
        approx_out['theta_q'][item_dim] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.theta_q[:,item_dim], (n_cQ,n_x)))
        approx_out['theta_e'][item_dim] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.theta_e[:,item_dim], (n_cQ,n_x)))
        approx_out['eta_H'][item_dim] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.eta_H[:,item_dim], (n_cQ,n_x)))
        approx_out['eta_F'][item_dim] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.eta_F[:,item_dim], (n_cQ,n_x)))
        approx_out['eta_I_H'][item_dim] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.eta_I_H[:,item_dim], (n_cQ,n_x)))
        approx_out['eta_I_F'][item_dim] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.eta_I_F[:,item_dim], (n_cQ,n_x)))
        approx_out['theta_H'][item_dim] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.theta_H[:,item_dim], (n_cQ,n_x)))
        approx_out['theta_F'][item_dim] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.theta_F[:,item_dim], (n_cQ,n_x)))
        approx_out['theta_psi_H'][item_dim] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.theta_psi_H[:,item_dim], (n_cQ,n_x)))
        approx_out['theta_psi_F'][item_dim] = torch_bilinear((cQ_array, x_array), torch.reshape(econ.theta_psi_F[:,item_dim], (n_cQ,n_x)))

    return approx_out

def torch_get_approx_scalar(curr_state, curr_out):
    out_dict = {}
    out_dict['pi_H']  = curr_out['pi_H'].interp(curr_state)
    out_dict['pi_F']  = curr_out['pi_F'].interp(curr_state)
    out_dict['r_H']   = curr_out['r_H'].interp(curr_state)
    out_dict['r_F']   = curr_out['r_F'].interp(curr_state)
    out_dict['r_FH']  = curr_out['rc_H'].interp(curr_state)
    out_dict['r_HF']  = curr_out['rc_F'].interp(curr_state)
    out_dict['mu_q']  = curr_out['mu_q'].interp(curr_state)
    out_dict['mu_e']  = curr_out['mu_e'].interp(curr_state)
    out_dict['mu_H']  = curr_out['mu_H'].interp(curr_state)
    out_dict['mu_F']  = curr_out['mu_F'].interp(curr_state)  

    return out_dict  

def torch_get_approx_array(curr_state, curr_out):
    out_dict = {}
    n_shock  = len(curr_out['theta_q'])
    out_dict['cQ']    = torch.cat([curr_out['theta_q'][item].interp(curr_state).unsqueeze(-1) for item in range(n_shock)], dim=1)
    out_dict['cE']    = torch.cat([curr_out['theta_e'][item].interp(curr_state).unsqueeze(-1) for item in range(n_shock)], dim=1)
    out_dict['cM_H']  = torch.cat([curr_out['theta_H'][item].interp(curr_state).unsqueeze(-1) for item in range(n_shock)], dim=1)
    out_dict['cM_F']  = torch.cat([curr_out['theta_F'][item].interp(curr_state).unsqueeze(-1) for item in range(n_shock)], dim=1)
    out_dict['M_I_H']   = torch.cat([curr_out['eta_I_H'][item].interp(curr_state).unsqueeze(-1) for item in range(n_shock)], dim=1)
    out_dict['M_I_F']   = torch.cat([curr_out['eta_I_F'][item].interp(curr_state).unsqueeze(-1) for item in range(n_shock)], dim=1)

    return out_dict

# %% simulate sample paths
def torch_get_sim(n_path, T, step_size, init_x, init_cQ, is_frictional, paras, gs_x, gk_x):
    betas = torch.tensor(paras[0], device=device, dtype=dtype)
    cQ_array, x_array = get_states(betas)
    if is_frictional:
        curr_out = torch_set_approx(True, paras)
    else:
        curr_out = torch_set_approx(False, paras)
    
    n_step     = int(T/step_size)         # number of steps
    state_dim  = len(curr_out['theta_q']) # number of diffusion shocks

    df_dict    = {} # holder of the full sample path
    curr_dict  = {} # holder of the current slice of the path (state)

    # output variable list
    res_columns = ['pi_H', 'pi_F', 'r_H', 'r_F', 'r_FH', 'r_HF', 
                    'mu_q', 'mu_e', 'x', 'GH', 'GF', 
                    'cQ', 'cE', 'cM_H', 'cM_F']
                    #'MI_H', 'cM_H', 'cM_F', 'Psi_H', 'Psi_F']

    # variables list that requires special treatment
    sub_columns = ['pi_H', 'pi_F', 'r_H', 'r_F', 'r_FH', 'r_HF', 
                    'mu_q', 'mu_e']

    ### output initialization
    for item_col in res_columns:
        df_dict[item_col]   = torch.ones((n_step+1, n_path), device=device, dtype=dtype)
        curr_dict[item_col] = torch.ones((1, n_path), device=device, dtype=dtype)
    df_dict['S_H']   = torch.ones((n_step+1, n_path), device=device, dtype=dtype)
    df_dict['S_F']   = torch.ones((n_step+1, n_path), device=device, dtype=dtype)
    curr_dict['S_H'] = torch.ones((1, n_path), device=device, dtype=dtype)
    curr_dict['S_F'] = torch.ones((1, n_path), device=device, dtype=dtype)

    # set the home-bias parameter
    beta_H, beta_F = betas #torch.tensor((0.9, 0.9), device=device, dtype=dtype) 
    # set the initial value of the xt process
    x = init_x 
    # set the long-run mean of the xt process
    x_bar = .0 
    gamma  = 4 # set the leverage of equity claims on the aggregate endowments

    # initialize the state variables
    curr_dict['x'][0,:]  = curr_dict['x'][0,:]*x
    curr_dict['cQ'][0,:] = curr_dict['cQ'][0,:]*init_cQ
    curr_dict['cE'][0,:] = curr_dict['cQ'][0,:]

    # output the initial state
    for state_item in ['x', 'cQ', 'cE']:
        df_dict[state_item][0,:] = curr_dict[state_item][0,:]

    # simulation begins
    for step_item in range(n_step):
        curr_time   = step_item*step_size 
        curr_state  = ( curr_dict['cQ'][0,:], curr_dict['x'][0,:] ) 

        # compute the drift and diffusion coefficients
        curr_scalar = torch_get_approx_scalar(curr_state, curr_out)
        curr_array  = torch_get_approx_array(curr_state, curr_out)
        
        curr_drift         = {}
        curr_drift['cQ']   = curr_scalar['mu_q']
        curr_drift['cE']   = curr_scalar['mu_e']
        curr_drift['cM_H'] = curr_scalar['mu_H']
        curr_drift['cM_F'] = curr_scalar['mu_F']

        for col_item in sub_columns:
            curr_dict[col_item][0,:]       = curr_scalar[col_item]
            df_dict[col_item][step_item,:] = curr_dict[col_item][0,:]

        curr_drift['S_H'] = gamma*(curr_scalar['mu_H']) + gamma**2*0.5*torch.sum(curr_array['cM_H']*curr_array['cM_H'], dim=1) - gamma*torch.sum(curr_array['cM_H']*curr_array['M_I_H'], dim=1) -curr_scalar['r_H']
        curr_drift['S_F'] = gamma*(curr_scalar['mu_F']) + gamma**2*0.5*torch.sum(curr_array['cM_F']*curr_array['cM_F'], dim=1) - gamma*torch.sum(curr_array['cM_F']*curr_array['M_I_F'], dim=1) -curr_scalar['r_F']

        # simulate the diffusion shocks for the current step
        bm_shocks = torch.randn(n_path, state_dim+1, device=device, dtype=dtype)

        dt = torch.tensor((step_item+1)*step_size-curr_time)
        curr_dict['S_H']  = torch.exp(curr_drift['S_H']*dt)*curr_dict['cM_H']**gamma
        curr_dict['S_F']  = torch.exp(curr_drift['S_F']*dt)*curr_dict['cM_F']**gamma
        df_dict['S_H'][step_item,:] = curr_dict['S_H'][0,:]
        df_dict['S_F'][step_item,:] = curr_dict['S_F'][0,:]

        # simulate the state variables for the next step
        for state_item in ['cQ', 'cE', 'cM_H', 'cM_F']:
            curr_item = curr_dict[state_item][0,:]

            curr_dict[state_item][0,:] = torch.exp( torch.log(curr_item) 
                                                + dt*curr_drift[state_item]
                                                + torch.sqrt(dt)*torch.sum(bm_shocks[:,:-1]*curr_array[state_item], dim=1))

            if state_item == 'cQ':
                curr_dict[state_item][0,:] = torch.minimum(torch.maximum(curr_dict[state_item][0,:], cQ_array[0]), cQ_array[-1])
            # output the state variables for the next step
            df_dict[state_item][step_item+1,:] = curr_dict[state_item][0,:]
        # simulate the xt variable for the next step
        curr_x  = curr_dict['x'][0,:]
        shock_x = bm_shocks[:,-1]
        curr_x  = curr_x + dt*gk_x*(x_bar-curr_x) \
                         + torch.sqrt(dt)*gs_x*torch.sqrt((x_array[-1]-curr_x)*(curr_x-x_array[0]))*shock_x#torch.sqrt(dt)*gs_x*torch.sqrt((x_array[-23]-curr_x)*(curr_x-x_array[22]))*shock_x
        curr_x = torch.minimum(torch.maximum(curr_x, x_array[0]), x_array[-1])
        # output the state variables for the next step
        curr_dict['x'][0,:] = curr_x
        df_dict['x'][step_item+1,:] = curr_dict['x'][0,:]

    # computing the state variables and output results for the final step
    curr_state  = (curr_dict['cQ'][0,:], curr_dict['x'][0,:])
    curr_scalar = torch_get_approx_scalar(curr_state, curr_out)
    for col_item in sub_columns:
        curr_dict[col_item][0,:]         = curr_scalar[col_item]
        df_dict[col_item][step_item+1,:] = curr_dict[col_item][0,:]

    curr_array  = torch_get_approx_array(curr_state, curr_out)
    curr_drift['S_H'] = gamma*(curr_scalar['mu_H']) + gamma**2*0.5*torch.sum(curr_array['cM_H']*curr_array['cM_H'], dim=1) - gamma*torch.sum(curr_array['cM_H']*curr_array['M_I_H'], dim=1) -curr_scalar['r_H']
    curr_drift['S_F'] = gamma*(curr_scalar['mu_F']) + gamma**2*0.5*torch.sum(curr_array['cM_F']*curr_array['cM_F'], dim=1) - gamma*torch.sum(curr_array['cM_F']*curr_array['M_I_F'], dim=1) -curr_scalar['r_F']

    curr_dict['S_H']  = torch.exp(curr_drift['S_H']*dt)*curr_dict['cM_H']**gamma
    curr_dict['S_F']  = torch.exp(curr_drift['S_F']*dt)*curr_dict['cM_F']**gamma
    df_dict['S_H'][step_item+1,:] = curr_dict['S_H'][0,:]
    df_dict['S_F'][step_item+1,:] = curr_dict['S_F'][0,:]

    # computing other endogenous variables
    df_dict['r_H-r_F'] = df_dict['r_H'] - df_dict['r_F']
    df_dict['e']       = torch.log(df_dict['cE'])
    df_dict['q']       = torch.log(df_dict['cQ'])
    df_dict['CIP_H']   = df_dict['r_F'] - df_dict['r_HF'] # US being the foreign country, and CIP_H captures the home country investor's USD borrowing cost
    df_dict['CIP_F']   = df_dict['r_H'] - df_dict['r_FH']
    df_dict['C_H']     = df_dict['pi_H']*df_dict['cM_H']/beta_H
    df_dict['C_F']     = df_dict['pi_F']*df_dict['cM_F']/beta_F
    df_dict['g_H']     = torch.log(df_dict['cM_H'])
    df_dict['g_F']     = torch.log(df_dict['cM_F'])
    df_dict['c_H']     = torch.log(df_dict['C_H'])
    df_dict['c_F']     = torch.log(df_dict['C_F'])

    return df_dict


# %% computing the moments
# Computing beta coefficients for simulated moments
def get_beta(df_dict):
    path_size = df_dict['CIP_H'].size()[0]-1

    res_dict  = {}
    res_dict['CIP-$\\beta$']      = []
    res_dict['Fama-$\\beta$']     = []
    res_dict['t CIP']             = []
    res_dict['t Fama-$\\beta$']   = []
    res_dict['std(CIP-$\\beta$)'] = []
    res_dict['$R^2$ CIP(\%)']     = []
    res_dict['$R^2$ Fama(\%)']    = []

    for path_item in range(df_dict['r_H'].shape[1]):
        df_reg = pd.DataFrame(torch.concat([torch.reshape(df_dict['CIP_H'][:, path_item].diff(), (path_size, 1)),
                                            torch.reshape(df_dict['e'][:, path_item].diff(), (path_size, 1)),
                                            torch.reshape(df_dict['r_H-r_F'][:-1, path_item], (path_size, 1)),
                                            torch.reshape(df_dict['CRP_H'][:, [path_item]], (path_size, 1))], dim=1).cpu().numpy())
        df_reg.columns = ['dCIP', 'de', 'r_H-r_F', 'CRP_H'] 

        if df_reg['r_H-r_F'].std()>1e-7: # dealing with the case when the interest rate is constant
            # Fama beta
            X     = sm.add_constant((df_reg[['r_H-r_F']])/12)
            y     = df_reg['CRP_H']
            model = sm.OLS(y, X)
            results = model.fit(cov_type='HC3')
            res_dict['Fama-$\\beta$'].append(results.params.iloc[1])
            res_dict['t Fama-$\\beta$'].append(results.tvalues.iloc[1])
            res_dict['$R^2$ Fama(\%)'].append(results.rsquared_adj*100) # converting to percentage
        else:
            res_dict['Fama-$\\beta$'].append(np.NaN)
            res_dict['t Fama-$\\beta$'].append(np.NaN)
            res_dict['$R^2$ Fama(\%)'].append(np.NaN)
        
        if df_reg['dCIP'].std()>1e-7: # dealing with the case when the cip deviation is zero
            X     = sm.add_constant(df_reg[['de']])
            y     = df_reg['dCIP']*100 # converting to basis points
            model = sm.OLS(y, X)
            results = model.fit(cov_type='HC3')
            res_dict['CIP-$\\beta$'].append(results.params.iloc[1])
            res_dict['t CIP'].append(results.tvalues.iloc[1])
            res_dict['std(CIP-$\\beta$)'].append(results.bse.iloc[1])
            res_dict['$R^2$ CIP(\%)'].append(results.rsquared_adj*100) # converting to percentage
        else:
            res_dict['CIP-$\\beta$'].append(np.NaN)
            res_dict['std(CIP-$\\beta$)'].append(np.NaN)
            res_dict['t CIP'].append(np.NaN)
            res_dict['$R^2$ CIP(\%)'].append(np.NaN)
    return pd.DataFrame(res_dict)

# helper for the covariance computation
def get_cov(x, y):
    return torch.mean(x.diff(dim=0)*y.diff(dim=0), dim=0) - torch.mean(x.diff(dim=0), dim=0)*torch.mean(y.diff(dim=0), dim=0)

# computing simulated moments
class torch_moments(object):
    def __init__(self, df_dict, n_path):
        
        for item in df_dict.keys():
            df_dict[item] = df_dict[item][480:,:] # discarding the first 40 years

        gamma  = 4 # setting the leverage of equity claims on the aggregate endowments
        moments_dict = {}
        moments_dict['std($d\\bar c_H$)(\%)']       = pd.DataFrame(df_dict['c_H'].cpu().numpy()).diff().std().median()*np.sqrt(12)*100
        moments_dict['std($d\\bar c_F$)(\%)']       = pd.DataFrame(df_dict['c_F'].cpu().numpy()).diff().std().median()*np.sqrt(12)*100
        moments_dict['std($dg_H$)(\%)']             = pd.DataFrame(df_dict['g_H'].cpu().numpy()).diff().std().median()*np.sqrt(12)*100
        moments_dict['std($dg_F$)(\%)']             = pd.DataFrame(df_dict['g_F'].cpu().numpy()).diff().std().median()*np.sqrt(12)*100
        moments_dict['std($\pi_H$)(\%)']            = pd.DataFrame(df_dict['pi_H'].cpu().numpy()).std().median()*100
        moments_dict['std($de$)(\%)']               = pd.DataFrame(df_dict['e'].cpu().numpy()).diff().std().median()*np.sqrt(12)*100
        moments_dict['std($r_H-r_F$)(\%)']          = pd.DataFrame(df_dict['r_H-r_F'].cpu().numpy()).std().median()*100
        moments_dict['std($r_H$)(\%)']              = pd.DataFrame(df_dict['r_H'].cpu().numpy()).std().median()*100
        moments_dict['std($r_F$)(\%)']              = pd.DataFrame(df_dict['r_F'].cpu().numpy()).std().median()*100
        moments_dict['mean(CIP$_H$)(\%)']           = pd.DataFrame(df_dict['CIP_H'].cpu().numpy()).mean().median()*100
        moments_dict['mean(CIP$_F$)(\%)']           = pd.DataFrame(df_dict['CIP_F'].cpu().numpy()).mean().median()*100
        moments_dict['ac1($r_H-r_F$)']              = pd.DataFrame(df_dict['r_H-r_F'].cpu().numpy()).apply(lambda col: col.autocorr(1)).median()
        moments_dict['ac1($r_H$)']                  = pd.DataFrame(df_dict['r_H'].cpu().numpy()).apply(lambda col: col.autocorr(1)).median()
        moments_dict['ac1($de$)']                   = pd.DataFrame(df_dict['e'].cpu().numpy()).diff().apply(lambda col: col.autocorr(1)).median()
        moments_dict['std($de$)/std($d\\bar c_H$)'] = (pd.DataFrame(df_dict['e'].cpu().numpy()).diff().std()/pd.DataFrame(df_dict['c_H'].cpu().numpy()).diff().std()).median()
        moments_dict['std($de$)/std($dc_F$)']       = (pd.DataFrame(df_dict['e'].cpu().numpy()).diff().std()/pd.DataFrame(df_dict['c_F'].cpu().numpy()).diff().std()).median()
        moments_dict['std($de$)/std($dg_H$)']       = (pd.DataFrame(df_dict['e'].cpu().numpy()).diff().std()/pd.DataFrame(df_dict['g_H'].cpu().numpy()).diff().std()).median()
        moments_dict['std($de$)/std($dg_F$)']       = (pd.DataFrame(df_dict['e'].cpu().numpy()).diff().std()/pd.DataFrame(df_dict['g_F'].cpu().numpy()).diff().std()).median()
        moments_dict['std($r_H-r_F$)/std($de$)']    = pd.DataFrame(df_dict['r_H-r_F'].cpu().numpy()).std().median()*100/(pd.DataFrame(df_dict['e'].cpu().numpy()).diff().std().median()*np.sqrt(12)*100)
        moments_dict['mean($r_H$)(\%)']             = pd.DataFrame(df_dict['r_H'].cpu().numpy()).mean().median()*100
        moments_dict['mean($r_F$)(\%)']             = pd.DataFrame(df_dict['r_F'].cpu().numpy()).mean().median()*100

        # computing excess stock return for home and foreign equities
        df_dict['r_S_H'] = (df_dict['cM_H'][1:,:]**gamma/df_dict['S_H'][0:-1,:]-1-df_dict['r_H'][0:-1,:]/12)
        df_dict['r_S_F'] = (df_dict['cM_F'][1:,:]**gamma/df_dict['S_F'][0:-1,:]-1-df_dict['r_F'][0:-1,:]/12)
        moments_dict['ERP$_H$(\%)']                 = torch.median( df_dict['r_S_H'].mean(dim=0) ).cpu().numpy().item()*12*100
        moments_dict['equity SR$_H$(\%)']           = torch.median( df_dict['r_S_H'].mean(dim=0)/df_dict['r_S_H'].std(dim=0) ).cpu().numpy().item()*np.sqrt(12)*100
        moments_dict['ERP$_F$(\%)']                 = torch.median( df_dict['r_S_F'].mean(dim=0) ).cpu().numpy().item()*12*100
        moments_dict['equity SR$_F$(\%)']           = torch.median( df_dict['r_S_F'].mean(dim=0)/df_dict['r_S_F'].std(dim=0) ).cpu().numpy().item()*np.sqrt(12)*100
        
        # helper for computing the excess equity return correlation cross countries
        tmp = torch.concat([df_dict['r_S_H'], df_dict['r_S_F']], dim=1).T.cov()
        moments_dict['corr(ERP$_H$, ERP$_F$)']      = torch.median(tmp[n_path:,:n_path].diag()/torch.sqrt(tmp[:n_path,:n_path].diag()*tmp[n_path:,n_path:].diag())).cpu().numpy().item()

        # helper for computing the carry trade related quatiities
        df_dict['rate_ret']    = df_dict['r_H'][0:-1,:]/12 - df_dict['r_F'][0:-1,:]/12
        df_dict['rate_ret']    = torch.sign(df_dict['rate_ret'])*df_dict['rate_ret']
        df_dict['rate_de']     = torch.sign(df_dict['rate_ret'])*(- torch.log(df_dict['cE'])[1:,:] + torch.log(df_dict['cE'])[0:-1,:])
        df_dict['CRP_H']       = df_dict['r_H'][0:-1,:]/12 - df_dict['r_F'][0:-1,:]/12 - torch.log(df_dict['cE'])[1:,:] + torch.log(df_dict['cE'])[0:-1,:]
        df_dict['crp']         = df_dict['CRP_H']*(df_dict['r_H'][0:-1,:]>df_dict['r_F'][0:-1,:]) -  df_dict['CRP_H']*(df_dict['r_H'][0:-1,:]<df_dict['r_F'][0:-1,:])
        
        moments_dict['carry SR(\%)']               = torch.median( torch.mean(df_dict['crp'], dim=0)/torch.std(df_dict['crp'], dim=0) ).cpu().numpy().item()*np.sqrt(12)*100
        moments_dict['carry (\%)']                 = torch.median( torch.mean(df_dict['crp'], dim=0) ).cpu().numpy().item()*12*100
        moments_dict['std(carry) (\%)']            = torch.median( torch.std(df_dict['crp'], dim=0) ).cpu().numpy().item()*np.sqrt(12)*100
        moments_dict['carry i-diff (\%)']          = torch.median(torch.mean(df_dict['rate_ret'], dim=0)).cpu().numpy().item()*12*100
        moments_dict['std(carry i-diff) (\%)']     = torch.median(torch.std(df_dict['rate_ret'], dim=0)).cpu().numpy().item()*np.sqrt(12)*100
        moments_dict['rate_de']                    = torch.median(torch.mean(df_dict['rate_de'], dim=0)).cpu().numpy().item()*12
        moments_dict['carry ratio (\%)']           = torch.median(torch.mean(df_dict['rate_ret'], dim=0)/torch.mean(df_dict['crp'], dim=0)).cpu().numpy()*100

        # helper for computing the correlation between short rates cross countries
        tmp = torch.concat([df_dict['r_H'], df_dict['r_F']], dim=1).T.cov()
        moments_dict['corr($r_H, r_F$)']           = torch.median(tmp[n_path:,:n_path].diag()/torch.sqrt(tmp[:n_path,:n_path].diag()*tmp[n_path:,n_path:].diag())).cpu().numpy().item()

        moments_dict['corr($dr_H, dr_F$)']         = torch.median( get_cov(df_dict['r_H'], df_dict['r_F'])/torch.sqrt(get_cov(df_dict['r_H'], df_dict['r_H'])*get_cov(df_dict['r_F'], df_dict['r_F'])) ).cpu().numpy().item()
        moments_dict['corr($dg_H, dg_F$)']         = torch.median( get_cov(df_dict['g_H'],df_dict['g_F'])/torch.sqrt(get_cov(df_dict['g_H'],df_dict['g_H'])*get_cov(df_dict['g_F'],df_dict['g_F'])) ).cpu().numpy().item()
        moments_dict['corr($d\\bar c_H, d\\bar c_F$)']    = torch.median( get_cov(df_dict['c_H'],df_dict['c_F'])/torch.sqrt(get_cov(df_dict['c_H'],df_dict['c_H'])*get_cov(df_dict['c_F'],df_dict['c_F'])) ).cpu().numpy().item()
        moments_dict['corr($d\\bar c_H, dg_H$)']          = torch.median( get_cov(df_dict['c_H'],df_dict['g_H'])/torch.sqrt(get_cov(df_dict['g_H'],df_dict['g_H'])*get_cov(df_dict['c_H'],df_dict['c_H'])) ).cpu().numpy().item()
        moments_dict['corr($d\\bar c_H-d\\bar c_F, de$)'] = torch.median( get_cov(df_dict['c_H']-df_dict['c_F'],df_dict['e'])/torch.sqrt(get_cov(df_dict['e'],df_dict['e'])*get_cov(df_dict['c_H']-df_dict['c_F'],df_dict['c_H']-df_dict['c_F'])) ).cpu().numpy().item()  
        
        df_tmp = get_beta(df_dict).median()
        for item in df_tmp.index:
            moments_dict[item] = df_tmp[item]
        
        self.moments_dict = moments_dict


# %% routine to plot the endogenous variables for diagnosis examination
def econ_plot_wrapper(econ):
    betas = (econ.beta_H, econ.beta_F)
    betas = torch.tensor(betas, device=device, dtype=dtype)
    cQ_array, x_array = get_states(betas)
    
    x, y = torch.meshgrid(cQ_array, x_array, indexing='ij')
    x = torch.log(x)
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    
    n_cQ = cQ_array.size()[0]
    n_x  = x_array.size()[0]
    
    moments = {}
    
    moments['SR_H']     = torch.reshape(econ.SR_H, (n_cQ,n_x)).cpu().numpy()
    moments['r_H-r_F']  = torch.reshape(econ.r_H-econ.r_F, (n_cQ,n_x)).cpu().numpy()
    moments['r_H']      = torch.reshape(econ.r_H, (n_cQ,n_x)).cpu().numpy()
    moments['r_F']      = torch.reshape(econ.r_F, (n_cQ,n_x)).cpu().numpy()
    moments['rc_H-r_H'] = torch.reshape(econ.rc_H-econ.r_H, (n_cQ,n_x)).cpu().numpy()
    moments['rc_F-r_F'] = torch.reshape(econ.rc_F-econ.r_F, (n_cQ,n_x)).cpu().numpy()
    moments['mu_q']     = torch.reshape(econ.mu_q, (n_cQ,n_x)).cpu().numpy()
    moments['mu_e']     = torch.reshape(econ.mu_e, (n_cQ,n_x)).cpu().numpy()
    moments['gs_e']     = torch.reshape(torch.sqrt(econ.gs2_e), (n_cQ,n_x)).cpu().numpy()
    moments['gs_H']     = torch.reshape(torch.sqrt(econ.gs2_H), (n_cQ,n_x)).cpu().numpy()
    moments['gs_F']     = torch.reshape(torch.sqrt(econ.gs2_F), (n_cQ,n_x)).cpu().numpy()
    moments['gs_M_H']   = torch.reshape(torch.sqrt(econ.gs2_M_H), (n_cQ,n_x)).cpu().numpy()
    moments['gs_M_F']   = torch.reshape(torch.sqrt(econ.gs2_M_F), (n_cQ,n_x)).cpu().numpy()
    moments['gs_MI_H']  = torch.reshape(torch.sqrt(econ.gs2_MI_H), (n_cQ,n_x)).cpu().numpy()
    moments['gs_MI_F']  = torch.reshape(torch.sqrt(econ.gs2_MI_F), (n_cQ,n_x)).cpu().numpy()
    moments['gs_c_H']   = torch.reshape(torch.sqrt(econ.gs2_c_H), (n_cQ,n_x)).cpu().numpy()
    moments['gs_c_F']   = torch.reshape(torch.sqrt(econ.gs2_c_F), (n_cQ,n_x)).cpu().numpy()
    moments['CRP_H']    = torch.reshape(econ.r_H-econ.r_F-econ.mu_e, (n_cQ,n_x)).cpu().numpy()
    
    moments['corr(dcH-dcF, de)'] = torch.reshape(econ.corr_ddc_de, (n_cQ,n_x)).cpu().numpy()
    moments['corr(dcH, dgH)']    = torch.reshape(econ.corr_dcH_dgH, (n_cQ,n_x)).cpu().numpy()
    moments['corr(dcF, dgF)']    = torch.reshape(econ.corr_dcF_dgF, (n_cQ,n_x)).cpu().numpy()
    moments['corr(dgH, dgF)']    = torch.reshape(econ.corr_dgH_dgF, (n_cQ,n_x)).cpu().numpy()
    moments['corr(dmH, dmF)']    = torch.reshape(econ.corr_dmH_dmF, (n_cQ,n_x)).cpu().numpy()
    moments['corr(dmIH, dmIF)']  = torch.reshape(econ.corr_dmIH_dmIF, (n_cQ,n_x)).cpu().numpy()
    moments['std(de)/std(dgH)']  = torch.reshape(econ.std_de_dgH, (n_cQ,n_x)).cpu().numpy()
    moments['std(de)/std(dgF)']  = torch.reshape(econ.std_de_dgF, (n_cQ,n_x)).cpu().numpy()
    moments['std(dcH)/std(dgH)'] = torch.reshape(econ.std_dcH_dgH, (n_cQ,n_x)).cpu().numpy()
    moments['std(dcF)/std(dgF)'] = torch.reshape(econ.std_dcF_dgF, (n_cQ,n_x)).cpu().numpy()
    
    # uncomment if you want to plot the figures
    # fig, axs = plt.subplots(6, 5, figsize=(20, 25))
    
    # v_states = [13, 28, 32, 36, 51]
    # for fig_item in range(5):
    #     axs[0,fig_item].plot(x[1:-1,0], moments['gs_e'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['gs_H'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['gs_F'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['gs_c_H'][1:-1, v_states[fig_item]], '.-',
    #                          x[1:-1,0], moments['gs_c_F'][1:-1, v_states[fig_item]], '.-')
    #     axs[0,fig_item].legend(['gs_e', 'gs_H', 'gs_F', 'gs_c_H', 'gs_c_F'])
    #     axs[0,fig_item].set_title('v = %.2f' % v_array[v_states[fig_item]])
    
    #     axs[1,fig_item].plot(x[1:-1,0], moments['r_H-r_F'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['mu_q'][1:-1, v_states[fig_item]], '.-',
    #                          x[1:-1,0], moments['mu_e'][1:-1, v_states[fig_item]], '.-',
    #                          x[1:-1,0], moments['CRP_H'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['SR_H'][1:-1, v_states[fig_item]])
    #     axs[1,fig_item].legend(['r_H-r_F', 'mu_q', 'mu_e', 'CRP_H', 'SR_H'])
    #     axs[1,fig_item].set_title('v = %.2f' % v_array[v_states[fig_item]])
        
    #     axs[2,fig_item].plot(x[1:-1,0], moments['corr(dcH-dcF, de)'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['corr(dcH, dgH)'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['corr(dcF, dgF)'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['std(de)/std(dgH)'][1:-1, v_states[fig_item]], '.-',
    #                          x[1:-1,0], moments['std(dcH)/std(dgH)'][1:-1, v_states[fig_item]], '.-',
    #                          )
    #     axs[2,fig_item].legend(['corr(dcH-dcF, de)', 'corr(dcH, dgH)', 'corr(dcF, dgF)', 'std(de)/std(dgH)', 'std(dcH)/std(dgH)'])
    #     axs[2,fig_item].set_title('v = %.2f' % v_array[v_states[fig_item]])
        
    #     axs[3,fig_item].plot(x[1:-1,0], moments['rc_H-r_H'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['rc_F-r_F'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['gs_M_H'][1:-1, v_states[fig_item]], '.-',
    #                          x[1:-1,0], moments['gs_M_F'][1:-1, v_states[fig_item]], '.-',
    #                          )
    #     axs[3,fig_item].legend(['rc_H-r_H', 'rc_F-r_F', 'gs_M_H', 'gs_M_F'])
    #     axs[3,fig_item].set_title('v = %.2f' % v_array[v_states[fig_item]])
        
    #     axs[4,fig_item].plot(
    #                          x[1:-1,0], moments['gs_MI_H'][1:-1, v_states[fig_item]], '*-',
    #                          x[1:-1,0], moments['gs_MI_F'][1:-1, v_states[fig_item]], '*-',
    #                          x[1:-1,0], moments['corr(dmH, dmF)'][1:-1, v_states[fig_item]], 'o-',
    #                          x[1:-1,0], moments['corr(dmIH, dmIF)'][1:-1, v_states[fig_item]], 'o-')
    #     axs[4,fig_item].legend(['gs_MI_H', 'gs_MI_F', 'corr(dmH, dmF)', 'corr(dmIH, dmIF)'])
    #     axs[4,fig_item].set_title('v = %.2f' % v_array[v_states[fig_item]])
        
    #     axs[5,fig_item].plot(x[1:-1,0], moments['r_H'][1:-1, v_states[fig_item]], '--',
    #                          x[1:-1,0], moments['r_F'][1:-1, v_states[fig_item]], '--'
    #                          )
    #     axs[5,fig_item].legend(['r_H', 'r_F'])
    #     axs[5,fig_item].set_title('v = %.2f' % v_array[v_states[fig_item]])
    return (x, moments)

def make_plots(item_name, is_frictional, paras):
    econ          = get_econ_wrapper(is_frictional, paras) 
    lncQ, moments = econ_plot_wrapper(econ)
    x_states      = [20, 24, 26, 32]
    
    y1 =  moments[item_name][1:-1, x_states[0]]*1e4
    y2 =  moments[item_name][1:-1, x_states[1]]*1e4
    y3 =  moments[item_name][1:-1, x_states[2]]*1e4
    y4 =  moments[item_name][1:-1, x_states[3]]*1e4
    if item_name == 'rc_F-r_F':
        y1 = -y1
        y2 = -y2
        y3 = -y3
        y4 = -y4
    x  = lncQ[1:-1,0]
    # Create a DataFrame
    data = pd.DataFrame({
        'x': np.tile(x, 4),
        'y': np.concatenate([y1, y2, y3, y4]),
        'label': np.repeat(['$|x_t|=3/8$', '$|x_t|=1/4$', '$|x_t|=3/16$', '$|x_t|=0$'], len(y1))
    })

    # Plot the curves
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=data, x='x', y='y', hue='label', style='label', markers=True, dashes=False)

    # Customize the plot
    plt.xlabel(r'$\log \mathcal{Q}$')
    match item_name:
        case 'r_H-r_F':
            plt.ylabel('$r_H-r_F$ (basis points)')
        case 'CRP_H':
            plt.ylabel(r'$r_H-r_F - E_t[d\log\mathcal{E}]/dt$ (basis points)')
        case 'mu_e':
            plt.ylabel(r'$E_t[d\log\mathcal{E}]/dt$ (basis points)')
        case 'rc_F-r_F':
            plt.ylabel('CIP$_H$ (basis points)')

    # Remove grey grids and borders
    sns.despine(offset=10, trim=True)
    plt.grid(False)

    # Place the legend at the bottom right outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

    # Adjust layout to make room for the legend
    plt.tight_layout()
