from random import randint, choice
from scipy.optimize import minimize
from myscipy_optimize import _minimize_cg, _minimize_bfgs, _minimize_tpgd
from LJ import LJ, LJ_force
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def single_optimize(pos, dim=3, kt=0.5, mu=0.1, beta=1.001, method='CG'):
    """
    perform optimization for a given cluster
    Args: 
    pos: N*dim0 array representing the atomic positions
    dim: dimension of the hyper/normal space
    kt: perturbation factors

    output:
    energy: optmized energy
    pos: optimized positions
    """
    N_atom = len(pos)
    diff = dim - np.shape(pos)[1]
    if diff > 0:
        extra = kt*(np.random.uniform(-1, 1, (N_atom, diff)))
        pos = np.hstack((pos, extra))
    elif diff < 0:
        pos = pos[:, :dim]

    pos = pos.flatten()
    res = _minimize_tpgd(LJ, pos, args=(dim, mu), jac=LJ_force, gtol=1e-4, maxiter=20)
    pos = res.x
    if method == 'mycg':
        res = _minimize_cg(LJ, pos, args=(dim, mu), jac=LJ_force, gtol=1e-4, maxiter=500)
        res = _minimize_cg(LJ, res.x, args=(dim, mu), jac=LJ_force, gtol=1e-4, maxiter=500)
        res = _minimize_cg(LJ, res.x, args=(dim, mu), jac=LJ_force, gtol=1e-4)
    elif method== 'mybfgs':
        res = _minimize_bfgs(LJ, pos, args=(dim, mu), jac=LJ_force, gtol=1e-4, maxiter=500)
        res = _minimize_bfgs(LJ, res.x, args=(dim, mu), jac=LJ_force, gtol=1e-4, maxiter=500)
        res = _minimize_bfgs(LJ, res.x, args=(dim, mu), jac=LJ_force, gtol=1e-4)
    elif method == 'mytpgd':
        res = _minimize_tpgd(LJ, pos, args=(dim, mu), jac=LJ_force, gtol=1e-4, maxiter=500)
        res = _minimize_tpgd(LJ, res.x, args=(dim, mu), jac=LJ_force, gtol=1e-4, maxiter=500)
        res = _minimize_tpgd(LJ, res.x, args=(dim, mu), jac=LJ_force, gtol=1e-4)
    else:
        res = minimize(LJ, pos, args=(dim, mu), jac=LJ_force, method=method, tol=1e-4)
    energy = res.fun
    pos = np.reshape(res.x, (N_atom, dim))
    return energy, pos, res.nit

def one_step(pos):
    eng = []
    for method in ['mycg', 'mybfgs', 'mytpgd']:
        pos1 = pos.copy()
        energy1, pos1, it1 = single_optimize(pos1, dim=3, method=method)
        pos2 = pos.copy()
        energy2, pos2, it2 = single_optimize(pos2, dim=4, mu=10, kt=1, method=method)
        energy2, pos2, it2 = single_optimize(pos2, dim=3, method=method)
        print('Optmization {:10s}  3D: {:10.4f}  {:6d} 4D: {:10.4f} {:6d}'.format(method, energy1, it1, energy2, it2))
        eng.append([energy1, energy2])
    eng = np.array(eng)
    return eng.flatten()

if __name__ == "__main__":
    #np.random.seed(10)
    N = 10000
    clusters = []
    for i in range(N):
        pos = np.loadtxt('data/LJ-38.txt')
        pos += np.random.uniform(-1, 1, (len(pos), 3))
        clusters.append(pos)

    ncpu = 20
    if ncpu > 1:
        from multiprocessing import Pool
        with Pool(ncpu) as p:
            engs = p.map(one_step, clusters)
            #p.close()
            #p.join()
    else:
        engs = []
        for pos in clusters:
            engs.append(one_step(pos))

    engs = np.array(engs)
    eng_min = -173.92
    eng_cg3, eng_cg4 = engs[:,0], engs[:,1]
    eng_bf3, eng_bf4 = engs[:,2], engs[:,3]
    eng_gd3, eng_gd4 = engs[:,4], engs[:,5]
    ground_cg3 = len(eng_cg3[eng_cg3-1e-2<eng_min])
    ground_cg4 = len(eng_cg4[eng_cg4-1e-2<eng_min])
    ground_bf3 = len(eng_bf3[eng_bf3-1e-2<eng_min])
    ground_bf4 = len(eng_bf4[eng_bf4-1e-2<eng_min])
    ground_gd3 = len(eng_gd3[eng_gd3-1e-2<eng_min])
    ground_gd4 = len(eng_gd4[eng_gd4-1e-2<eng_min])

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.switch_backend('agg')
    plt.style.use("bmh")

    bins = np.linspace(eng_min-0.1, eng_min+20, 100)
    fig = plt.figure(figsize=(9,10))

    gs=GridSpec(3,2)
    ax1=fig.add_subplot(gs[0,0]) 
    ax2=fig.add_subplot(gs[1,0]) 
    ax3=fig.add_subplot(gs[2,0]) 

    label1 = '3d CG: ' + str(ground_cg3) + '/' + str(len(eng_cg3))
    label2 = '4d CG: ' + str(ground_cg4) + '/' + str(len(eng_cg4))
    ax1.hist(eng_cg3, bins, alpha=0.5, label=label1)
    ax1.hist(eng_cg4, bins, alpha=0.5, label=label2)
    ax1.set_xlim([eng_min-0.1, eng_min+20])
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Counts')
    ax1.legend()

    label1 = '3d BFGS: ' + str(ground_bf3) + '/' + str(len(eng_bf3))
    label2 = '4d BFGS: ' + str(ground_bf4) + '/' + str(len(eng_bf4))
    ax2.hist(eng_bf3, bins, alpha=0.5, label=label1)
    ax2.hist(eng_bf4, bins, alpha=0.5, label=label2)
    ax2.set_xlim([eng_min-0.1, eng_min+20])
    ax2.set_xlabel('Energy (eV)')
    #ax2.set_ylabel('Counts')
    ax2.legend()

    label1 = '3d TPGD: ' + str(ground_gd3) + '/' + str(len(eng_gd3))
    label2 = '4d TPGD: ' + str(ground_gd4) + '/' + str(len(eng_gd4))
    ax3.hist(eng_gd3, bins, alpha=0.5, label=label1)
    ax3.hist(eng_gd4, bins, alpha=0.5, label=label2)
    ax3.set_xlim([eng_min-0.1, eng_min+20])
    ax3.set_xlabel('Energy (eV)')
    ax3.set_ylabel('Counts')
    ax3.legend()

    ax4=fig.add_subplot(gs[0,1]) 
    ax5=fig.add_subplot(gs[1,1]) 
    ax6=fig.add_subplot(gs[2,1]) 
    x = np.linspace(eng_min, eng_min+20, 10)

    ax4.plot(x, x, 'k-', lw=1)
    ax4.scatter(eng_cg3, eng_cg4, label='CG', s=5)
    ax4.set_xlabel('3D optimization (eV)')
    ax4.set_ylabel('4D optimization (eV)')
    ax4.set_xlim([eng_min-0.1, eng_min+20])
    ax4.set_ylim([eng_min-0.1, eng_min+20])
    ax4.legend()

    ax5.plot(x, x, 'k-', lw=1)
    ax5.scatter(eng_bf3, eng_bf4, label='BFGS', s=5)
    ax5.set_xlabel('3D optimization (eV)')
    ax5.set_ylabel('4D optimization (eV)')
    ax5.set_xlim([eng_min-0.1, eng_min+20])
    ax5.set_ylim([eng_min-0.1, eng_min+20])
    ax5.legend()

    ax6.scatter(eng_gd3, eng_gd4, label='TPGD', s=5)
    ax6.plot(x, x, 'k-', lw=1)
    ax6.set_xlabel('3D optimization (eV)')
    ax6.set_ylabel('4D optimization (eV)')
    ax6.set_xlim([eng_min-0.1, eng_min+20])
    ax6.set_ylim([eng_min-0.1, eng_min+20])
    ax6.legend()
    plt.tight_layout()
    plt.savefig('comparison.png')
