"""Independent constrained-system and conductor-allocation checks, no data reads."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'mast')]
from calib_joint import sufficient
from simple import coil_groups, P_NAMES, SOL
import numpy as np
from scipy.linalg import null_space, svd


def joint_check(free_c):
    # Compare projection with a dense KKT solve in the original J variables.
    rng = np.random.default_rng(42)
    nt,npix,ns = 4,31,5
    g = rng.normal(size=(npix,ns))
    f,e,y = [rng.normal(size=(nt,npix)) for _ in range(3)]
    ip = rng.uniform(.3,1.4,nt)
    q = null_space(np.ones((1,ns)))
    a = g@q
    if free_c:
        a = np.column_stack((a,np.ones(npix)))
    u,s,vt = svd(a,full_matrices=False)
    pinv = (vt.T/s)@u.T
    x = f+ip[:,None]*g.mean(axis=1)
    gram,rhs,_,responses = sufficient(y,x,e,u,pinv)
    projection = np.linalg.solve(gram,rhs)
    local_size = ns+int(free_c)
    design = np.zeros((nt*npix,2+nt*local_size))
    design[:,0] = f.ravel(); design[:,1] = e.ravel()
    constraint = np.zeros((nt,design.shape[1]))
    for t in range(nt):
        j = 2+t*local_size
        design[t*npix:(t+1)*npix,j:j+ns] = g
        if free_c:
            design[t*npix:(t+1)*npix,j+ns] = 1.
        constraint[t,0] = -ip[t]
        constraint[t,j:j+ns] = 1.
    kkt = np.block([[design.T@design,constraint.T],
                    [constraint,np.zeros((nt,nt))]])
    dense = np.linalg.solve(kkt,np.r_[design.T@y.ravel(),np.zeros(nt)])[:design.shape[1]]
    np.testing.assert_allclose(projection,dense[:2],atol=1e-10)
    coeff = responses[0]-projection[0]*responses[1]-projection[1]*responses[2]
    j = projection[0]*ip[:,None]/ns+coeff[:,:ns-1]@q.T
    np.testing.assert_allclose(j.sum(axis=1),projection[0]*ip,atol=1e-12)
    pred = projection[0]*f+projection[1]*e+j@g.T
    if free_c:
        pred += coeff[:,-1,None]
    np.testing.assert_allclose(pred.ravel(),design@dense,atol=1e-10)


def allocation_check():
    # Splitting one conductor into duplicate geometry must not multiply current.
    groups = P_NAMES+[SOL]
    row = dict(coil_input_column=groups,coil_R=np.full(11,.5),coil_Z=np.full(11,.4),
               coil_width=np.full(11,.02),coil_height=np.full(11,.03),
               coil_angle1=np.zeros(11),coil_angle2=np.zeros(11),
               efit_grid_R=np.array([.3,.7]),efit_grid_Z=np.array([-.2,.2]))
    g,_ = coil_groups(row)
    split = {k:(v*2 if isinstance(v,list) else np.tile(v,2))
             if k.startswith('coil_') else v for k,v in row.items()}
    g2,checks = coil_groups(split)
    np.testing.assert_allclose(g,g2,rtol=1e-13,atol=1e-20)
    assert all(np.isclose(c['allocation_sum'],1) for c in checks.values())


if __name__=='__main__':
    joint_check(True); joint_check(False); allocation_check()
    print('PASS: j1/j2 match independent dense equality-constrained solve; conductor duplication invariant')
