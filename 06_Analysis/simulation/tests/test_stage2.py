"""Independent physical/transfer checks plus DOE design checks."""
import sys
import unittest
from pathlib import Path
import json
import numpy as np
from scipy import signal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from doe import factorial, face_centered, features, contrasts, decode
from model import PendulumParameters, continuous_matrices, exact_discretization
from observer import simulate_matched_model, ackermann_observer_gain
from stage2_system import plant, nonlinear_plant, estimate, observer_system, force_frequency_response


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.p=PendulumParameters(.00215194497,.00059676098,.01178181,.245)
        self.dt=.01

    def test_design_rank_and_unique_points(self):
        z=face_centered();x,_=features(z)
        self.assertEqual(z.shape,(43,5))
        self.assertEqual(len(np.unique(z,axis=0)),43)
        self.assertEqual(np.linalg.matrix_rank(x),21)
        f=factorial();np.testing.assert_allclose(f.T@f,32*np.eye(5))

    def test_known_interactions_are_recovered(self):
        z=factorial();r=4+2*z[:,0]+3*z[:,1]*z[:,3]
        e=contrasts(r)
        self.assertAlmostEqual(e['0'],4)
        self.assertAlmostEqual(e['1:3'],6)
        self.assertAlmostEqual(e['2'],0)

    def test_nonlinear_factor_coding_endpoints(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'config/stage2_doe.json').read_text())
        np.testing.assert_allclose(decode(np.zeros(5),config),np.ones(5))
        np.testing.assert_allclose(decode(-np.ones(5),config),[.75,.5,.9,.95,.5])
        np.testing.assert_allclose(decode(np.ones(5),config),[1.25,1.5,1.1,1.05,2])

    def test_plant_static_equilibrium(self):
        f=np.full(15001,.0008)
        x=plant(f,self.p,self.dt)
        self.assertAlmostEqual(x[-1,0],self.p.force_lever_m*f[0]/self.p.restoring_n_m_per_rad,places=8)

    def test_nonlinear_plant_static_equilibrium(self):
        f=np.full(30001,.0008)
        x=nonlinear_plant(f,self.p,self.dt)
        expected=np.arctan(self.p.force_lever_m*f[0]/self.p.restoring_n_m_per_rad)
        self.assertAlmostEqual(x[-1,0],expected,places=6)

    def test_mismatch_changes_estimate_not_plant(self):
        f=np.full(15001,.0008);x=plant(f,self.p,self.dt)
        results=estimate(x[:,0],self.p,[1.25,.5,1.1,.95,1.],self.dt)
        target=f[-1]*1.1/.95
        for y in results.values():
            self.assertLess(abs(y[-1]/target-1),1e-6)

    def test_transfer_filter_matches_stage1_recurrence(self):
        a,c=continuous_matrices(self.p);ad=exact_discretization(a,self.dt)
        ld=ackermann_observer_gain(ad,c,np.full(3,np.exp(-2*np.pi*self.dt)))
        x,xhat,_=simulate_matched_model(ad,c,ld,np.array([.03,.02,.001]),np.zeros(3),2001)
        y=estimate(x[:,0],self.p,np.ones(5),self.dt)['Observer']
        np.testing.assert_allclose(y,xhat[:,2]/self.p.force_lever_m,atol=1e-9,rtol=1e-7)

    def test_discrete_frequency_response_matches_time_domain(self):
        t=np.arange(20001)*self.dt;frequency=.4
        force=.0008*np.cos(2*np.pi*frequency*t)
        x=plant(force,self.p,self.dt)
        ratios=np.array([1.15,.8,.95,1.03,1.2])
        outputs=estimate(x[:,0],self.p,ratios,self.dt)
        transfer=force_frequency_response([frequency],self.p,ratios,self.dt)
        keep=t>=100
        fit=np.column_stack([np.cos(2*np.pi*frequency*t[keep]),np.sin(2*np.pi*frequency*t[keep]),np.ones(keep.sum())])
        for name,y in outputs.items():
            beta=np.linalg.lstsq(fit,y[keep],rcond=None)[0]
            measured=(beta[0]-1j*beta[1])/.0008
            self.assertLess(abs(measured-transfer[name][0]),1e-5)

    def test_no_current_or_future_angle_leaks(self):
        a=np.zeros(100);b=a.copy();b[50]=.1
        ya=estimate(a,self.p,np.ones(5),self.dt)['Observer']
        yb=estimate(b,self.p,np.ones(5),self.dt)['Observer']
        np.testing.assert_allclose(ya[:51],yb[:51],atol=1e-12)
        self.assertGreater(abs(yb[51]),1e-6)

if __name__=='__main__':unittest.main()
