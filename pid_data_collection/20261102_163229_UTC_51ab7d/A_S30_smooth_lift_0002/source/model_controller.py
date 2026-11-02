"""Inverse-dynamics compensation + fixed PID acceleration feedback.
Cable remains unilateral and capped. No reserve actuator or prescribed elbow motion.
"""
import math
import numpy as np

class ModelPID:
    def __init__(self,o,model,elbow,cable,cfg):
        self.o=o;self.model=model;self.elbow=elbow;self.cable=cable;self.cfg=cfg
        self.solver=o.InverseDynamicsSolver(model)
        self.integral=0.
        # build() asserts shoulder/elbow mobilizer order. Shoulder is locked.
        if model.getNumSpeeds()!=2:raise ValueError('Expected two generalized speeds.')

    def step(self,state,qr,vr,ar,dt,arm):
        o=self.o;model=self.model;cfg=self.cfg
        q=self.elbow.getValue(state);v=self.elbow.getSpeedValue(state)
        error=qr-q;vel_error=vr-v
        integral_used=self.integral
        p=cfg['accel_kp']*error;d=cfg['accel_kd']*vel_error;i=cfg['accel_ki']*self.integral
        acceleration=ar+p+d+i
        # Removing cable force only in the copied state prevents double counting.
        probe=o.State(state);self.cable.setOverrideActuation(probe,0.)
        model.realizeDynamics(probe)
        udot=o.Vector(model.getNumSpeeds(),0.);udot.set(1,acceleration)
        residual=self.solver.solve(probe,udot)
        torque=float(residual.get(1));arm=float(arm)
        if arm<=cfg['minimum_moment_arm_m']:raise RuntimeError('Insufficient positive cable moment arm.')
        requested=torque/arm;force=float(np.clip(requested,0.,cfg['force_cap_N']))
        saturated=abs(force-requested)>1e-7
        if not ((requested>cfg['force_cap_N'] and error>0) or (requested<0 and error<0)):
            self.integral=float(np.clip(self.integral+error*dt,-cfg['integral_limit_rad_s'],cfg['integral_limit_rad_s']))
        return dict(force_N=force,requested_force_N=requested,torque_Nm=torque,
            desired_acceleration_rad_s2=acceleration,P_accel=p,I_accel=i,D_accel=d,
            error_integral_rad_s=integral_used,saturated=saturated)


def path_moment_arms(o,model,state,elbow,cable,muscles,delta=1e-5):
    """Signed -d(path length)/dq with shoulder held at its actual fixed position.
    Avoids OpenSim MomentArmSolver's constraint/cache failure at nonzero locked
    shoulder angles. Central differences apply to original cable/muscle paths.
    """
    q=elbow.getValue(state);lengths=[]
    for sign in [-1.,1.]:
        probe=o.State(state);elbow.setValue(probe,q+sign*delta,False)
        model.realizePosition(probe)
        lengths.append([cable.getLength(probe)]+[muscles.get(i).getLength(probe) for i in range(muscles.getSize())])
    return -(np.array(lengths[1])-np.array(lengths[0]))/(2*delta)
