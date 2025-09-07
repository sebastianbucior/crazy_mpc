from casadi import SX, vertcat, horzcat, diag, inv_minor, cross, sqrt, cos, sin
import numpy as np

from acados_template import AcadosModel

class QuadrotorFull:
    def __init__(self, mass, arm_length, Ixx, Iyy, Izz, cm, tau, thrust_constant, moment_constant, gravity=9.80665):
        self.mass = mass
        self.gravity = gravity
        self.arm_length = arm_length
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.cm = cm
        self.tau = tau
        self.Ct = thrust_constant
        self.Cm = moment_constant

    def model(self):
        px = SX.sym('px')  # type: ignore
        py = SX.sym('py')  # type: ignore
        pz = SX.sym('pz')  # type: ignore

        qw = SX.sym('qw')  # type: ignore
        qx = SX.sym('qx')  # type: ignore
        qy = SX.sym('qy')  # type: ignore
        qz = SX.sym('qz')  # type: ignore

        vx = SX.sym('vx')  # type: ignore
        vy = SX.sym('vy')  # type: ignore
        vz = SX.sym('vz')  # type: ignore

        wx = SX.sym('wx')  # type: ignore
        wy = SX.sym('wy')  # type: ignore
        wz = SX.sym('wz')  # type: ignore

        # -- conctenated vector
        x = vertcat(
            px, py, pz,
            qw, qx, qy, qz,
            vx, vy, vz,
            wx, wy, wz
        )

        r1 = SX.sym('r1') # type: ignore
        r2 = SX.sym('r2') # type: ignore
        r3 = SX.sym('r3') # type: ignore
        r4 = SX.sym('r4') # type: ignore

        u = vertcat(r1, r2, r3, r4)

        # Zamiana KF na Ct, KM na Cm, d na arm_length
        thrust = self.Ct * (r1 ** 2 + r2 ** 2 + r3 ** 2 + r4 ** 2) / self.mass
        Mx = self.arm_length * self.Ct / sqrt(2) * (-r1 ** 2 - r2 ** 2 + r3 ** 2 + r4 ** 2)
        My = self.arm_length * self.Ct / sqrt(2) * (-r1 ** 2 + r2 ** 2 + r3 ** 2 - r4 ** 2)
        Mz = self.Cm * (-r1 ** 2 + r2 ** 2 - r3 ** 2 + r4 ** 2)

        # Przypisanie pochodnych do zmiennych odpowiadających stanom w _x
        dpx = vx
        dpy = vy
        dpz = vz
        dqw = 0.5 * (-wx * qx - wy * qy - wz * qz)
        dqx = 0.5 * (wx * qw + wz * qy - wy * qz)
        dqy = 0.5 * (wy * qw - wz * qx + wx * qz)
        dqz = 0.5 * (wz * qw + wy * qx - wx * qy)
        dvx = 2 * (qw * qy + qx * qz) * thrust
        dvy = 2 * (qy * qz - qw * qx) * thrust
        dvz = (qw * qw - qx * qx - qy * qy + qz * qz) * thrust - self.gravity
        dwx = (Mx + self.Izz * wy * wz - self.Iyy * wy * wz) / self.Ixx
        dwy = (My + self.Ixx * wz * wx - self.Izz * wx * wz) / self.Iyy
        dwz = (Mz + self.Iyy * wx * wy - self.Ixx * wy * wx) / self.Izz

        f_expl = vertcat(
            dpx, dpy, dpz,
            dqw, dqx, dqy, dqz,
            dvx, dvy, dvz,
            dwx, dwy, dwz
        )

        px_dot = SX.sym('px_dot')  # type: ignore
        py_dot = SX.sym('py_dot')  # type: ignore
        pz_dot = SX.sym('pz_dot')  # type: ignore
        qw_dot = SX.sym('qw_dot')  # type: ignore
        qx_dot = SX.sym('qx_dot')  # type: ignore
        qy_dot = SX.sym('qy_dot')  # type: ignore
        qz_dot = SX.sym('qz_dot')  # type: ignore
        vx_dot = SX.sym('vx_dot')  # type: ignore
        vy_dot = SX.sym('vy_dot')  # type: ignore
        vz_dot = SX.sym('vz_dot')  # type: ignore
        wx_dot = SX.sym('wx_dot')  # type: ignore
        wy_dot = SX.sym('wy_dot')  # type: ignore
        wz_dot = SX.sym('wz_dot')  # type: ignore

        x_dot = vertcat(
            px_dot, py_dot, pz_dot,
            qw_dot, qx_dot, qy_dot, qz_dot,
            vx_dot, vy_dot, vz_dot,
            wx_dot, wy_dot, wz_dot
        )

        f_impl = x_dot - f_expl

        z = []
        p = []

        model = AcadosModel()
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.z = z
        model.p = p
        model.name = 'quadrotor_full'

        return model