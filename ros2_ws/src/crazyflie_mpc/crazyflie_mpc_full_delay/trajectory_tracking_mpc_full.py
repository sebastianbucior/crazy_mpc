import os
import pathlib
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from casadi import SX, vertcat, horzcat, diag, inv_minor, cross, sqrt,  cos, sin, norm_2, tanh, GenMX_zeros
from scipy.linalg import block_diag
import yaml

from ament_index_python.packages import get_package_share_directory
from crazyflie_mpc_full.quadrotor_full_model import QuadrotorFull
from threading import Thread
from time import sleep, time
from pathlib import Path
import importlib
import sys

import tf_transformations

class TrajectoryTrackingMpc:
    def __init__(self, name: str, quadrotor: QuadrotorFull, horizon: float, num_steps: int, code_export_directory : Path=Path('acados_generated_files')):
        self.model_name = name
        self.quad = quadrotor
        self.horizon = horizon
        self.num_steps = num_steps
        self.ocp_solver = None
        self.solver_locked = False
        self.hover_control = np.array([1962.6, 1962.6, 1962.6, 1962.6]) # [rpm]
        # self.acados_generated_files_path = Path(__file__).parent.resolve() / 'acados_generated_files'
        self.acados_generated_files_path = code_export_directory

        self.generate_mpc()

        try:
            if self.acados_generated_files_path.is_dir():
                sys.path.append(str(self.acados_generated_files_path))
            acados_ocp_solver_pyx = importlib.import_module('c_generated_code.acados_ocp_solver_pyx')
            self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)
            print('Acados cython module imported successfully.')
        except ImportError:
            print('Acados cython code not generated. Generating cython code now...')
            self.generate_mpc()
    
    def __copy__(self):
        return type(self)(self.model_name, self.quad, self.horizon, self.num_steps, self.acados_generated_files_path)
    
    def generate_mpc(self):
        model = self.quad.model()

        # Define the optimal control problem
        ocp = AcadosOcp()
        ocp.model = model

        ocp.code_export_directory = (str)(self.acados_generated_files_path / ('c_generated_code'))
        nx = model.x.size()[0] # number of states
        nu = model.u.size()[0] # number of controls
        ny = nx + nu  # size of intermediate cost reference vector in least squares objective
        ny_e = nx # size of terminal reference vector

        N = self.num_steps
        Tf = self.horizon
        ocp.dims.N = N
        ocp.solver_options.tf = Tf
  
        Q = np.eye(nx)
        Q[0,0] = 100.0      # x
        Q[1,1] = 100.0      # y
        Q[2,2] = 100.0      # z
        Q[3,3] = 1.0e-1     # qw
        Q[4,4] = 1.0e-1     # qx
        Q[5,5] = 1.0e-1     # qy
        Q[6,6] = 1.0e-1     # qz
        Q[7,7] = 5       # vbx
        Q[8,8] = 5        # vby
        Q[9,9] = 5        # vbz
        Q[10,10] = 1e-2     # wx
        Q[11,11] = 1e-2     # wy
        Q[12,12] = 1e-1     # wz

        R = np.eye(nu)
        R[0,0] = 1e-5    # w1
        R[1,1] = 1e-5    # w2
        R[2,2] = 1e-5    # w3
        R[3,3] = 1e-5    # w4

        W = block_diag(Q,R)

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.Vx = np.vstack([np.identity(nx), np.zeros((nu,nx))])
        ocp.cost.Vu = np.vstack([np.zeros((nx,nu)), np.identity(nu)])
        ocp.cost.W = W
        ocp.cost.yref = np.zeros(ny)

        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W_e = 1000*Q
        ocp.cost.Vx_e = np.vstack([np.identity(nx)])
        ocp.cost.yref_e = np.zeros(ny_e)
  
        # bounds on control
        max_angle = np.radians(15) # [rad]
        max_thrust = 0.477627618 # [N]

        ocp.constraints.lbu = np.array([0, 0, 0, 0.])
        ocp.constraints.ubu = np.array([3052, 3052, 3052, 3052])
        ocp.constraints.idxbu = np.array([0,1,2,3])

        # max_height = 4.0
        # x_bound = np.inf
        # ocp.constraints.lbx = np.array([-x_bound for _ in range(nx)])
        # ocp.constraints.ubx = np.array([+x_bound for _ in range(nx)])
        # ocp.constraints.idxbx = np.array([0,1,2,3,4,5,6,7,8])

        # initial state
        ocp.constraints.x0 = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0])

        json_file = str(self.acados_generated_files_path / ('acados_ocp.json'))
        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.tol = 1e-3
        # ocp.solver_options.qp_tol = 1e-3
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.qp_solver_iter_max = 50
        ocp.solver_options.print_level = 0
        # ocp.solver_options.timeout_max_time = 0.015
        
        AcadosOcpSolver.generate(ocp, json_file=json_file)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)

        if self.acados_generated_files_path.is_dir():
            sys.path.append(str(self.acados_generated_files_path))
        acados_ocp_solver_pyx = importlib.import_module('c_generated_code.acados_ocp_solver_pyx')
        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)

    def generate_mpc_1(self):
        model = self.quad.model()
    
        ocp = AcadosOcp()
        ocp.model = model

        ocp.code_export_directory = (str)(self.acados_generated_files_path / ('c_generated_code'))
        nx = model.x.size()[0] # number of states
        nu = model.u.size()[0] # number of controls
        ny = nx + nu  # size of intermediate cost reference vector in least squares objective
        ny_e = nx # size of terminal reference vector

        N = self.num_steps
        Tf = self.horizon
        ocp.dims.N = N
        ocp.solver_options.tf = Tf

        nlp_cost = ocp.cost
        Q = np.eye(nx)
        Q[0,0] = 100.0      # x
        Q[1,1] = 100.0      # y
        Q[2,2] = 100.0      # z
        Q[3,3] = 1.0e-3     # qw
        Q[4,4] = 1.0e-3     # qx
        Q[5,5] = 1.0e-3     # qy
        Q[6,6] = 1.0e-3     # qz
        Q[7,7] = 7e-1       # vbx
        Q[8,8] = 1.0        # vby
        Q[9,9] = 4.0        # vbz
        Q[10,10] = 1e-5     # wx
        Q[11,11] = 1e-5     # wy
        Q[12,12] = 10.0     # wz

        R = np.eye(nu)
        R[0,0] = 0.06    # w1
        R[1,1] = 0.06    # w2
        R[2,2] = 0.06    # w3
        R[3,3] = 0.06    # w4

        nlp_cost.W = block_diag(Q, R)

        Vx = np.zeros((ny, nx))
        Vx[0,0] = 1.0
        Vx[1,1] = 1.0
        Vx[2,2] = 1.0
        Vx[3,3] = 1.0
        Vx[4,4] = 1.0
        Vx[5,5] = 1.0
        Vx[6,6] = 1.0
        Vx[7,7] = 1.0
        Vx[8,8] = 1.0
        Vx[9,9] = 1.0
        Vx[10,10] = 1.0
        Vx[11,11] = 1.0
        Vx[12,12] = 1.0
        nlp_cost.Vx = Vx

        Vu = np.zeros((ny, nu))
        Vu[13,0] = 1.0
        Vu[14,1] = 1.0
        Vu[15,2] = 1.0
        Vu[16,3] = 1.0
        nlp_cost.Vu = Vu

        nlp_cost.W_e = Q

        Vx_e = np.zeros((ny_e, nx))
        Vx_e[0,0] = 1.0
        Vx_e[1,1] = 1.0
        Vx_e[2,2] = 1.0
        Vx_e[3,3] = 1.0
        Vx_e[4,4] = 1.0
        Vx_e[5,5] = 1.0
        Vx_e[6,6] = 1.0
        Vx_e[7,7] = 1.0
        Vx_e[8,8] = 1.0
        Vx_e[9,9] = 1.0
        Vx_e[10,10] = 1.0
        Vx_e[11,11] = 1.0
        Vx_e[12,12] = 1.0

        nlp_cost.Vx_e = Vx_e
        nlp_cost.yref   = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        nlp_cost.yref_e = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        nlp_con = ocp.constraints

        nlp_con.lbu = np.array([0,0,0,0])
        nlp_con.ubu = np.array([+3052,+3052,+3052,+3052])
        nlp_con.x0  = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0])
        nlp_con.idxbu = np.array([0, 1, 2, 3])

        json_file = str(self.acados_generated_files_path / ('acados_ocp.json'))
        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.integrator_type = 'ERK'

        ocp.solver_options.print_level = 0
        
        AcadosOcpSolver.generate(ocp, json_file=json_file)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)

        if self.acados_generated_files_path.is_dir():
            sys.path.append(str(self.acados_generated_files_path))
        acados_ocp_solver_pyx = importlib.import_module('c_generated_code.acados_ocp_solver_pyx')
        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)

    def solve_mpc(self, x0, yref, yref_e, last_u, solution_callback=None):
        # print("Pos:", x0[:3])
        # print("Ref pos:", yref[:3,1])


        if self.ocp_solver is None:
            raise RuntimeError("Solver nie jest zainicjalizowany/generowany.")

        if self.solver_locked:
            # print('mpc solver locked, skipping...')
            return
        self.solver_locked = True

        N = self.num_steps
        nx = len(x0)
        nu = 4
        
        if yref.shape[1] != self.num_steps:
            raise Exception('incorrect size of yref')
    
        for i in range(N):
            self.ocp_solver.set(i, 'yref', np.array([*yref[:,i], *self.hover_control]))
            # self.ocp_solver.set(i, 'x', yref[:,i])
            # self.ocp_solver.set(i, 'u', last_u[i,:])
            self.ocp_solver.set(i, 'u', self.hover_control)
        
        self.ocp_solver.set(N, 'yref', yref_e)

        x_mpc = np.zeros((N+1, nx))
        u_mpc = np.zeros((N, nu))
        r_mpc = np.zeros((N, nu))
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        
        status = self.ocp_solver.solve()

        # print(f'MPC solver status: {status}')

        # extract state and control solution from solver
        for i in range(N):
            x_mpc[i,:] = self.ocp_solver.get(i, "x")

            # Convert quaternion (x_mpc[i,3:7]) to roll, pitch, yaw
            q = self.ocp_solver.get(i+1, "x")[3:7]
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            # Pitch (y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            if abs(sinp) >= 1:
                pitch = np.pi/2 * np.sign(sinp)
            else:
                pitch = np.arcsin(sinp)
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            # Optionally, store or print roll, pitch, yaw
            # For example:
            # print(f"Step {i}: roll={roll}, pitch={pitch}, yaw={yaw}")

            roll, pitch, yaw = tf_transformations.euler_from_quaternion([qx,
                                                                  qy,
                                                                  qz,
                                                                  qw], axes='rxyz')

            r1, r2, r3, r4 = self.ocp_solver.get(i, "u")
            mass = 0.0282
            motorConstant = 1.7965e-8
            thrust = motorConstant * (r1 ** 2 + r2 ** 2 + r3 ** 2 + r4 ** 2)

            yaw_rate = self.ocp_solver.get(i+1, "x")[12]



            q = self.ocp_solver.get(8, "x")[3:7]
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            roll, pitch, yaw = tf_transformations.euler_from_quaternion([qx,
                                                                    qy,
                                                                    qz,
                                                                    qw], axes='rxyz')
            yaw_rate = self.ocp_solver.get(8, "x")[12]
            r1, r2, r3, r4 = self.ocp_solver.get(0, "u")
            motorConstant = 1.7965e-8
            thrust = motorConstant * (r1 ** 2 + r2 ** 2 + r3 ** 2 + r4 ** 2)

            
            u_mpc[i,:] = np.array([roll, pitch, yaw_rate, thrust])
            r_mpc[i,:] = np.array([r1, r2, r3, r4])


    




        x_mpc[N,:] = self.ocp_solver.get(N, "x")

        self.solver_locked = False

        cost_val = self.ocp_solver.get_cost()
        print("Objective value:", cost_val)

        if solution_callback is not None:
            solution_callback(status, x_mpc, u_mpc)
        else:    
            return status, x_mpc, u_mpc, r_mpc
        
def main():
    crazyflie_mpc_config_yaml = os.path.join(
        get_package_share_directory('crazyflie_mpc'),
        'config',
        'mpc.yaml')
    
    with open(crazyflie_mpc_config_yaml, 'r') as file:
        crazyflie_mpc_config = yaml.safe_load(file)
    
    build_acados = crazyflie_mpc_config['build_acados']


    # Quadrotor Parameters
    mass = crazyflie_mpc_config['drone_properties']['mass']
    arm_length = crazyflie_mpc_config['drone_properties']['arm_length']
    Ixx = crazyflie_mpc_config['drone_properties']['Ixx']
    Iyy = crazyflie_mpc_config['drone_properties']['Iyy']
    Izz = crazyflie_mpc_config['drone_properties']['Izz']
    cm = crazyflie_mpc_config['drone_properties']['cm']
    tau = crazyflie_mpc_config['drone_properties']['attitude_time_constant']
    motorConstant = crazyflie_mpc_config['drone_properties']['motorConstant']
    momentConstant = crazyflie_mpc_config['drone_properties']['momentConstant']

    # MPC Parameters
    mpc_tf = crazyflie_mpc_config['mpc']['horizon']
    mpc_N = crazyflie_mpc_config['mpc']['num_steps']
    control_update_rate = crazyflie_mpc_config['mpc']['control_update_rate']
    plot_trajectory = crazyflie_mpc_config['mpc']['plot_trajectory']

    print(f'mass: {mass}, arm_length: {arm_length}, Ixx: {Ixx}, Iyy: {Iyy}, Izz: {Izz}, cm: {cm}, tau: {tau}, mpc_tf: {mpc_tf}, mpc_N: {mpc_N}, control_update_rate: {control_update_rate}, plot_trajectory: {plot_trajectory}')

    quadrotor_dynamics = QuadrotorFull(mass, arm_length, Ixx, Iyy, Izz, cm, tau, motorConstant, momentConstant)
    acados_c_generated_code_path = pathlib.Path(get_package_share_directory('crazyflie_mpc')).resolve() / 'acados_generated_files'
    mpc_solver = TrajectoryTrackingMpc('crazyflie', quadrotor_dynamics, mpc_tf, mpc_N, code_export_directory=acados_c_generated_code_path)
    if build_acados:
        mpc_solver.generate_mpc()


    # Generate yref for 25 steps and 13 state variables
    num_steps = 50
    num_states = 13
    yref = np.zeros((num_states, num_steps))
    yref[2, :] = 0.2  # z position to 1 meter
    yref[3, :] = 1.0
    # yref[0,:]=10.0  # x position to 10 meters

    x0 = np.zeros(num_states)
    x0[2] = 0.1  # initial z position
    x0[3] = 1.0  # initial qw (quaternion w)

    yref_e = np.zeros(num_states)
    yref_e[2] = 0.2
    yref_e[3] = 1.0
    # yref_e[0] = 10.0

    start_time = time()
    result = mpc_solver.solve_mpc(x0, yref, yref_e, None)
    end_time = time()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"Solve MPC execution time: {elapsed_ms:.2f} ms")

    if result is not None:
        status, x_mpc, u_mpc, motors = result
    else:
        print("MPC solver did not return a result.")


    print("\n\n\n RESULT:\n")
    print(f"Status: {status}")
    print(f"x_mpc: {x_mpc}")
    print(f"u_mpc: {u_mpc}")
    print(f"Motors: {motors}")


if __name__ == "__main__":
    main()