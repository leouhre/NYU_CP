import numpy as np
import qdraw

### Equations of motion
def derivative(state_vec):
    r1, r2, r3, v1, v2, v3 = state_vec

    def f(r1, r2, eps=1e-5): # denominator
        r = r2 - r1
        r2norm = np.dot(r, r) + eps**2 
        return r / (r2norm * np.sqrt(r2norm))

    ar1 = G * (m2 * f(r1, r2) + m3 * f(r1, r3))
    ar2 = G * (m3 * f(r2, r3) + m1 * f(r2, r1))
    ar3 = G * (m1 * f(r3, r1) + m2 * f(r3, r2))

    state_vec_dot = np.array([v1, v2, v3, ar1, ar2, ar3])
    return state_vec_dot

### RK4 integrator
def rk4_step(state_vec, h):
    k1 = derivative(state_vec)
    k2 = derivative(state_vec + 0.5*h*k1)
    k3 = derivative(state_vec + 0.5*h*k2)
    k4 = derivative(state_vec + h*k3)
    return state_vec + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

### Adaptive time step
def adaptive_rk4(state_vec0, t_max, h_init, tol):
    t = 0.0
    h = h_init
    state_vec = state_vec0.copy()
    
    traj = []
    times = []
    delta = tol

    while t < t_max:
        # one double step
        state_vec_double = rk4_step(state_vec, h)
        # two single steps
        state_vec_single = rk4_step(state_vec, h/2)
        state_vec_single = rk4_step(state_vec_single, h/2)
        
        error = np.linalg.norm(state_vec_double - state_vec_single) # total error on all three 

        rho = 30 * delta * h / (error + 1e-40) # rho from the book

        if rho >= 1.0:
            state_vec = state_vec_single
            t += h
            traj.append(state_vec.copy())
            times.append(h)
            h *= min(1.5, rho**0.25)
        else:
            h *= max(0.5, rho**0.25)
    
    return np.array(times), np.array(traj)

if __name__ == "__main__":
    G = 1.0
    # initial positions and velocities
    # closed solution
    # m1, m2, m3 = 1.0, 1.0, 1.0
    # r1 = np.array([ 0.0, 0.0 ])
    # v1 = np.array([ 0.93240737, 0.86473146 ])
    # r2 = np.array([ 0.97000436, -0.24308753 ])
    # v2 = np.array([ -0.46620369, -0.43236573 ])
    # r3 = np.array([ -0.97000436, 0.24308753 ])
    # v3 = np.array([ -0.46620369, -0.43236573 ])

    # chaotic 
    m1, m2, m3 = 150.0, 200.0, 250.0
    r1 = np.array([ 3.0, 1.0 ])
    v1 = np.array([  0.0, 0.0 ])
    r2 = np.array([  -1.0, -2.0 ])
    v2 = np.array([  0.0, 0.0 ])
    r3 = np.array([  -1.0, 1.0 ])
    v3 = np.array([  0.0, 0.0 ])

    state_vec0 = np.array([r1, r2, r3, v1, v2, v3])

    # Integrate
    times, traj = adaptive_rk4(state_vec0, t_max=100.0, h_init=1e-3, tol=1e-3)

    x1 = traj[:,0,0]
    y1 = traj[:,0,1]
    x2 = traj[:,1,0]
    y2 = traj[:,1,1]
    x3 = traj[:,2,0]
    y3 = traj[:,2,1]

    win = qdraw.window(xlim=(-4,4), ylim=(-4,4))

    b1 = qdraw.circle(size=0.05, color='r'); b1.trail(True, width=1, length=100, color='r')
    b2 = qdraw.circle(size=0.05, color='g'); b2.trail(True, width=1, length=100, color='g')
    b3 = qdraw.circle(size=0.05, color='b'); b3.trail(True, width=1, length=100, color='b')

    for i in range(0, len(times)):
        b1.setpos(x1[i], y1[i])
        b2.setpos(x2[i], y2[i])
        b3.setpos(x3[i], y3[i])
        qdraw.draw(0.001)
    qdraw.hold()