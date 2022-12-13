from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

u_x = SX.sym('u_x')
u_y = SX.sym('u_y')
u = vertcat(
    u_x,
    u_y,
)

P = SX.sym('P', 2 + 2 + 6 + 15) # 2 (v_desired + 2(current_rel_v) + 6(mean terms) + 15(variance terms) ))

mean_ctrl_coeffs = vertcat( u_x**2, u_y**2, u_x*u_y, u_x, u_y, 1 )

var_ctrl_coeffs = vertcat( u_x**4, u_y**4,  u_x**3*u_y, u_x*u_y**3, (u_x**2)*(u_y**2), u_x**3, u_y**3, (u_x**2)*(u_y), (u_x)*(u_y**2), u_x**2, u_y**2, u_x*u_y, u_x, u_y, 1 )

r_rad=0.5 
r_pos = np.array([0.0,0.0])
r_vel= np.array([1.0,1.0])
v_max=  1.0
g=0.5
goal = np.array([15.0,15.0])
v_desired = v_max*(goal -r_pos)/np.linalg.norm(goal-r_pos)
dt = 0.2


obst_pos = np.array([7.5,7.5])
obst_vel = np.array([-1.0, -1.0])
obs_rad = 0.5

mu_x = 0
std_x = 0.1

mu_y = 0
std_y = 0.1

n_samples = 10





def calc_mean_terms(C_i_coeffs):
    mean_terms = np.zeros(shape=(6,1))

    for i in range(0,6):
        mean_terms[i] = np.mean(C_i_coeffs[i]) 
    return mean_terms


def calc_variance_terms(C_i_coeffs,mean_terms):
    variance_terms = np.zeros(shape=(15,1)) 
    variance_terms[0] = np.mean(C_i_coeffs[0] - mean_terms[0] )**2                                                                                                                          # u_x^4 coeff 
    variance_terms[1] = np.mean(C_i_coeffs[1] - mean_terms[1] )**2                                                                                                                          # u_y^4 coeff 
    variance_terms[2] = 2* (np.mean((C_i_coeffs[0] - mean_terms[0])*(C_i_coeffs[2] - mean_terms[2])))                                                                                       # u_x^3*u_y coeff
    variance_terms[3] = 2* (np.mean((C_i_coeffs[1] - mean_terms[1])*(C_i_coeffs[2] - mean_terms[2])))                                                                                       # u_x*u_y^3 coeff
    variance_terms[4] = np.mean(C_i_coeffs[2] - mean_terms[2] )**2 + 2* (np.mean((C_i_coeffs[0] - mean_terms[0])*(C_i_coeffs[1] - mean_terms[1])))                                          # u_1^2*u_2^2 coeff
    variance_terms[5] = 2* (np.mean((C_i_coeffs[0] - mean_terms[0])*(C_i_coeffs[3] - mean_terms[3])))                                                                                       # u_x^3 coeff              
    variance_terms[6] = 2* (np.mean((C_i_coeffs[1] - mean_terms[1])*(C_i_coeffs[4] - mean_terms[4])))                                                                                       # u_y^3 coeff  
    variance_terms[7] = 2* (np.mean((C_i_coeffs[0] - mean_terms[0])*(C_i_coeffs[4] - mean_terms[4]))) + 2* (np.mean((C_i_coeffs[2] - mean_terms[2])*(C_i_coeffs[3] - mean_terms[3])))       # u_1^2*u_2 coeff
    variance_terms[8] = 2* (np.mean((C_i_coeffs[1] - mean_terms[1])*(C_i_coeffs[3] - mean_terms[3]))) + 2* (np.mean((C_i_coeffs[2] - mean_terms[2])*(C_i_coeffs[4] - mean_terms[4])))       # u_1*u_2^2 coeff
    variance_terms[9] = np.mean(C_i_coeffs[3] - mean_terms[3] )**2 + 2* (np.mean((C_i_coeffs[0] - mean_terms[0])*(C_i_coeffs[5] - mean_terms[5])))                                          # u_1^2 coeff  
    variance_terms[10] = np.mean(C_i_coeffs[4] - mean_terms[4] )**2 + 2* (np.mean((C_i_coeffs[1] - mean_terms[1])*(C_i_coeffs[5] - mean_terms[5])))                                         # u_2^2 coeff
    variance_terms[11] = 2* (np.mean((C_i_coeffs[4] - mean_terms[4])*(C_i_coeffs[3] - mean_terms[3]))) + 2* (np.mean((C_i_coeffs[2] - mean_terms[2])*(C_i_coeffs[5] - mean_terms[5])))      # u_1*u_2^2 coeff
    variance_terms[12] = 2* (np.mean((C_i_coeffs[3] - mean_terms[3])*(C_i_coeffs[5] - mean_terms[5])))                                                                                      # u_x coeff
    variance_terms[13] = 2* (np.mean((C_i_coeffs[4] - mean_terms[4])*(C_i_coeffs[5] - mean_terms[5])))                                                                                      # u_y coeff
    variance_terms[14] = np.mean(C_i_coeffs[5] - mean_terms[5] )**2                                                                                                                         # const 

    return variance_terms


def calc_coeffs(r_x, r_y, v_x, v_y,r,o): 

    C_i_coeffs = np.zeros(shape=(6,10))

    C_i_coeffs[0] = (r + o)**2 - (r_y)**2                                                                       # u_x^2 coeff
    C_i_coeffs[1] = (r + o)**2 - (r_x)**2                                                                       # u_y^2 coeff
    C_i_coeffs[2] = 2*(r_x)*(r_y)                                                                               # u_x*u_y coeff  
    C_i_coeffs[3] = -(+ 2*(r_x)*(r_y)*v_y - 2*(r_y**2)*v_x + 2*((r+o)**2)*v_x )                                 # u_x coeff
    C_i_coeffs[4] = -(+ 2*(r_x)*(r_y)*v_x - 2*(r_x**2)*v_y + 2*((r+o)**2)*v_y )                                 # u_y coeff
    C_i_coeffs[5] = ((r+o)**2)*(v_x**2 + v_y**2) - (r_x**2)*(v_y**2) - (r_y**2)*(v_x**2) + 2*r_x*r_y*v_x*v_y    # const

    return C_i_coeffs

def calc_variance_terms_anish(C_i_coeffs,mean_terms):
    variance_terms = np.zeros(shape=(15,1)) 
    variance_terms[0] = np.mean((C_i_coeffs[0])**2) - mean_terms[0]**2                                                                                                                          # u_x^4 coeff done
    variance_terms[1] = np.mean((C_i_coeffs[1])**2) - mean_terms[1]**2                                                                                                                          # u_y^4 coeff done
    variance_terms[2] = 2* (np.mean(np.dot(C_i_coeffs[0] , C_i_coeffs[2]) - (mean_terms[0]*mean_terms[2])))                                                                                       # u_x^3*u_y coeff done
    variance_terms[3] = 2* (np.mean(np.dot(C_i_coeffs[1] , C_i_coeffs[2]) - (mean_terms[1]*mean_terms[2])))                                                                                        # u_x*u_y^3 coeff done
    variance_terms[4] = np.mean((C_i_coeffs[2])**2) - mean_terms[2]**2 + 2* (np.mean(np.dot(C_i_coeffs[0] , C_i_coeffs[1]) - (mean_terms[0]*mean_terms[1])))                                        # u_1^2*u_2^2 coeff done
    variance_terms[5] = 2* (np.mean(np.dot(C_i_coeffs[0] , C_i_coeffs[3]) - (mean_terms[0]*mean_terms[3])))                                                                                       # u_x^3 coeff  done            
    variance_terms[6] = 2* (np.mean(np.dot(C_i_coeffs[1] , C_i_coeffs[4]) - (mean_terms[1]*mean_terms[4])))                                                                                       # u_y^3 coeff  done
    variance_terms[7] = 2* (np.mean(np.dot(C_i_coeffs[0] , C_i_coeffs[4]) - (mean_terms[0]*mean_terms[4]))) + 2* (np.mean(np.dot(C_i_coeffs[2] , C_i_coeffs[3]) - (mean_terms[2]*mean_terms[3])))      # u_1^2*u_2 coeff done
    variance_terms[8] = 2* (np.mean(np.dot(C_i_coeffs[1] , C_i_coeffs[3]) - (mean_terms[1]*mean_terms[3]))) + 2* (np.mean(np.dot(C_i_coeffs[2] , C_i_coeffs[4]) - (mean_terms[2]*mean_terms[4])))       # u_1*u_2^2 coeff done
    variance_terms[9] = np.mean((C_i_coeffs[3])**2) - mean_terms[3]**2   + 2* (np.mean(np.dot(C_i_coeffs[0] , C_i_coeffs[5]) - (mean_terms[0]*mean_terms[5])))                                           # u_1^2 coeff  done
    variance_terms[10] = np.mean((C_i_coeffs[4])**2) - mean_terms[4]**2 + 2* (np.mean(np.dot(C_i_coeffs[1] , C_i_coeffs[5]) - (mean_terms[1]*mean_terms[5])))                                       # u_2^2 coeff done
    variance_terms[11] = 2* (np.mean(np.dot(C_i_coeffs[3] , C_i_coeffs[4]) - (mean_terms[3]*mean_terms[4]))) + 2* (np.mean(np.dot(C_i_coeffs[2] , C_i_coeffs[5]) - (mean_terms[2]*mean_terms[5])))      # u_1*u_2^2 coeff
    variance_terms[12] = 2* (np.mean(np.dot(C_i_coeffs[3] , C_i_coeffs[5]) - (mean_terms[3]*mean_terms[5])))                                                                                        # u_x coeff done
    variance_terms[13] = 2* (np.mean(np.dot(C_i_coeffs[4] , C_i_coeffs[5]) - (mean_terms[4]*mean_terms[5])))                                                                                        # u_y coeff done
    variance_terms[14] = np.mean((C_i_coeffs[5])**2) - (mean_terms[5] )**2                                                                                                                         # const done

    return variance_terms 


    # Symbols/expressions

# cost_fn = norm_2(v_desired - (u + r_vel))

cost_fn = norm_2(P[0:2] - (u + P[2:4]))

lbx = -v_max* DM.ones((2, 1))
ubx = v_max*DM.ones((2, 1))

g = vertcat(dot(mean_ctrl_coeffs, P[4:10]) + 0.5*sqrt((dot(var_ctrl_coeffs, P[10:25])))
             dot(var_ctrl_coeffs, P[10:25])
             )   
 lbg = vertcat(-inf,0)
 ubg = vertcat(0,inf)

#g = dot(mean_ctrl_coeffs, P[4:10]) + 1.0*sqrt((dot(var_ctrl_coeffs, P[10:25])))
# g = vertcat(dot(mean_ctrl_coeffs, P[4:10]))
#lbg = -inf
#ubg = -0.5

#ff = Function('ff', [mean_ctrl_coeffs[0:5]], [u])



nlp_prob = {}                 # NLP declaration
nlp_prob['x']= u # decision vars
nlp_prob['f'] = cost_fn             # objective
nlp_prob['g'] = g            # constraints
nlp_prob['p'] = P

opts = {
    'ipopt': {
        'max_iter': 1000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

# Create solver instance
solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

u0 = DM.zeros((2, 1))

args = {
    'lbx': lbx,
    'ubx': ubx,
    'lbg': lbg,
    'ubg': ubg,
    'u0' : u0,
    'p'  : P
} 

goal_reached = False


# Solve the problem using a guess
mean_terms = np.zeros(shape=(6,1))
variance_terms = np.zeros(shape=(15,1))


while goal_reached == False:

    rel_obst_pos = obst_pos - r_pos
    rel_vel = obst_vel - r_vel

   # print(rel_obst_pos)
    noisy_obst_pos_x = np.array(rel_obst_pos[0]) + np.random.normal(mu_x, std_x, n_samples)
    noisy_obst_pos_y = np.array(rel_obst_pos[1]) + np.random.normal(mu_y, std_y, n_samples)

    C_i_coeffs = calc_coeffs(noisy_obst_pos_x, noisy_obst_pos_y, rel_vel[0], rel_vel[1], r_rad, obs_rad)

    mean_terms = calc_mean_terms(C_i_coeffs)
    variance_terms = calc_variance_terms(C_i_coeffs,mean_terms)

    # if(norm_2(rel_obst_pos) > 5):
    #     mean_terms = -1000*np.ones(shape=(6,1))
    #     variance_terms = -1000*np.ones(shape=(15,1))

    print(mean_terms)

    print(variance_terms)




    args['p'] = vertcat(
    v_desired,
    r_vel,
    mean_terms,
    variance_terms
)
    
    sol = solver(x0=args['u0'],
                lbx=args['lbx'],
                ubx=args['ubx'],
                lbg=args['lbg'],
                ubg=args['ubg'],
                p = args['p']
                )
    u_sol = sol['x']
   # print("u_sol: ")
   # print(u_sol)
   # print(sol)

    #u_sol = np.array(u_sol.full())

    args['u0'] = u_sol

    r_vel =r_vel + np.array([u_sol[0], u_sol[1]])

    print("r_vel: ")
    print(r_vel)
    r_pos=r_pos+(r_vel * dt)
    obst_pos = obst_pos + (obst_vel*dt)
    print(obst_pos)
    
    v_desired= v_max*(goal - r_pos)/np.linalg.norm(goal-r_pos)
    print("v_desired")
    print(v_desired)
    ubx = v_max* DM.ones((2, 1))
    lbx = -v_max* DM.ones((2, 1)) 


    circle1 = plt.Circle((r_pos[0], r_pos[1]), 0.5, color='r')
    circle2 = plt.Circle((obst_pos[0], obst_pos[1]), 0.5, color='r')
    goal_plt = plt.Circle((15.0, 15.0), 0.1, color='g')

    
    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    # change default range so that new circles will work
    ax.set_xlim((0, 20))
    ax.set_ylim((0, 20))

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(goal_plt)

    plt.pause(0.1)
