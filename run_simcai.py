import sys


from RVO import update_agents, reach, compute_V_des
from visuals import visualize_traj_dynamic


#------------------------------
#define workspace model
ws_model = dict()
#robot radius
ws_model['robot_radius'] = 0.2
#circular obstacles, format [x,y,rad]
# no obstacles
ws_model['circular_obstacles'] = []
# with obstacles
# ws_model['circular_obstacles'] = [[-0.3, 2.5, 0.3], [1.5, 2.5, 0.3], [3.3, 2.5, 0.3], [5.1, 2.5, 0.3]]
#rectangular boundary, format [x,y,width/2,heigth/2]
ws_model['boundary'] = []

ws_model['social_region'] = 1
ws_model['personal_space'] = 0.05    # compute the optimal vel to avoid collision

ws_model['steering_angle'] = 60

#------------------------------
#initialization for robot
# position of [x,y]
X = [[-0.5+1.0*i, 0.0] for i in range(7)] + [[-0.5+1.0*i, 5.0] for i in range(7)]
# velocity of [vx,vy]
V = [[0,0] for i in range(len(X))]
# maximal velocity norm
V_max = [1.0 for i in range(len(X))]
# goal of [x,y]
goal = [[5.5-1.0*i, 5.0] for i in range(7)] + [[5.5-1.0*i, 0.0] for i in range(7)]

is_interact_agent = [False] * len(X)
is_interact_agent[1] = True
is_interact_agent[5] = True
is_interact_agent[7] = True
# is_interact_agent[2] = True
is_interact_agent[10] = True

agents_within_soc_regs = {i: {k: 0} for i in range(len(X)) for k in range(len(X)) if i != k}
interacting_agents = {i: None for i in range(len(X))}
#------------------------------
#simulation setup
# total simulation time (s)
total_time = 15
# simulation step
step = 0.01
# min timesteps to consider as having intent to interact
tau = 0.5
#------------------------------
#simulation starts
t = 0

# Compute desired velocity to goal
#V_des = compute_V_des(X, goal, V_max)
while t*step < total_time:
    interacting = set()
    for pi, pk in interacting_agents.items():
        if pk:
            interacting.add(pi)
    if t == 0:
        V_des = compute_V_des(X, goal, V_max, [], [])
        print(V_des)
    else:
        V_des = compute_V_des(X, goal, V_max, interacting, V)
    #print('V_des: ', V_des)
    # compute the optimal vel to avoid collision
    V, agents_within_soc_regs, interacting_agents = update_agents(X, V_des, V, ws_model,
                                              agents_within_soc_regs,
                                              interacting_agents, step, tau,
                                              is_interact_agent)
    #print(interacting_agents)

    #print(V)
    # update position
    for i in range(len(X)):
        X[i][0] += V[i][0]*step
        X[i][1] += V[i][1]*step
    #----------------------------------------
    # visualization
    if t%10 == 0:
        visualize_traj_dynamic(ws_model, X, V, goal, time=t*step, name='data/snap%s.png'%str(t/10))
        #visualize_traj_dynamic(ws_model, X, V, goal, time=t*step, name='data/snap%s.png'%str(t/10))
    t += 1
