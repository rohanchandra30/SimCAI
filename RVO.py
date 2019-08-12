from math import ceil, floor, sqrt
import copy
import numpy as np

from math import cos, sin, tan, atan2, asin

from math import pi as PI

######################### BEGIN SimCAI IMPLEMENTATION ##########################

def distance(pose1, pose2):
    """ compute Euclidean distance for 2D """
    return sqrt((pose1[0]-pose2[0])**2+(pose1[1]-pose2[1])**2)+0.001

# def has_intent(pi, pk):
#     pass
#
# def has_ability(pi, pk):
#     pass
#
# def interact(pi, pk):
#     pass

def personal_space_ray_dist(ray_slope, agent_pos):
    agent_x = agent_pos[0]
    agent_y = agent_pos[1]
    dist = abs(-ray_slope*agent_x + agent_y) / sqrt((-ray_slope)**2 + 1**2)
    return dist

def update_agents(X, V_des, V, ws_model, agents_within_SOC_REGs, interacting_agents, step, tau, is_interact_agent):
    # Section IV-B.1 in paper: determine interaction intent
    interaction_intents = {agent_id: [] for agent_id in range(len(X))} # e.g. {agent1: [agent2, agent5, agent6, ...]}
    able_to_interact = {agent_id: [] for agent_id in range(len(X))}

    for i, k_lst in agents_within_SOC_REGs.items():
        for k, timesteps in k_lst.items():
            # see if agent pk has been within agent pi's social region for at
            # least tau time steps
            if timesteps >= tau:
                interaction_intents[i].append(k)

    V_opt = list(V)
    for i in range(len(X)):
        V_opt_rvo = RVO_update(i, X, V_des, V_opt, ws_model)

    # for i in range(len(V_opt)):
    #     if is_interact_agent[i]:
    #         V_opt[i] = V_des[i]
    for i in range(len(X)):
        if is_interact_agent[i]:
            V_opt, agents_within_SOC_REGs, interacting_agents = SimCAI_update(i, X, V_des, ws_model, agents_within_SOC_REGs, interacting_agents, interaction_intents, step, tau, is_interact_agent)

    for i in list(np.where(is_interact_agent)[0]):
        V_opt_rvo[i] = V_opt[i]


    return V_opt_rvo, agents_within_SOC_REGs, interacting_agents

# Must modularize further with above methods
def SimCAI_update(i, psi, V_current, ws_model, agents_within_SOC_REGs,
                    interacting_agents, interaction_intents, step, tau, is_interact_agent):
    """
    MUST OPTIMIZE
    Notes:
        - If agents p_i and p_k are able to interact, should I immediately make
          them both start interacting with each other? Otherwise, there could
          be the issue of p_i going toward p_k, if p_k minimizes the p_w
          distance for p_i, and p_k going toward p_j, if p_j minimizes the p_w
          distance for p_k.
            - ***Ended up going with this logic***
    """
    #print('interacting_agents: ', interacting_agents)
    #print('agents_within_SOC_REGs: ', agents_within_SOC_REGs)
    #ROB_RAD = ws_model['robot_radius']+0.1
    SOC_REG = ws_model['social_region']
    ZETA_RAD = ws_model['personal_space']
    PHI = ws_model['steering_angle']
    ROB_RAD = ws_model['robot_radius'] + 0.1
    # print('V_current: ', V_current)
    # print('V_des: ', V_des)
    #print(interaction_intents)
    # velocity and position of agent pi
    V_pi = [V_current[i][0], V_current[i][1]]
    p_pi = [psi[i][0], psi[i][1]]
    theta_i = atan2(V_pi[1] - p_pi[1], V_pi[0] - p_pi[0])
    V_des_slope_i = tan(theta_i)
    r1_i_angle = theta_i - PHI
    r2_i_angle = theta_i + PHI
    r1_i = tan(r1_i_angle)
    r2_i = tan(r2_i_angle)
    #RVO_BA_all = []
    for k in range(len(psi)):
        if i != k and is_interact_agent[k]:
            # Section IV-B.2 in paper
            # velocity and position of agent pk
            V_pk = [V_current[k][0], V_current[k][1]]
            p_pk = [psi[k][0], psi[k][1]]
            theta_k = atan2(V_pk[1] - p_pk[1], V_pk[0] - p_pk[0])
            V_des_slope_k = tan(theta_k)

            r1_k_angle = theta_i - PHI
            r2_k_angle = theta_i + PHI
            r1_k = tan(r1_k_angle)
            r2_k = tan(r2_k_angle)
            #transl_vB_vA = [pA[0]+0.5*(vB[0]+vA[0]), pA[1]+0.5*(vB[1]+vA[1])]
            dist_pi_pk = distance(p_pi, p_pk)

            if interacting_agents[i] == k and interacting_agents[k] == i:
                if dist_pi_pk <= 2 * ZETA_RAD: # Check if interacting agents are within personal space
                    V_current[i] = [0, 0]
                    V_current[k] = [0, 0]
                else:
                    #     See if agent pk is close to agent pi and within agent pi's
                    #     steering angle. Wait for pk to pass.
                    for j in range(len(psi)):
                        if i != j and k != j:
                            p_pj = [psi[j][0], psi[j][1]]
                            theta_pi_pj = atan2(p_pj[1] - p_pi[1], p_pj[0] - p_pi[0])
                            theta_i = theta_pi_pj
                            V_des_mag_i = distance(p_pi, [V_pi[0] + p_pi[0], V_pi[1] + p_pi[1]])
                            V_des_slope_i = tan(theta_i)
                            V_des_u_i = V_des_mag_i * cos(theta_i)
                            V_des_v_i = V_des_mag_i * sin(theta_i)
                            if theta_pi_pj >= r1_i_angle and theta_pi_pj <= r2_i_angle and \
                                    distance(p_pi, p_pj) - 2*ZETA_RAD <= ZETA_RAD + 0.2:
                                V_current[i] = [0, 0]
            elif interacting_agents[i] is None and interacting_agents[k] is None:
                #print('interaction_intents: ', interaction_intents)
                theta_pi_pk = atan2(p_pk[1] - p_pi[1], p_pk[0] - p_pi[0])
                if k in interaction_intents[i] and not interacting_agents[i]:
                    # Redirect cone of pi toward pk
                    theta_pi_pk = atan2(p_pk[1] - p_pi[1], p_pk[0] - p_pi[0])
                    theta_i = theta_pi_pk
                    V_des_mag_i = distance(p_pi, [V_pi[0] + p_pi[0], V_pi[1] + p_pi[1]])
                    V_des_slope_i = tan(theta_i)
                    V_des_u_i = V_des_mag_i * cos(theta_i)
                    V_des_v_i = V_des_mag_i * sin(theta_i)

                    # Redirect cone of pk toward pi
                    theta_pk_pi = atan2(p_pi[1] - p_pk[1], p_pi[0] - p_pk[0])
                    theta_k = theta_pk_pi
                    V_des_mag_k = distance(p_pk, [V_pk[0] + p_pk[0], V_pk[1] + p_pk[1]])
                    V_des_slope_k = tan(theta_k)
                    V_des_u_k = V_des_mag_k * cos(theta_k)
                    V_des_v_k = V_des_mag_k * sin(theta_k)

                    if personal_space_ray_dist(r1_i, p_pk) <= ZETA_RAD or \
                            personal_space_ray_dist(r2_i, p_pk) <= ZETA_RAD: # Condition 1: check if rays
                                                                             # intersect with pk personal space
                        can_interact = True
                    else:
                        can_interact = True
                        for j in range(len(psi)):
                            if j != i and j != k:
                                p_pj = [psi[j][0], psi[j][1]]
                                dist_pi_pj = distance(p_pi, p_pj)

                                # Condition 2: check if agent p_pj is between p_pi and p_pk
                                if dist_pi_pj <= dist_pi_pk and \
                                        personal_space_ray_dist(V_des_slope_i, p_pj) <= ZETA_RAD:
                                    can_interact = False

                    # Section IV-B.3 in paper
                    # Still have to implement overlapping agents
                    #print('can_interact: ', can_interact)
                    if can_interact:
                        t_to_converge = np.linalg.norm(np.array(p_pi) - np.array(p_pk), 2) / \
                                        np.linalg.norm(np.array(V_pi) - np.array(V_pk), 2) # placeholder (implement equation with L2 norms in section 3 Interaction)
                        delta_t = 0.05 # placeholder
                        p_pw_lst = []
                        V_pw_lst = []
                        for pw_id in interaction_intents[k]:
                            # convert to np arrays for matrix operations in
                            # distance minimization
                            if interacting_agents[pw_id] is None:
                                p_pw_lst.append(np.array([psi[pw_id][0], psi[pw_id][1]]))
                                V_pw_lst.append(np.array([V_current[pw_id][0], V_current[pw_id][1]]))
                        p_pw_V_pw_lst = zip(p_pw_lst, V_pw_lst)
                        # should I use intend to interact or able to interact?
                        dists = map(lambda p_pw, V_pw:
                                    p.linalg.norm(p_pw + V_pw*delta_t - p_pk, order=1),
                                    p_pw_V_pw_lst)
                        agent_with_min_dist = np.argmin(dists)
                        interacting_agents[i] = k
                        interacting_agents[k] = i
                        #print('agents {} and {} interacting'.format(i, k))
                        V_current[i] = [V_des_u_i, V_des_v_i]
                        V_current[k] = [V_des_u_k, V_des_v_k]

                        # V_new[i] = V_des[i]
                        # V_new[k] = V_des[k]
                # else:
                #     See if agent pk is close to agent pi and within agent pi's
                #     steering angle. Wait for pk to pass.
                #     if theta_pi_pk >= r1_i_angle and theta_pi_pk <= r2_i_angle and \
                #             distance(p_pi, p_pk) - 2*ZETA_RAD <= ZETA_RAD + 0.2:
                #         V_des[i] = [0, 0]

                elif dist_pi_pk <= SOC_REG:
                    # agents_within_SOC_REGs is in the form of
                    # {pi: {pk: count, ...}, pk: {...}, ...}
                    #print('agent {} within soc reg of {}'.format(k, i))
                    agents_within_SOC_REGs[i][k] += step
                else:
                    # Remove
                    agents_within_SOC_REGs[i][k] = 0
            # elif interacting_agents[i] != k and interacting_agents[k] != i:
            #     theta_pi_pk = atan2(p_pk[1] - p_pi[1], p_pk[0] - p_pi[0])
            #
            #     # See if agent pk is close to agent pi and within agent pi's
            #     # steering angle
            #     if theta_pi_pk >= r1_i_angle and theta_pi_pk <= r2_i_angle and \
            #             distance(p_pi, p_pk) - 2*ZETA_RAD <= ZETA_RAD + 0.2:
            #         V_des[i] = [0, 0]\
        #print(interacting_agents)

    return V_current, agents_within_SOC_REGs, interacting_agents



########################## END SimCAI IMPLEMENTATION ###########################

def RVO_update_rvo(X, V_des, V_current, ws_model):
    """ compute best velocity given the desired velocity, current velocity and workspace model"""
    ROB_RAD = ws_model['robot_radius'] + 0.1
    V_opt = list(V_current)
    for i in range(len(X)):
        vA = [V_current[i][0], V_current[i][1]]
        pA = [X[i][0], X[i][1]]
        RVO_BA_all = []
        for j in range(len(X)):
            if i != j:
                vB = [V_current[j][0], V_current[j][1]]
                pB = [X[j][0], X[j][1]]
                # use RVO
                transl_vB_vA = [pA[0] + 0.5 * (vB[0] + vA[0]), pA[1] + 0.5 * (vB[1] + vA[1])]
                # use VO
                # transl_vB_vA = [pA[0]+vB[0], pA[1]+vB[1]]
                dist_BA = distance(pA, pB)
                theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
                if 2 * ROB_RAD > dist_BA:
                    dist_BA = 2 * ROB_RAD
                theta_BAort = asin(2 * ROB_RAD / dist_BA)
                theta_ort_left = theta_BA + theta_BAort
                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                theta_ort_right = theta_BA - theta_BAort
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
                # use HRVO
                # dist_dif = distance([0.5*(vB[0]-vA[0]),0.5*(vB[1]-vA[1])],[0,0])
                # transl_vB_vA = [pA[0]+vB[0]+cos(theta_ort_left)*dist_dif, pA[1]+vB[1]+sin(theta_ort_left)*dist_dif]
                RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, 2 * ROB_RAD]
                RVO_BA_all.append(RVO_BA)
        for hole in ws_model['circular_obstacles']:
            # hole = [x, y, rad]
            vB = [0, 0]
            pB = hole[0:2]
            transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
            dist_BA = distance(pA, pB)
            theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
            # over-approximation of square to circular
            OVER_APPROX_C2S = 1.5
            rad = hole[2] * OVER_APPROX_C2S
            if (rad + ROB_RAD) > dist_BA:
                dist_BA = rad + ROB_RAD
            theta_BAort = asin((rad + ROB_RAD) / dist_BA)
            theta_ort_left = theta_BA + theta_BAort
            bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
            theta_ort_right = theta_BA - theta_BAort
            bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
            RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad + ROB_RAD]
            RVO_BA_all.append(RVO_BA)
        vA_post = intersect(pA, V_des[i], RVO_BA_all)
        V_opt[i] = vA_post[:]
    return V_opt


def RVO_update(i, X, V_des, V_current, ws_model):
    """ compute best velocity given the desired velocity, current velocity and workspace model"""
    ROB_RAD = ws_model['robot_radius'] + 0.1

    vA = [V_current[i][0], V_current[i][1]]
    pA = [X[i][0], X[i][1]]
    RVO_BA_all = []
    for j in range(len(X)):
        if i!=j:
            vB = [V_current[j][0], V_current[j][1]]
            pB = [X[j][0], X[j][1]]
            # use RVO
            transl_vB_vA = [pA[0]+0.5*(vB[0]+vA[0]), pA[1]+0.5*(vB[1]+vA[1])]
            # use VO
            #transl_vB_vA = [pA[0]+vB[0], pA[1]+vB[1]]
            dist_BA = distance(pA, pB)
            theta_BA = atan2(pB[1]-pA[1], pB[0]-pA[0])
            if 2*ROB_RAD > dist_BA:
                dist_BA = 2*ROB_RAD
            theta_BAort = asin(2*ROB_RAD/dist_BA)
            theta_ort_left = theta_BA+theta_BAort
            bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
            theta_ort_right = theta_BA-theta_BAort
            bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
            # use HRVO
            # dist_dif = distance([0.5*(vB[0]-vA[0]),0.5*(vB[1]-vA[1])],[0,0])
            # transl_vB_vA = [pA[0]+vB[0]+cos(theta_ort_left)*dist_dif, pA[1]+vB[1]+sin(theta_ort_left)*dist_dif]
            RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, 2*ROB_RAD]
            RVO_BA_all.append(RVO_BA)
    for hole in ws_model['circular_obstacles']:
        # hole = [x, y, rad]
        vB = [0, 0]
        pB = hole[0:2]
        transl_vB_vA = [pA[0]+vB[0], pA[1]+vB[1]]
        dist_BA = distance(pA, pB)
        theta_BA = atan2(pB[1]-pA[1], pB[0]-pA[0])
        # over-approximation of square to circular
        OVER_APPROX_C2S = 1.5
        rad = hole[2]*OVER_APPROX_C2S
        if (rad+ROB_RAD) > dist_BA:
            dist_BA = rad+ROB_RAD
        theta_BAort = asin((rad+ROB_RAD)/dist_BA)
        theta_ort_left = theta_BA+theta_BAort
        bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
        theta_ort_right = theta_BA-theta_BAort
        bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
        RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad+ROB_RAD]
        RVO_BA_all.append(RVO_BA)
    vA_post = intersect(pA, V_des[i], RVO_BA_all)
    V_current[i] = vA_post[:]
    #print(V_opt)
    return V_current


def intersect(pA, vA, RVO_BA_all):
    # print '----------------------------------------'
    # print 'Start intersection test'
    norm_v = distance(vA, [0, 0])
    suitable_V = []
    unsuitable_V = []
    for theta in np.arange(0, 2*PI, 0.1):
        for rad in np.arange(0.02, norm_v+0.02, norm_v/5.0):
            new_v = [rad*cos(theta), rad*sin(theta)]
            suit = True
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    suit = False
                    break
            if suit:
                suitable_V.append(new_v)
            else:
                unsuitable_V.append(new_v)
    new_v = vA[:]
    suit = True
    for RVO_BA in RVO_BA_all:
        p_0 = RVO_BA[0]
        left = RVO_BA[1]
        right = RVO_BA[2]
        dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
        theta_dif = atan2(dif[1], dif[0])
        theta_right = atan2(right[1], right[0])
        theta_left = atan2(left[1], left[0])
        if in_between(theta_right, theta_dif, theta_left):
            suit = False
            break
    if suit:
        suitable_V.append(new_v)
    else:
        unsuitable_V.append(new_v)
    #----------------------
    if suitable_V:
        # print 'Suitable found'
        vA_post = min(suitable_V, key = lambda v: distance(v, vA))
        new_v = vA_post[:]
        for RVO_BA in RVO_BA_all:
            p_0 = RVO_BA[0]
            left = RVO_BA[1]
            right = RVO_BA[2]
            dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
            theta_dif = atan2(dif[1], dif[0])
            theta_right = atan2(right[1], right[0])
            theta_left = atan2(left[1], left[0])
    else:
        # print 'Suitable not found'
        tc_V = dict()
        for unsuit_v in unsuitable_V:
            tc_V[tuple(unsuit_v)] = 0
            tc = []
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dist = RVO_BA[3]
                rad = RVO_BA[4]
                dif = [unsuit_v[0]+pA[0]-p_0[0], unsuit_v[1]+pA[1]-p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    small_theta = abs(theta_dif-0.5*(theta_left+theta_right))
                    if abs(dist*sin(small_theta)) >= rad:
                        rad = abs(dist*sin(small_theta))
                    big_theta = asin(abs(dist*sin(small_theta))/rad)
                    dist_tg = abs(dist*cos(small_theta))-abs(rad*cos(big_theta))
                    if dist_tg < 0:
                        dist_tg = 0
                    tc_v = dist_tg/distance(dif, [0,0])
                    tc.append(tc_v)
            tc_V[tuple(unsuit_v)] = min(tc)+0.001
        WT = 0.2
        vA_post = min(unsuitable_V, key = lambda v: ((WT/tc_V[tuple(v)])+distance(v, vA)))
    return vA_post

def in_between(theta_right, theta_dif, theta_left):
    if abs(theta_right - theta_left) <= PI:
        if theta_right <= theta_dif <= theta_left:
            return True
        else:
            return False
    else:
        if (theta_left <0) and (theta_right >0):
            theta_left += 2*PI
            if theta_dif < 0:
                theta_dif += 2*PI
            if theta_right <= theta_dif <= theta_left:
                return True
            else:
                return False
        if (theta_left >0) and (theta_right <0):
            theta_right += 2*PI
            if theta_dif < 0:
                theta_dif += 2*PI
            if theta_left <= theta_dif <= theta_right:
                return True
            else:
                return False

def compute_V_des_rvo(X, goal, V_max):
    V_des = []
    for i in range(len(X)):
        dif_x = [goal[i][k]-X[i][k] for k in range(2)]
        norm = distance(dif_x, [0, 0])
        norm_dif_x = [dif_x[k]*V_max[k]/norm for k in range(2)]
        V_des.append(norm_dif_x[:])
        if reach(X[i], goal[i], 0.1):
            V_des[i][0] = 0
            V_des[i][1] = 0
    return V_des

def compute_V_des(X, goal, V_max, interacting, old_V_des):
    V_des = []
    for i in range(len(X)):
        if i in interacting:
            V_des.append(old_V_des[i])
        else:
            dif_x = [goal[i][k]-X[i][k] for k in range(2)]
            norm = distance(dif_x, [0, 0])
            norm_dif_x = [dif_x[k]*V_max[k]/norm for k in range(2)]
            V_des.append(norm_dif_x[:])
            if reach(X[i], goal[i], 0.1):
                V_des[i][0] = 0
                V_des[i][1] = 0
    return V_des

def reach(p1, p2, bound=0.5):
    if distance(p1,p2)< bound:
        return True
    else:
        return False
