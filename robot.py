class Agent:
    def __init__(self, pos, goal, vel, max_vel_norm, interact):
        self.pos = pos
        self.goal = goal
        self.vel = vel
        self.max_vel_norm = max_vel_norm
        self.interact = interact

    def get_pos(self):
        return self.pos

    def set_pos(self, new_pos):
        self.pos = new_pos

    def get_goal(self):
        return self.goal

    def get_vel(self):
        return self.vel

    def set_vel(self, new_vel):
        self.vel = new_vel

    def get_max_vel_norm(self):
        return self.max_vel_norm

    def is_interact(self):
        return self.interact
