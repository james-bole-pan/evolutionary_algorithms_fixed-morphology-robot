import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Constants
g = np.array([0.0, 0.0, -9.81])  # Gravity
dt = 0.0001
k = 1000.0  # Spring constant
L0 = 1.0  # Rest length of the spring
damping = 1.0  # Damping constant

# Mass Definition
class Mass:
    def __init__(self, p, v, m=0.1):
        self.m = m
        self.p = np.array(p)
        self.v = np.array(v)
        self.a = np.zeros(3,dtype=float)
        self.f = np.zeros(3,dtype=float)

# Spring Definition
class Spring:
    def __init__(self, L0, k, m1, m2):
        self.L0 = L0
        self.k = k
        self.m1 = m1
        self.m2 = m2

# Initialize 8 masses for the cube
half_L0 = L0/2
drop_height = 2.0
masses = [
    Mass([-half_L0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 0
    Mass([half_L0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 1
    Mass([-half_L0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 2
    Mass([half_L0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 3
    Mass([-half_L0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 4
    Mass([half_L0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 5
    Mass([-half_L0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 6
    Mass([half_L0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0])      # 7
]

# add 4 more masses for the second cube
masses += [
    Mass([-half_L0, half_L0 + L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 8
    Mass([half_L0, half_L0 + L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 9
    Mass([-half_L0, half_L0 + L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 10
    Mass([half_L0, half_L0 + L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 11
]

# Connect the masses with springs to form a cube
springs = [
    Spring(L0, k, masses[0], masses[1]),  # Base square
    Spring(L0, k, masses[1], masses[3]),
    Spring(L0, k, masses[3], masses[2]),
    Spring(L0, k, masses[2], masses[0]),
    Spring(L0, k, masses[4], masses[5]),  # Top square
    Spring(L0, k, masses[5], masses[7]),
    Spring(L0, k, masses[7], masses[6]),
    Spring(L0, k, masses[6], masses[4]),
    Spring(L0, k, masses[0], masses[4]),  # Vertical edges
    Spring(L0, k, masses[1], masses[5]),
    Spring(L0, k, masses[2], masses[6]),
    Spring(L0, k, masses[3], masses[7]),
]
short_diag_length = np.sqrt(2 * L0**2)
springs += [
    Spring(short_diag_length, k, masses[0], masses[3]),
    Spring(short_diag_length, k, masses[1], masses[2]),
    Spring(short_diag_length, k, masses[4], masses[7]),
    Spring(short_diag_length, k, masses[5], masses[6]),
    # Short Diagonals between opposite faces
    Spring(short_diag_length, k, masses[0], masses[5]),
    Spring(short_diag_length, k, masses[1], masses[4]),
    Spring(short_diag_length, k, masses[2], masses[7]),
    Spring(short_diag_length, k, masses[3], masses[6]),
    Spring(short_diag_length, k, masses[1], masses[7]),
    Spring(short_diag_length, k, masses[0], masses[6]),
    Spring(short_diag_length, k, masses[3], masses[5]),
    Spring(short_diag_length, k, masses[2], masses[4])
]

# Long Diagonals
long_diag_length = np.sqrt(3 * L0**2)
springs += [
    Spring(long_diag_length, k, masses[0], masses[7]),
    Spring(long_diag_length, k, masses[1], masses[6]),
    Spring(long_diag_length, k, masses[2], masses[5]),
    Spring(long_diag_length, k, masses[3], masses[4])
]

# Springs for the second cube
springs += [
    Spring(L0, k, masses[8], masses[9]), 
    Spring(L0, k, masses[9], masses[11]),
    Spring(L0, k, masses[11], masses[10]),
    Spring(L0, k, masses[10], masses[8]),
    Spring(L0, k, masses[6], masses[10]),
    Spring(L0, k, masses[7], masses[11]),
    Spring(L0, k, masses[4], masses[8]),
    Spring(L0, k, masses[5], masses[9]),
    Spring(short_diag_length, k, masses[6], masses[11]),
    Spring(short_diag_length, k, masses[7], masses[10]),
    Spring(short_diag_length, k, masses[4], masses[9]),
    Spring(short_diag_length, k, masses[5], masses[8]),
    Spring(short_diag_length, k, masses[4], masses[10]),
    Spring(short_diag_length, k, masses[5], masses[11]),
    Spring(short_diag_length, k, masses[6], masses[8]),
    Spring(short_diag_length, k, masses[7], masses[9]),
    Spring(short_diag_length, k, masses[9], masses[10]),
    Spring(short_diag_length, k, masses[8], masses[11]),
    Spring(long_diag_length, k, masses[6], masses[9]),
    Spring(long_diag_length, k, masses[7], masses[8]),
    Spring(long_diag_length, k, masses[4], masses[11]),
    Spring(long_diag_length, k, masses[5], masses[10])
]

def get_cube_faces():
    return [
        [masses[0].p, masses[1].p, masses[5].p, masses[4].p],  # Bottom face
        [masses[2].p, masses[3].p, masses[7].p, masses[6].p],  # Top face
        [masses[0].p, masses[1].p, masses[3].p, masses[2].p],  # Front face
        [masses[0].p, masses[4].p, masses[6].p, masses[2].p],  # Left face
        [masses[1].p, masses[5].p, masses[7].p, masses[3].p],   # Right face
        [masses[6].p, masses[7].p, masses[11].p, masses[10].p],
        [masses[7].p, masses[11].p, masses[9].p, masses[5].p],
        [masses[5].p, masses[9].p, masses[8].p, masses[4].p],
        [masses[4].p, masses[8].p, masses[10].p, masses[6].p],
        [masses[9].p, masses[11].p, masses[10].p, masses[8].p],
        [masses[6].p, masses[7].p, masses[5].p, masses[4].p]
    ]

def get_floor_tile():
    floor_size = 2.5
    return [[-floor_size, -floor_size, 0], 
            [floor_size, -floor_size, 0], 
            [floor_size, floor_size, 0], 
            [-floor_size, floor_size, 0]]

KE_list = []
PE_list = []
TE_list = []
G_PE_list = []
S_PE_list = []

spring_initial_length_dict = {}
for spring in springs:
    spring_initial_length_dict[spring] = spring.L0
b = 0.2
c = 0.0
omega = 2*np.pi*2

t = 0.0

def simulation_step(masses, springs, dt):
    global t
    t += dt

    # Reset forces on each mass
    for mass in masses:
        mass.f = np.zeros(3, dtype=float)
        mass.f += mass.m * g  # Gravity
    
    KE = sum([0.5 * mass.m * np.linalg.norm(mass.v)**2 for mass in masses])
    PE = sum([mass.m * -g[2] * mass.p[2] for mass in masses])
    G_PE_list.append(PE)
    S_PE = 0

    # Calculate spring forces
    for spring in springs:
        spring.L0 = spring_initial_length_dict[spring] + b*np.sin(omega*t+c) 

        delta_p = spring.m1.p - spring.m2.p
        delta_length = np.linalg.norm(delta_p)
        if delta_length == 0:
            direction = np.zeros(3, dtype=float)
        else:
            direction = delta_p / delta_length
        force_magnitude = spring.k * (delta_length - spring.L0)
        force = force_magnitude * direction

        PE_spring = 0.5 * spring.k * (delta_length - spring.L0)**2
        PE += PE_spring
        S_PE += PE_spring

        # Apply spring force to masses
        spring.m1.f -= force
        spring.m2.f += force

    # Update positions and velocities for each mass
    for mass in masses:
        mass.a = mass.f / mass.m
        mass.v += mass.a * dt
        mass.p += mass.v * dt

        # Simple collision with the ground
        if mass.p[2] < 0:
            mass.p[2] = 0
            mass.v[2] = -damping * mass.v[2]  # Some damping on collision

    TE = KE + PE
    print("Total Energy: ", TE)
    print("Kinetic Energy: ", KE)
    print("Potential Energy: ", PE)

    KE_list.append(KE)
    PE_list.append(PE)
    TE_list.append(TE)
    S_PE_list.append(S_PE)


# Visualization setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize 8 points for the cube's vertices
points = [ax.plot([], [], [], 'ro')[0] for _ in range(len(masses))]

# Initialize 12 lines for the springs
lines = [ax.plot([], [], [], 'b-')[0] for _ in range(len(springs))] 
shadows = [ax.plot([], [], [], 'k-')[0] for _ in range(len(springs))] 

cube_faces_collection = Poly3DCollection(get_cube_faces(), color='cyan', alpha=0.3)
ax.add_collection3d(cube_faces_collection)

floor_tile_collection = Poly3DCollection([get_floor_tile()], color='gray', alpha=0.5)
ax.add_collection3d(floor_tile_collection)

ax.set_xlim([-2, 2]) 
ax.set_ylim([-2, 2])
ax.set_zlim([0, 4])
ax.set_xlabel('X')
ax.set_ylabel('Y')  
ax.set_zlabel('Z')
ax.set_title('Dropping and Bouncing Cube in 3D')

def init():
    for point in points:
        point.set_data([], [])
        point.set_3d_properties([])
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    for shadow in shadows:
        shadow.set_data([], [])
        shadow.set_3d_properties([])
    return points + lines + shadows

def animate(i):
    for _ in range(100):
        simulation_step(masses, springs, dt)
    
    for mass, point in zip(masses, points):
        x, y, z = mass.p
        point.set_data([x], [y])
        point.set_3d_properties([z])  # Setting the Z value for 3D

    # Update the spring lines
    for spring, line in zip(springs, lines):
        x_data = [spring.m1.p[0], spring.m2.p[0]]
        y_data = [spring.m1.p[1], spring.m2.p[1]]
        z_data = [spring.m1.p[2], spring.m2.p[2]]
        line.set_data(x_data, y_data)
        line.set_3d_properties(z_data)
    
    # Update the shadow lines
    for spring, shadow in zip(springs, shadows):
        x_data = [spring.m1.p[0], spring.m2.p[0]]
        y_data = [spring.m1.p[1], spring.m2.p[1]]
        z_data = [0, 0]
        shadow.set_data(x_data, y_data)
        shadow.set_3d_properties(z_data)
        
    # Update the cube faces
    cube_faces_collection.set_verts(get_cube_faces())

    return points + lines + shadows

ani = animation.FuncAnimation(fig, animate, frames=1000, init_func=init, blit=False, interval=5)

plt.show()

N = len(KE_list)
time_list = np.arange(0, N*dt, dt)

plt.figure()
plt.plot(time_list, KE_list, label='Kinetic Energy')
plt.plot(time_list, PE_list, label='Potential Energy')
plt.plot(time_list, TE_list, label='Total Energy')
plt.plot(time_list, G_PE_list, label='Gravitational Potential Energy')
plt.plot(time_list, S_PE_list, label='Spring Potential Energy')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy vs Time for a Bouncing Cube')
plt.legend()
plt.show()

print("Length of KE_list: ", len(KE_list))