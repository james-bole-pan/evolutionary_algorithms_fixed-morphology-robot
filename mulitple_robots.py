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
masses_1 = [
    Mass([-half_L0-1.0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 0
    Mass([half_L0-1.0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 1
    Mass([-half_L0-1.0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 2
    Mass([half_L0-1.0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 3
    Mass([-half_L0-1.0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 4
    Mass([half_L0-1.0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 5
    Mass([-half_L0-1.0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 6
    Mass([half_L0-1.0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),      # 7
    Mass([-half_L0-1.0, half_L0 + L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 8
    Mass([half_L0-1.0, half_L0 + L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 9
    Mass([-half_L0-1.0, half_L0 + L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 10
    Mass([half_L0-1.0, half_L0 + L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 11
]

masses_2 = [
    Mass([-half_L0+2.0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 0
    Mass([half_L0+2.0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 1
    Mass([-half_L0+2.0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 2
    Mass([half_L0+2.0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 3
    Mass([-half_L0+2.0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 4
    Mass([half_L0+2.0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 5
    Mass([-half_L0+2.0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 6
    Mass([half_L0+2.0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),      # 7
    Mass([-half_L0+2.0, half_L0 + L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 8
    Mass([half_L0+2.0, half_L0 + L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 9
    Mass([-half_L0+2.0, half_L0 + L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 10
    Mass([half_L0+2.0, half_L0 + L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 11
]

# Connect the masses with springs to form a cube
short_diag_length = np.sqrt(2 * L0**2)
long_diag_length = np.sqrt(3 * L0**2)

springs_1 = [
    Spring(L0, k, masses_1[0], masses_1[1]),  # Base square
    Spring(L0, k, masses_1[1], masses_1[3]),
    Spring(L0, k, masses_1[3], masses_1[2]),
    Spring(L0, k, masses_1[2], masses_1[0]),
    Spring(L0, k, masses_1[4], masses_1[5]),  # Top square
    Spring(L0, k, masses_1[5], masses_1[7]),
    Spring(L0, k, masses_1[7], masses_1[6]),
    Spring(L0, k, masses_1[6], masses_1[4]),
    Spring(L0, k, masses_1[0], masses_1[4]),  # Vertical edges
    Spring(L0, k, masses_1[1], masses_1[5]),
    Spring(L0, k, masses_1[2], masses_1[6]),
    Spring(L0, k, masses_1[3], masses_1[7]),
    Spring(short_diag_length, k, masses_1[0], masses_1[3]),
    Spring(short_diag_length, k, masses_1[1], masses_1[2]),
    Spring(short_diag_length, k, masses_1[4], masses_1[7]),
    Spring(short_diag_length, k, masses_1[5], masses_1[6]),
    # Short Diagonals between opposite faces
    Spring(short_diag_length, k, masses_1[0], masses_1[5]),
    Spring(short_diag_length, k, masses_1[1], masses_1[4]),
    Spring(short_diag_length, k, masses_1[2], masses_1[7]),
    Spring(short_diag_length, k, masses_1[3], masses_1[6]),
    Spring(short_diag_length, k, masses_1[1], masses_1[7]),
    Spring(short_diag_length, k, masses_1[0], masses_1[6]),
    Spring(short_diag_length, k, masses_1[3], masses_1[5]),
    Spring(short_diag_length, k, masses_1[2], masses_1[4]),
    Spring(long_diag_length, k, masses_1[0], masses_1[7]),
    Spring(long_diag_length, k, masses_1[1], masses_1[6]),
    Spring(long_diag_length, k, masses_1[2], masses_1[5]),
    Spring(long_diag_length, k, masses_1[3], masses_1[4]),
    Spring(L0, k, masses_1[8], masses_1[9]), 
    Spring(L0, k, masses_1[9], masses_1[11]),
    Spring(L0, k, masses_1[11], masses_1[10]),
    Spring(L0, k, masses_1[10], masses_1[8]),
    Spring(L0, k, masses_1[6], masses_1[10]),
    Spring(L0, k, masses_1[7], masses_1[11]),
    Spring(L0, k, masses_1[4], masses_1[8]),
    Spring(L0, k, masses_1[5], masses_1[9]),
    Spring(short_diag_length, k, masses_1[6], masses_1[11]),
    Spring(short_diag_length, k, masses_1[7], masses_1[10]),
    Spring(short_diag_length, k, masses_1[4], masses_1[9]),
    Spring(short_diag_length, k, masses_1[5], masses_1[8]),
    Spring(short_diag_length, k, masses_1[4], masses_1[10]),
    Spring(short_diag_length, k, masses_1[5], masses_1[11]),
    Spring(short_diag_length, k, masses_1[6], masses_1[8]),
    Spring(short_diag_length, k, masses_1[7], masses_1[9]),
    Spring(short_diag_length, k, masses_1[9], masses_1[10]),
    Spring(short_diag_length, k, masses_1[8], masses_1[11]),
    Spring(long_diag_length, k, masses_1[6], masses_1[9]),
    Spring(long_diag_length, k, masses_1[7], masses_1[8]),
    Spring(long_diag_length, k, masses_1[4], masses_1[11]),
    Spring(long_diag_length, k, masses_1[5], masses_1[10])
]

springs_2 = [
    Spring(L0, k, masses_2[0], masses_2[1]),  # Base square
    Spring(L0, k, masses_2[1], masses_2[3]),
    Spring(L0, k, masses_2[3], masses_2[2]),
    Spring(L0, k, masses_2[2], masses_2[0]),
    Spring(L0, k, masses_2[4], masses_2[5]),  # Top square
    Spring(L0, k, masses_2[5], masses_2[7]),
    Spring(L0, k, masses_2[7], masses_2[6]),
    Spring(L0, k, masses_2[6], masses_2[4]),
    Spring(L0, k, masses_2[0], masses_2[4]),  # Vertical edges
    Spring(L0, k, masses_2[1], masses_2[5]),
    Spring(L0, k, masses_2[2], masses_2[6]),
    Spring(L0, k, masses_2[3], masses_2[7]),
    Spring(short_diag_length, k, masses_2[0], masses_2[3]),
    Spring(short_diag_length, k, masses_2[1], masses_2[2]),
    Spring(short_diag_length, k, masses_2[4], masses_2[7]),
    Spring(short_diag_length, k, masses_2[5], masses_2[6]),
    # Short Diagonals between opposite faces
    Spring(short_diag_length, k, masses_2[0], masses_2[5]),
    Spring(short_diag_length, k, masses_2[1], masses_2[4]),
    Spring(short_diag_length, k, masses_2[2], masses_2[7]),
    Spring(short_diag_length, k, masses_2[3], masses_2[6]),
    Spring(short_diag_length, k, masses_2[1], masses_2[7]),
    Spring(short_diag_length, k, masses_2[0], masses_2[6]),
    Spring(short_diag_length, k, masses_2[3], masses_2[5]),
    Spring(short_diag_length, k, masses_2[2], masses_2[4]),
    Spring(long_diag_length, k, masses_2[0], masses_2[7]),
    Spring(long_diag_length, k, masses_2[1], masses_2[6]),
    Spring(long_diag_length, k, masses_2[2], masses_2[5]),
    Spring(long_diag_length, k, masses_2[3], masses_2[4]),
    Spring(L0, k, masses_2[8], masses_2[9]), 
    Spring(L0, k, masses_2[9], masses_2[11]),
    Spring(L0, k, masses_2[11], masses_2[10]),
    Spring(L0, k, masses_2[10], masses_2[8]),
    Spring(L0, k, masses_2[6], masses_2[10]),
    Spring(L0, k, masses_2[7], masses_2[11]),
    Spring(L0, k, masses_2[4], masses_2[8]),
    Spring(L0, k, masses_2[5], masses_2[9]),
    Spring(short_diag_length, k, masses_2[6], masses_2[11]),
    Spring(short_diag_length, k, masses_2[7], masses_2[10]),
    Spring(short_diag_length, k, masses_2[4], masses_2[9]),
    Spring(short_diag_length, k, masses_2[5], masses_2[8]),
    Spring(short_diag_length, k, masses_2[4], masses_2[10]),
    Spring(short_diag_length, k, masses_2[5], masses_2[11]),
    Spring(short_diag_length, k, masses_2[6], masses_2[8]),
    Spring(short_diag_length, k, masses_2[7], masses_2[9]),
    Spring(short_diag_length, k, masses_2[9], masses_2[10]),
    Spring(short_diag_length, k, masses_2[8], masses_2[11]),
    Spring(long_diag_length, k, masses_2[6], masses_2[9]),
    Spring(long_diag_length, k, masses_2[7], masses_2[8]),
    Spring(long_diag_length, k, masses_2[4], masses_2[11]),
    Spring(long_diag_length, k, masses_2[5], masses_2[10])
]

def get_cube_faces():
    return [
        [masses_1[0].p, masses_1[1].p, masses_1[5].p, masses_1[4].p],  # Bottom face
        [masses_1[2].p, masses_1[3].p, masses_1[7].p, masses_1[6].p],  # Top face
        [masses_1[0].p, masses_1[1].p, masses_1[3].p, masses_1[2].p],  # Front face
        [masses_1[0].p, masses_1[4].p, masses_1[6].p, masses_1[2].p],  # Left face
        [masses_1[1].p, masses_1[5].p, masses_1[7].p, masses_1[3].p],   # Right face
        [masses_1[6].p, masses_1[7].p, masses_1[11].p, masses_1[10].p],
        [masses_1[7].p, masses_1[11].p, masses_1[9].p, masses_1[5].p],
        [masses_1[5].p, masses_1[9].p, masses_1[8].p, masses_1[4].p],
        [masses_1[4].p, masses_1[8].p, masses_1[10].p, masses_1[6].p],
        [masses_1[9].p, masses_1[11].p, masses_1[10].p, masses_1[8].p],
        [masses_1[6].p, masses_1[7].p, masses_1[5].p, masses_1[4].p],
        [masses_2[0].p, masses_2[1].p, masses_2[5].p, masses_2[4].p],  # Bottom face
        [masses_2[2].p, masses_2[3].p, masses_2[7].p, masses_2[6].p],  # Top face
        [masses_2[0].p, masses_2[1].p, masses_2[3].p, masses_2[2].p],  # Front face
        [masses_2[0].p, masses_2[4].p, masses_2[6].p, masses_2[2].p],  # Left face
        [masses_2[1].p, masses_2[5].p, masses_2[7].p, masses_2[3].p],   # Right face
        [masses_2[6].p, masses_2[7].p, masses_2[11].p, masses_2[10].p],
        [masses_2[7].p, masses_2[11].p, masses_2[9].p, masses_2[5].p],
        [masses_2[5].p, masses_2[9].p, masses_2[8].p, masses_2[4].p],
        [masses_2[4].p, masses_2[8].p, masses_2[10].p, masses_2[6].p],
        [masses_2[9].p, masses_2[11].p, masses_2[10].p, masses_2[8].p],
        [masses_2[6].p, masses_2[7].p, masses_2[5].p, masses_2[4].p]
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
spring_b_list = {}
spring_c_list = {}
omega = 2*np.pi*2
for spring in springs_1:
    spring_initial_length_dict[spring] = spring.L0
    spring_b_list[spring] = np.random.uniform(0.2, 0.3)
    spring_c_list[spring] = np.random.uniform(0, 2*np.pi*0.1)
    spring.k = np.random.uniform(1000, 1200)

for spring in springs_2:
    spring_initial_length_dict[spring] = spring.L0
    spring_b_list[spring] = np.random.uniform(0.2, 0.3)
    spring_c_list[spring] = np.random.uniform(0, 2*np.pi*0.1)
    spring.k = np.random.uniform(1000, 1200)

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
        b = spring_b_list[spring]
        c = spring_c_list[spring]
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
points_1 = [ax.plot([], [], [], 'ro')[0] for _ in range(len(masses_1))]

# Initialize 12 lines for the springs
lines_1 = [ax.plot([], [], [], 'b-')[0] for _ in range(len(springs_1))] 
shadows_1 = [ax.plot([], [], [], 'k-')[0] for _ in range(len(springs_1))] 

# Initialize 8 points for the cube's vertices
points_2 = [ax.plot([], [], [], 'ro')[0] for _ in range(len(masses_2))]

# Initialize 12 lines for the springs
lines_2 = [ax.plot([], [], [], 'b-')[0] for _ in range(len(springs_2))] 
shadows_2 = [ax.plot([], [], [], 'k-')[0] for _ in range(len(springs_2))] 

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
ax.set_title('Multiple Robots')

def init():
    for point in points_1:
        point.set_data([], [])
        point.set_3d_properties([])
    for line in lines_1:
        line.set_data([], [])
        line.set_3d_properties([])
    for shadow in shadows_1:
        shadow.set_data([], [])
        shadow.set_3d_properties([])
    for point in points_2:
        point.set_data([], [])
        point.set_3d_properties([])
    for line in lines_2:
        line.set_data([], [])
        line.set_3d_properties([])
    for shadow in shadows_2:
        shadow.set_data([], [])
        shadow.set_3d_properties([])
    return points_1 + lines_1 + shadows_1 + points_2 + lines_2 + shadows_2

def animate(i):
    for _ in range(100):
        simulation_step(masses_1, springs_1, dt)
        simulation_step(masses_2, springs_2, dt)
    
    for mass, point in zip(masses_1, points_1):
        x, y, z = mass.p
        point.set_data([x], [y])
        point.set_3d_properties([z])  # Setting the Z value for 3D

    for mass, point in zip(masses_2, points_2):
        x, y, z = mass.p
        point.set_data([x], [y])
        point.set_3d_properties([z])

    # Update the spring lines
    for spring, line in zip(springs_1, lines_1):
        x_data = [spring.m1.p[0], spring.m2.p[0]]
        y_data = [spring.m1.p[1], spring.m2.p[1]]
        z_data = [spring.m1.p[2], spring.m2.p[2]]
        line.set_data(x_data, y_data)
        line.set_3d_properties(z_data)
    
    for spring, line in zip(springs_2, lines_2):
        x_data = [spring.m1.p[0], spring.m2.p[0]]
        y_data = [spring.m1.p[1], spring.m2.p[1]]
        z_data = [spring.m1.p[2], spring.m2.p[2]]
        line.set_data(x_data, y_data)
        line.set_3d_properties(z_data)

    # Update the shadow lines
    for spring, shadow in zip(springs_1, shadows_1):
        x_data = [spring.m1.p[0], spring.m2.p[0]]
        y_data = [spring.m1.p[1], spring.m2.p[1]]
        z_data = [0, 0]
        shadow.set_data(x_data, y_data)
        shadow.set_3d_properties(z_data)
    
    for spring, shadow in zip(springs_2, shadows_2):
        x_data = [spring.m1.p[0], spring.m2.p[0]]
        y_data = [spring.m1.p[1], spring.m2.p[1]]
        z_data = [0, 0]
        shadow.set_data(x_data, y_data)
        shadow.set_3d_properties(z_data)

    # Update the cube faces
    cube_faces_collection.set_verts(get_cube_faces())

    return points_1 + lines_1 + shadows_1 + points_2 + lines_2 + shadows_2

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