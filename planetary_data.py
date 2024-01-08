import numpy as np
import rebound
from tqdm import tqdm

#returns dictionary with a planet's description
def create_planet(index, num_planets, af):
    return {
        'm': np.random.uniform(1e-5, 5e-5),
        'a': 1 + (af - 1) * index / (num_planets - 1),
        'e': np.random.uniform(0, 0.1),
        'theta': 0 if np.random.rand() < 0.3 else np.random.uniform(-np.pi/6, np.pi/6)
    }

#starts a simulation and returns the data list containing the position and velocity of each planet at timestep = stepsize
def planets_data(planets, stepsize):
    sim = rebound.Simulation()
    sim.add(m=1) #central mass

    for planet in planets:
        sim.add(m=planet['m'], a=planet['a'], e=planet['e'], f=planet['theta'])

    sim.dt = stepsize

    data = []

    pos, vel = [], []
    #starts at 1 to ignore central mass
    for i in range(1,len(planets)):
        pos.append([sim.particles[i].x, sim.particles[i].y])
        vel.append([sim.particles[i].vx, sim.particles[i].vy])

    data.append([pos, vel])
    return data

#creates a sample
def generate_sim():
    num_planets = np.random.randint(2, 5)
    af = np.random.uniform(1.5, 3)

    planets = [create_planet(i,num_planets,af) for i in range(num_planets)]
    stepsize = np.random.choice([0.2, 0.3, 0.5, 0.8])

    data = planets_data(planets, stepsize)

    #normalize mass and eccentricity
    for planet in planets:
        planet['m'] = np.interp(planet['m'], [1e-5, 5e-5], [1, 5])
        planet['e'] = np.interp(planet['e'], [0, 0.1], [0, 2])

    #create string
    description = {f'planet{i}': planets[i] for i in range(num_planets)}
    description['stepsize'] = stepsize
    sample = {'description': description, 'data': data}    
    return str(sample)


datasets = [('train',1000000), ('val',125000), ('test',125000)]

for name, length in datasets:
    with open(f'data/{name}.txt', 'w') as f:
        for i in tqdm(range(length)):
            sample = generate_sim()
            f.write(sample)
            f.write('\n')