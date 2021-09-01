import os
import lgsvl
import time
import psutil
import atexit
import math





def kill_mainboard():
    PROC_NAME = "mainboard"
    for proc in psutil.process_iter():
        # check whether the process to kill name matches
        if proc.name() == PROC_NAME:
            proc.kill()

def on_waypoint(agent, index):
    print("Waypoint {} reached".format(index))


def on_collision(agent1, agent2, contact):
  name1 = "STATIC OBSTACLE" if agent1 is None else agent1.name
  name2 = "STATIC OBSTACLE" if agent2 is None else agent2.name
  print("{} collided with {} at {}".format(name1, name2, contact))
  print('v_ego:', agent1.velocity)


def initialize_simulator_and_dv(map, sim):
    SIMULATOR_HOST = os.environ.get("SIMULATOR_HOST", "127.0.0.1")
    SIMULATOR_PORT = int(os.environ.get("SIMULATOR_PORT", 8181))
    BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
    BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", 9090))
    if not sim:

        sim = lgsvl.Simulator(SIMULATOR_HOST, SIMULATOR_PORT)
        print('dir(sim)', dir(sim))
        print('sim.current_scene', sim.current_scene)
        if sim.current_scene == map:
            sim.reset()
        else:
            # seed make sure the weather and NPC behvaiors deterministic
            sim.load(map, seed=0)


    spawns = sim.get_spawn()
    state = lgsvl.AgentState()
    state.transform = spawns[0]

    # 5.0: 47b529db-0593-4908-b3e7-4b24a32a0f70
    # 6.0: c354b519-ccf0-4c1c-b3cc-645ed5751bb5
    # 6.0(modular testing): 2e9095fa-c9b9-4f3f-8d7d-65fa2bb03921
    # 6.0(no telephoto camera and clock sensor): 4622f73a-250e-4633-9a3d-901ede6b9551
    # 6.0(no clock sensor): f68151d1-604c-438e-a1a5-aa96d5581f4b
    # 6.0(with signal sensor): 9272dd1a-793a-45b2-bff4-3a160b506d75
    ego = sim.add_agent("2e9095fa-c9b9-4f3f-8d7d-65fa2bb03921", lgsvl.AgentType.EGO, state)
    ego.connect_bridge(BRIDGE_HOST, BRIDGE_PORT)
    ego.on_collision(on_collision)

    # Dreamview setup
    dv = lgsvl.dreamview.Connection(sim, ego, BRIDGE_HOST)
    dv.set_hd_map('Borregas Ave')
    dv.set_vehicle('Lincoln2017MKZ_LGSVL')
    modules = [
        'Localization',
        'Perception',
        'Transform',
        'Routing',
        'Prediction',
        'Planning',
        'Camera',
        # 'Traffic Light',
        'Control'
    ]

    start = lgsvl.Transform(position=ego.state.transform.position, rotation=ego.state.transform.rotation)
    print('start', start)
    # destination = spawns[0].destinations[0]
    destination = lgsvl.Transform(position=lgsvl.Vector(24.970,-2.615,-29.956), rotation=lgsvl.Vector(0.731,104.547,358.660))

    print('destination', destination)
    dv.setup_apollo(destination.position.x, destination.position.z, modules, default_timeout=60)
    print('finish setup_apollo')

    return sim, ego, start, destination




def run_svl_simulation(map, config, sim):


    sim, ego, start, destination = initialize_simulator_and_dv(map, sim)


    middle_point = lgsvl.Transform(position=(destination.position + start.position) * 0.5, rotation=start.rotation)



    ped_x, ped_z, ped_yaw, ped_speed, ped_trigger_distance, ped_travel_distance = config
    ped_position_offset = lgsvl.Vector(ped_x, 0, ped_z)
    ped_rotation_offset = lgsvl.Vector(0, ped_yaw, 0)
    ped_point = lgsvl.Transform(position=middle_point.position+ped_position_offset, rotation=middle_point.rotation+ped_rotation_offset)


    forward = lgsvl.utils.transform_to_forward(ped_point)


    wp = [
        lgsvl.WalkWaypoint(ped_point.position, 0, ped_trigger_distance),
        lgsvl.WalkWaypoint(ped_point.position + ped_travel_distance * forward, 0, 0) ]
    state = lgsvl.AgentState()
    state.transform = ped_point
    state.velocity = ped_speed * forward

    p = sim.add_agent("Pamela", lgsvl.AgentType.PEDESTRIAN, state)
    p.on_waypoint_reached(on_waypoint)

    # This sends the list of waypoints to the pedestrian. The bool controls whether or not the pedestrian will continue walking (default false)
    p.follow(wp, False)



    t0 = time.time()
    s0 = sim.current_time
    print()
    print("Total real time elapsed = {:5.3f}".format(0))
    print("Simulation time = {:5.1f}".format(s0))
    print("Simulation frames =", sim.current_frame)

    # let simulator initialize and settle a bit before starting
    # sim.run(time_limit=2)

    t1 = time.time()
    s1 = sim.current_time

    duration = 20
    step_time = 1
    step_rate = int(1.0 / step_time)
    steps = duration * step_rate

    t0 = time.time()
    for i in range(steps):
        t1 = time.time()
        sim.run(time_limit=step_time, time_scale=1)
        t2 = time.time()
        s2 = sim.current_time

        state = ego.state
        pos = state.position
        rot = state.rotation
        speed = state.speed * 3.6


        print("Sim time = {:5.2f}".format(s2 - s1) + "; Real time elapsed = {:5.3f}; ".format(t2 - t1), end='')
        print("Speed = {:4.1f}; Position = {:5.3f},{:5.3f},{:5.3f}; Rotation = {:5.3f},{:5.3f},{:5.3f}".format(speed, pos.x, pos.y, pos.z, rot.x, rot.y, rot.z))
        # for sensor in ego.get_sensors():
        #     print(sensor)
        #     if sensor.name == "Main Camera":
        #         sensor.save("main-camera_"+str(i)+".png", compression=0)
        #         print('save image')

        # time.sleep(0.2)

    t3 = time.time()
    sim.reset()
    return sim



if __name__ == '__main__':
    map = "BorregasAve"
    config = [4, 4, 2, 3, 10, 50]
    atexit.register(kill_mainboard)
    sim = run_svl_simulation(map, config, None)
    run_svl_simulation(map, config, sim)
