import os
import lgsvl
import time
import psutil
import atexit
import logging
import math




def kill_mainboard():
    PROC_NAME = "mainboard"
    for proc in psutil.process_iter():
        # check whether the process to kill name matches
        if proc.name() == PROC_NAME:
            proc.kill()



def run_svl_simulation():
    log = logging.getLogger(__name__)
    atexit.register(kill_mainboard)





    SIMULATOR_HOST = os.environ.get("SIMULATOR_HOST", "127.0.0.1")
    SIMULATOR_PORT = int(os.environ.get("SIMULATOR_PORT", 8181))
    BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
    BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", 9090))

    sim = lgsvl.Simulator(SIMULATOR_HOST, SIMULATOR_PORT)
    if sim.current_scene == "BorregasAve":
        sim.reset()
    else:
        sim.load("BorregasAve")

    spawns = sim.get_spawn()

    state = lgsvl.AgentState()
    state.transform = spawns[0]
    # 5.0: 47b529db-0593-4908-b3e7-4b24a32a0f70
    # 6.0: c354b519-ccf0-4c1c-b3cc-645ed5751bb5
    # 6.0(modular testing): 2e9095fa-c9b9-4f3f-8d7d-65fa2bb03921
    # 6.0(no telephoto camera and clock sensor): 4622f73a-250e-4633-9a3d-901ede6b9551
    # 6.0(no clock sensor): f68151d1-604c-438e-a1a5-aa96d5581f4b
    # 6.0(with signal sensor): 9272dd1a-793a-45b2-bff4-3a160b506d75
    ego = sim.add_agent("9272dd1a-793a-45b2-bff4-3a160b506d75", lgsvl.AgentType.EGO, state)
    ego.connect_bridge(BRIDGE_HOST, BRIDGE_PORT)

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
        # 'Camera',
        # 'Traffic Light',
        'Control'
    ]
    destination = spawns[0].destinations[0]
    dv.setup_apollo(destination.position.x, destination.position.z, modules, default_timeout=60)
    print('finish setup_apollo')


    # x_long_east = destination.position.x
    # z_lat_north = destination.position.z
    # dv.set_destination(x_long_east, z_lat_north, y=0)


    forward = lgsvl.utils.transform_to_forward(spawns[0])
    right = lgsvl.utils.transform_to_right(spawns[0])

    print('forward', forward)
    print('right', right)



    radius = 4.5
    count = 10
    wp = []
    for i in range(count):
        x = radius * math.cos(i * 2 * math.pi / count)
        z = radius * math.sin(i * 2 * math.pi / count)
        # idle is how much time the pedestrian will wait once it reaches the waypoint
        idle = 0 if i < count // 2 else 0
        wp.append(
            lgsvl.WalkWaypoint(spawns[0].position + x * right + (z + 12) * forward, idle)
        )
    wp = [
        lgsvl.WalkWaypoint(spawns[0].position + 20 * right + 60 * forward, 0),
        lgsvl.WalkWaypoint(spawns[0].position + -20 * right + 60 * forward, 0) ]
    state = lgsvl.AgentState()
    state.transform = spawns[0]
    state.transform.position = wp[0].position
    # state.velocity.x = -3*right.x
    # state.velocity.z = -3*right.z

    p = sim.add_agent("Pamela", lgsvl.AgentType.PEDESTRIAN, state)


    def on_waypoint(agent, index):
        print("Waypoint {} reached".format(index))


    p.on_waypoint_reached(on_waypoint)

    # This sends the list of waypoints to the pedestrian. The bool controls whether or not the pedestrian will continue walking (default false)
    p.follow(wp, False)


    sim.run()



if __name__ == '__main__':
    run_svl_simulation()
