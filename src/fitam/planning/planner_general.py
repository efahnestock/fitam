from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    # other, probably not used attributes
    roll: float = 0.0
    vx: float = 0.0
    wz: float = 0.0
    k: float = 0.0
    ax: float = 0.0
    alphaz: float = 0.0
    ld: float = 0.0
    ad: float = 0.0

    def __str__(self):
        return f'State(x={self.x}, y={self.y}, yaw={self.yaw})'

    def __repr__(self):
        return self.__str__()

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.yaw))

    @classmethod
    def from_state(cls, state):
        return cls(x=state.x, y=state.y, yaw=state.yaw, roll=state.roll, vx=state.vx, wz=state.wz, k=state.k, ax=state.ax, alphaz=state.alphaz, ld=state.ld, ad=state.ad)

    def from_xml(self, root):
        self.x = float(root.attrib['x'])
        self.y = float(root.attrib['y'])
        self.roll = float(root.attrib['roll'])
        self.yaw = float(root.attrib['yaw'])
        self.vx = float(root.attrib['vx'])
        self.wz = float(root.attrib['wz'])
        self.k = float(root.attrib['k'])
        self.ax = float(root.attrib['ax'])
        self.alphaz = float(root.attrib['alphaz'])
        self.ld = float(root.attrib['ld'])
        self.ad = float(root.attrib['ad'])



class PlanningState:
    def __init__(self, x=0, y=0, yaw=0, bp=None):
        self.g: float = np.inf
        self.x: int = x
        self.y: int = y
        self.yaw: int = yaw
        self.bp: PlanningState = None

    def __eq__(self, other_state) -> bool:
        if (other_state != None) and \
            (self.x == other_state.x) and \
                (self.y == other_state.y):
            # (self.yaw == other_state.yaw):
            return True

        return False

    def __str__(self):
        return f"PlanningState(x:{self.x},y:{self.y},yaw:{self.yaw},bp:{id(self.bp)},g:{self.g})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # assumes y values never reach 1e6
        return self.x * 1_000_000 + self.y
        # return hash((self.x, self.y))

def l2_dist_heuristic(state: PlanningState, goal: PlanningState):
    return np.sqrt((state.x - goal.x)**2 + (state.y - goal.y)**2)


def snap_easl_state_to_planning_state(costmap, state: State) -> PlanningState:
    ps = PlanningState()
    ps.y, ps.x = costmap.get_index_from_point((state.x, state.y))
    # NOTE: ignoring heading
    return ps


def planning_state_to_easl_state(costmap, state: PlanningState) -> State:
    x, y = costmap.get_point_from_index((state.y, state.x))
    return State(x, y)


def get_path_from_bp(last_state, costmap=None, change_to_easl=True):
    # costmap only needed if change_to_easl is True
    path = []
    acc_costs = []
    path.append(last_state)
    acc_costs.append(last_state.g)
    while path[-1].bp != None:
        path.append(path[-1].bp)
        acc_costs.append(path[-1].g)

    if change_to_easl:
        path = [planning_state_to_easl_state(costmap, ps) for ps in path]

    return list(reversed(path)), list(reversed(acc_costs))


def list_of_state_to_tuples(path: list[State], costmap=None, convert_to_ps=False) -> list[tuple]:
    if convert_to_ps:
        path = [snap_easl_state_to_planning_state(costmap, state) for state in path]
        return [(state.y, state.x) for state in path]
    else:
        return [(state.x, state.y) for state in path]


def get_acc_costs(path: List[Tuple[int, int]], cost_array: np.ndarray, resolution: float) -> list:
    cost = np.zeros(len(path))
    # cost = np.zeros(len(path), dtype=np.float32)
    for item_index in range(1, len(path)):  # don't add cost for the start node
        cost_multiplier = 1
        # if diagonal, multiply by sqrt(2)
        if (path[item_index][0] != path[item_index-1][0]) and (path[item_index][1] != path[item_index-1][1]):
            cost_multiplier = np.sqrt(2)
        cost[item_index] = cost_array[path[item_index][0], path[item_index][1]] * resolution * cost_multiplier
    return np.cumsum(cost)
