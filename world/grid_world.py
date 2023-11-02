import numpy as np
import random
from .entity import Entity


class GridWorld:
    def __init__(self, map_loader, playable_team_size, playable_teams_num, spawn_bonus_every=-1):
        self.map_loader = map_loader
        self.playable_team_size = playable_team_size
        self.playable_teams_num = playable_teams_num
        self.team_spawn_coordinates = [[] for _ in range(self.playable_teams_num)]
        self.teams = [dict([(j, Entity(0, 0, j, i, False)) for j in range(self.playable_team_size)]) for i in range(self.playable_teams_num)]
        self.random = random.Random()
        self.eaten_preys = set()
        self.preys = []
        self.actions = {}
        self._steps = 0
        self.spawn_bonus_every = spawn_bonus_every

    def step(self):
        self._steps += 1
        initiative = [(t, ent) for t in range(self.playable_teams_num) for ent in self.teams[t]]
        eaten = dict()
        step_result = {i:
            {   
                'true_action': [0] * self.playable_team_size,
                'eaten_preys': [0] * self.playable_team_size,
                'eaten_enemies': [0] * self.playable_team_size,
                'bonus_on_step': [0] * self.playable_team_size,
                'team_attack': [0] * self.playable_team_size,
                'on_team_attack': [0] * self.playable_team_size,
                'dead_members': [0] * self.playable_team_size,
            } for i in range(self.playable_teams_num)
        }
        # Predators move
        self.random.shuffle(initiative)
        for i in initiative:
            action = self.actions.get(i, 0)
            if action == 0 or not self.teams[i[0]][i[1]].alive:
                continue
            ent = self.teams[i[0]][i[1]]
            x, y, tx, ty = self._action_coord_change(ent, action)
            if self.map[ty, tx][0] >= 0 and self.map[ty, tx][0] != i[0]:
                if self.map[ty, tx][0] == self.playable_teams_num:
                    self.eaten_preys.add(tuple(self.map[ty, tx]))
                    self.preys[self.map[ty, tx][1]].alive = False
                    step_result[ent.team]['eaten_preys'][ent.idx] += 1
                else:
                    other_ent = self.teams[self.map[ty, tx][0]][self.map[ty, tx][1]]
                    if other_ent.bonus_count > 0:
                        other_ent.bonus_count -= 1
                        step_result[ent.team]['team_attack'][ent.idx] += 1
                        step_result[other_ent.team]['on_team_attack'][other_ent.idx] += 1
                        continue
                    else:
                        self.teams[self.map[ty, tx][0]][self.map[ty, tx][1]].alive = False
                        step_result[ent.team]['eaten_enemies'][ent.idx] += 1
                        step_result[other_ent.team]['dead_members'][other_ent.idx] += 1

                eaten[tuple(self.map[ty, tx])] = i
                self.map[y, x] = np.array((-1, 0), dtype=np.int64)
                self.map[ty, tx] = np.array(i, dtype=np.int64)
                ent.x, ent.y = tx, ty
            else:
                if self.map[ty, tx][0] == -1:
                    if self.map[ty, tx][1] == 1:
                        ent.bonus_count += 1
                        step_result[ent.team]['bonus_on_step'][ent.idx] += 1
                        self.map[ty, tx][1] = 0
                    if self.map[ty, tx][1] == 0:
                        self.map[y, x] = np.array((-1, 0), dtype=np.int64)
                        self.map[ty, tx] = np.array(i, dtype=np.int64)
                        ent.x, ent.y = tx, ty
                        step_result[ent.team]['true_action'][ent.idx] += 1

        # Preys move
        for prey in self.preys:
            if not prey.alive:
                continue
            action = self.actions.get((self.playable_teams_num, prey.idx), 0)
            x, y, tx, ty = self._action_coord_change(prey, action)
            if self.map[ty, tx][0] == -1:
                if self.map[ty, tx][1] == 0:
                    self.map[y, x] = np.array((-1, 0), dtype=np.int64)
                    self.map[ty, tx] = np.array((self.playable_teams_num, prey.idx), dtype=np.int64)
                    prey.x, prey.y = tx, ty

        # Respawn predators
        for i in range(self.playable_teams_num):
            self._spawn_team(i)

        if self.spawn_bonus_every > 0 and self._steps % self.spawn_bonus_every == 0:
            self._spawn_bonus()

        return eaten, step_result

    def set_actions(self, team_idx2action):
        self.actions.update(team_idx2action)
        for i in range(len(self.preys)):
            self.actions[(self.playable_teams_num, self.preys[i].idx)] = self.random.randint(0, 4)

    def reset(self, seed=None):
        self.actions.clear()
        self.eaten_preys.clear()
        self.random = random.Random(seed)
        self.base_map = self.map_loader.load_next(seed)
        self._build_map()
        for i in range(self.playable_teams_num):
            self._spawn_team(i)

        if self.spawn_bonus_every > 0:
            for _ in range(self.playable_teams_num * self.playable_team_size):
                self._spawn_bonus()

    def _action_coord_change(self, ent, action):
        x, y = ent.x, ent.y
        tx, ty = x, y
        if action == 1:
            tx = (x + 1) % self.map.shape[1]
        elif action == 2:
            tx = (self.map.shape[0] + x - 1) % self.map.shape[1]
        elif action == 3:
            ty = (self.map.shape[0] + y - 1) % self.map.shape[0]
        elif action == 4:
            ty = (y + 1) % self.map.shape[0]
        return x, y, tx, ty

    def _spawn_bonus(self):
        x, y = np.random.randint(0, self.map.shape[1]), np.random.randint(0, self.map.shape[0])
        while self.map[y][x][0] != -1 or self.map[y][x][1] != 0:
            x, y = np.random.randint(0, self.map.shape[1]), np.random.randint(0, self.map.shape[0])

        self.map[y][x][0] = -1
        self.map[y][x][1] = 1

    def _spawn_team(self, team):
        entities_for_respawn = [e for e in self.teams[team].values() if not e.alive]
        if len(entities_for_respawn) == 0:
            return 0
        free_spaces = [(x, y) for x, y in self.team_spawn_coordinates[team] if self.map[y, x][0] == -1]
        self.random.shuffle(free_spaces)
        spawned = 0
        for e, space in zip(entities_for_respawn, free_spaces):
            x, y = space
            self.map[y, x] = np.array((team, e.idx))
            e.alive = True
            e.x, e.y = x, y
            e.bonus_count = 0
            spawned += 1
        return spawned

    def _build_map(self):
        self.map = np.zeros((self.base_map.shape[0], self.base_map.shape[1], 2), np.int32)
        self.preys.clear()
        for sc in self.team_spawn_coordinates:
            sc.clear()
        for team in self.teams:
            for e in team.values():
                e.alive = False
        for x in range(self.base_map.shape[1]):
            for y in range(self.base_map.shape[0]):
                if self.base_map[y, x] == -1:
                    self.map[y, x] = np.array([-1, -1], dtype=np.int64) # Unpassable tiles
                    continue # Passable tile
                elif self.base_map[y, x] == -2:
                    self.map[y, x] = np.array((self.playable_teams_num, len(self.preys)), dtype=np.int64)
                    self.preys.append(Entity(x, y, len(self.preys), self.playable_teams_num, True))
                    continue
                elif 0 <= self.base_map[y, x] <= self.playable_teams_num:
                    self.map[y, x] = np.array([-1, 0], dtype=np.int64)
                    if 0 < self.base_map[y, x] <= self.playable_teams_num:
                        self.team_spawn_coordinates[self.base_map[y, x]-1].append((x, y))  # Add Spawn point
                else:
                    raise f"Unknown value ({self.base_map[y, x]}) at {x}, {y}. Abort."

