"""
data_collection.py
------------------
Run N Catan simulations (2-player, RED vs BLUE RandomPlayers) and write two
output files after every turn:

  data/catan_data.csv          flat feature table (91 columns, good for sklearn / pandas)
  data/catan_snapshots.jsonl   one JSON object per line with the complete raw game state

CSV feature groups
  game metadata     : game_id, turn, total_actions, winner, is_initial_build_phase, current_player
  bank              : bank_wood/brick/sheep/wheat/ore, dev_cards_remaining
  robber            : robber_resource
  per-player ×2     : actual_vp, visible_vp, settlements, cities, roads_placed, road_length,
                      has_longest_road, has_largest_army, settlements/cities/roads_available,
                      wood/brick/sheep/wheat/ore, total_resources,
                      knight/monopoly/road_building/year_of_plenty/vp _cards, total_dev_cards,
                      played_knights/monopoly/road_building/year_of_plenty,
                      prod_wood/brick/sheep/wheat/ore/total,
                      has_3to1/wood/brick/sheep/wheat/ore _port

JSONL extras (on top of every flat field above)
  board_layout      : [{coordinate, resource, number}, …]  one entry per land tile
  ports             : [{resource, nodes}, …]  resource=null means 3:1 port
  robber_coordinate : [x, y, z] cube coords
  buildings         : {RED: {SETTLEMENT:[…], CITY:[…], ROAD:[[n1,n2],…]}, BLUE:{…}}
  raw_player_state  : full P0_*/P1_* dict straight from state.player_state
  playable_actions  : [{color, type, value}, …]  legal moves for the current player
  last_action       : {color, type, value}  action that produced this state
"""

import csv
import json
import time
import uuid
from pathlib import Path

from catanatron import Game, RandomPlayer, Color
from catanatron.models.decks import freqdeck_count
from catanatron.models.enums import (
    CITY,
    KNIGHT,
    MONOPOLY,
    RESOURCES,
    ROAD,
    ROAD_BUILDING,
    SETTLEMENT,
    VICTORY_POINT,
    WOOD, BRICK, SHEEP, WHEAT, ORE,
    YEAR_OF_PLENTY,
)
from catanatron.models.map import Port
from catanatron.state_functions import (
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_largest_army,
    get_longest_road_color,
    get_longest_road_length,
    get_played_dev_cards,
    get_visible_victory_points,
    player_num_resource_cards,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_GAMES   = 1_000
CSV_FILE    = Path("data/catan_data.csv")
JSONL_FILE  = Path("data/catan_snapshots.jsonl")
MAX_TURNS   = 1_000
COLORS      = [Color.RED, Color.BLUE]

DICE_PROBAS = {
    2: 1/36, 3: 2/36, 4: 3/36,  5: 4/36, 6: 5/36,
    8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36,
}

# ---------------------------------------------------------------------------
# Helpers shared by both output formats
# ---------------------------------------------------------------------------

def _tile_coord_map(board):
    return {tile.id: coord for coord, tile in board.map.land_tiles.items()}


def _production(state, color, tcm):
    board        = state.board
    robber_coord = board.robber_coordinate
    prod         = {r: 0.0 for r in RESOURCES}
    for btype, mult in [(SETTLEMENT, 1), (CITY, 2)]:
        for node in state.buildings_by_color[color].get(btype, []):
            for tile in board.map.adjacent_tiles.get(node, []):
                if tile.number is None:
                    continue
                if tcm.get(tile.id) == robber_coord:
                    continue
                prod[tile.resource] = prod.get(tile.resource, 0.0) + mult * DICE_PROBAS[tile.number]
    prod["total"] = sum(prod[r] for r in RESOURCES)
    return prod


def _port_flags(board, color):
    have = set(board.get_player_port_resources(color))
    return {
        "has_3to1_port":  int(None  in have),
        "has_wood_port":  int(WOOD  in have),
        "has_brick_port": int(BRICK in have),
        "has_sheep_port": int(SHEEP in have),
        "has_wheat_port": int(WHEAT in have),
        "has_ore_port":   int(ORE   in have),
    }

# ---------------------------------------------------------------------------
# CSV snapshot  (flat feature dict)
# ---------------------------------------------------------------------------

def csv_snapshot(game, game_id, tcm):
    state = game.state
    board = state.board

    robber_tile     = board.map.land_tiles.get(board.robber_coordinate)
    robber_resource = robber_tile.resource if robber_tile else None
    army_color, _   = get_largest_army(state)
    road_color      = get_longest_road_color(state)

    row = {
        "game_id":                game_id,
        "turn":                   state.num_turns,
        "total_actions":          len(state.actions),
        "winner":                 None,
        "is_initial_build_phase": int(state.is_initial_build_phase),
        "current_player":         state.current_color().value,
        "bank_wood":              freqdeck_count(state.resource_freqdeck, WOOD),
        "bank_brick":             freqdeck_count(state.resource_freqdeck, BRICK),
        "bank_sheep":             freqdeck_count(state.resource_freqdeck, SHEEP),
        "bank_wheat":             freqdeck_count(state.resource_freqdeck, WHEAT),
        "bank_ore":               freqdeck_count(state.resource_freqdeck, ORE),
        "dev_cards_remaining":    len(state.development_listdeck),
        "robber_resource":        robber_resource,
    }

    for color in COLORS:
        p   = color.value
        idx = state.color_to_index[color]
        ps  = state.player_state
        pfx = f"P{idx}"
        prod  = _production(state, color, tcm)
        ports = _port_flags(board, color)

        row.update({
            f"{p}_actual_vp":             get_actual_victory_points(state, color),
            f"{p}_visible_vp":            get_visible_victory_points(state, color),
            f"{p}_settlements":           len(state.buildings_by_color[color].get(SETTLEMENT, [])),
            f"{p}_cities":                len(state.buildings_by_color[color].get(CITY, [])),
            f"{p}_roads_placed":          15 - ps[f"{pfx}_ROADS_AVAILABLE"],
            f"{p}_road_length":           get_longest_road_length(state, color),
            f"{p}_has_longest_road":      int(road_color == color),
            f"{p}_has_largest_army":      int(army_color == color),
            f"{p}_settlements_available": ps[f"{pfx}_SETTLEMENTS_AVAILABLE"],
            f"{p}_cities_available":      ps[f"{pfx}_CITIES_AVAILABLE"],
            f"{p}_roads_available":       ps[f"{pfx}_ROADS_AVAILABLE"],
            f"{p}_wood":                  player_num_resource_cards(state, color, WOOD),
            f"{p}_brick":                 player_num_resource_cards(state, color, BRICK),
            f"{p}_sheep":                 player_num_resource_cards(state, color, SHEEP),
            f"{p}_wheat":                 player_num_resource_cards(state, color, WHEAT),
            f"{p}_ore":                   player_num_resource_cards(state, color, ORE),
            f"{p}_total_resources":       player_num_resource_cards(state, color),
            f"{p}_knight_cards":          get_dev_cards_in_hand(state, color, KNIGHT),
            f"{p}_monopoly_cards":        get_dev_cards_in_hand(state, color, MONOPOLY),
            f"{p}_road_building_cards":   get_dev_cards_in_hand(state, color, ROAD_BUILDING),
            f"{p}_year_of_plenty_cards":  get_dev_cards_in_hand(state, color, YEAR_OF_PLENTY),
            f"{p}_vp_cards":              get_dev_cards_in_hand(state, color, VICTORY_POINT),
            f"{p}_total_dev_cards":       get_dev_cards_in_hand(state, color),
            f"{p}_played_knights":        get_played_dev_cards(state, color, KNIGHT),
            f"{p}_played_monopoly":       get_played_dev_cards(state, color, MONOPOLY),
            f"{p}_played_road_building":  get_played_dev_cards(state, color, ROAD_BUILDING),
            f"{p}_played_year_of_plenty": get_played_dev_cards(state, color, YEAR_OF_PLENTY),
            f"{p}_prod_wood":             prod[WOOD],
            f"{p}_prod_brick":            prod[BRICK],
            f"{p}_prod_sheep":            prod[SHEEP],
            f"{p}_prod_wheat":            prod[WHEAT],
            f"{p}_prod_ore":              prod[ORE],
            f"{p}_prod_total":            prod["total"],
            **{f"{p}_{k}": v for k, v in ports.items()},
        })

    return row

# ---------------------------------------------------------------------------
# JSONL snapshot  (complete raw state + all flat fields)
# ---------------------------------------------------------------------------

def _ser(v):
    """Recursively make action values JSON-serializable."""
    if v is None:
        return None
    if isinstance(v, Color):
        return v.value
    if isinstance(v, (tuple, list)):
        return [_ser(x) for x in v]
    return v  # int, str, float


def _ser_action(action):
    if action is None:
        return None
    return {
        "color": action.color.value,
        "type":  action.action_type.name,
        "value": _ser(action.value),
    }


def jsonl_snapshot(game, flat_row, tcm):
    """Extend the flat CSV row with every raw state field."""
    state = game.state
    board = state.board

    # Board tile layout (static per game, included for self-containment)
    board_layout = [
        {
            "coordinate": list(coord),
            "resource":   tile.resource,
            "number":     tile.number,
        }
        for coord, tile in board.map.land_tiles.items()
    ]

    # Ports (resource=None means 3:1)
    ports = [
        {
            "resource": tile.resource,
            "nodes":    list(tile.nodes.values()),
        }
        for tile in board.map.tiles.values()
        if isinstance(tile, Port)
    ]

    # Exact building and road positions per player
    buildings = {}
    for color in COLORS:
        bc = state.buildings_by_color.get(color, {})
        # Roads are stored as bidirectional edge tuples; deduplicate
        seen, road_list = set(), []
        for edge in bc.get(ROAD, []):
            key = tuple(sorted(edge))
            if key not in seen:
                seen.add(key)
                road_list.append(list(edge))
        buildings[color.value] = {
            "SETTLEMENT": list(bc.get(SETTLEMENT, [])),
            "CITY":       list(bc.get(CITY, [])),
            "ROAD":       road_list,
        }

    snap = dict(flat_row)   # start with all flat CSV fields
    snap.update({
        # Raw board state
        "board_layout":      board_layout,
        "ports":             ports,
        "robber_coordinate": list(board.robber_coordinate),
        "buildings":         buildings,
        # Complete player state dictionary (all P0_*/P1_* keys)
        "raw_player_state":  dict(state.player_state),
        # Current legal moves for the active player
        "playable_actions":  [_ser_action(a) for a in state.playable_actions],
        # Action that produced this state (None at game start)
        "last_action":       _ser_action(state.actions[-1]) if state.actions else None,
    })
    return snap

# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run(num_games=NUM_GAMES, csv_file=CSV_FILE, jsonl_file=JSONL_FILE):
    csv_file  = Path(csv_file)
    jsonl_file = Path(jsonl_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    t0 = time.time()

    with open(csv_file, "w", newline="") as cf, open(jsonl_file, "w") as jf:
        csv_writer = None

        for game_num in range(num_games):
            game_id = uuid.uuid4().hex[:8]
            players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
            game    = Game(players)
            tcm     = _tile_coord_map(game.state.board)

            flat_rows = []
            full_rows = []
            prev_turn = None
            winner    = None

            while True:
                winner = game.winning_color()
                if winner is not None:
                    fr = csv_snapshot(game, game_id, tcm)
                    flat_rows.append(fr)
                    full_rows.append(jsonl_snapshot(game, fr, tcm))
                    break
                if game.state.num_turns >= MAX_TURNS:
                    break

                game.play_tick()

                cur = game.state.num_turns
                if cur != prev_turn:
                    fr = csv_snapshot(game, game_id, tcm)
                    flat_rows.append(fr)
                    full_rows.append(jsonl_snapshot(game, fr, tcm))
                    prev_turn = cur

            # Stamp winner retroactively on all rows for this game
            winner_val = winner.value if winner else None
            for row in flat_rows:
                row["winner"] = winner_val
            for row in full_rows:
                row["winner"] = winner_val

            # Write CSV (lazy header on first game)
            if csv_writer is None:
                csv_writer = csv.DictWriter(cf, fieldnames=list(flat_rows[0].keys()))
                csv_writer.writeheader()
            csv_writer.writerows(flat_rows)

            # Write JSONL (one line per snapshot)
            for row in full_rows:
                jf.write(json.dumps(row) + "\n")

            total_rows += len(flat_rows)

            if (game_num + 1) % 50 == 0 or game_num == 0:
                elapsed = time.time() - t0
                rate    = (game_num + 1) / elapsed
                eta     = (num_games - game_num - 1) / rate if rate > 0 else 0
                print(
                    f"  [{game_num+1:>5}/{num_games}]  "
                    f"{total_rows:>8} rows  |  "
                    f"{rate:.1f} games/s  |  ETA {eta:.0f}s"
                )

    elapsed = time.time() - t0
    print(f"\nDone.  {num_games} games  |  {total_rows} snapshots  |  {elapsed:.1f}s")
    print(f"  CSV   -> {csv_file.resolve()}")
    print(f"  JSONL -> {jsonl_file.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Catan game-state data")
    parser.add_argument("--games",  type=int, default=NUM_GAMES,        help="Number of games")
    parser.add_argument("--csv",    type=str, default=str(CSV_FILE),    help="Output CSV path")
    parser.add_argument("--jsonl",  type=str, default=str(JSONL_FILE),  help="Output JSONL path")
    args = parser.parse_args()

    print(f"Simulating {args.games} games")
    run(num_games=args.games, csv_file=args.csv, jsonl_file=args.jsonl)
