from catanatron import Game, RandomPlayer, Color
from catanatron.state_functions import (
    get_actual_victory_points,
    get_longest_road_length,
    player_num_resource_cards,
)
from catanatron.models.enums import SETTLEMENT, CITY, WOOD, BRICK, SHEEP, WHEAT, ORE

COLORS = [Color.RED, Color.BLUE]
MAX_TURNS = 1000


def get_player_snapshot(state, color):
    return {
        "victory_points": get_actual_victory_points(state, color),
        "settlements": len(state.buildings_by_color[color].get(SETTLEMENT, [])),
        "cities": len(state.buildings_by_color[color].get(CITY, [])),
        "road_length": get_longest_road_length(state, color),
        "resources": {
            "WOOD": player_num_resource_cards(state, color, WOOD),
            "BRICK": player_num_resource_cards(state, color, BRICK),
            "SHEEP": player_num_resource_cards(state, color, SHEEP),
            "WHEAT": player_num_resource_cards(state, color, WHEAT),
            "ORE": player_num_resource_cards(state, color, ORE),
        },
    }


def record_snapshot(game, winner=None):
    state = game.state
    snap = {"turn": state.num_turns, "winner": winner}
    for color in COLORS:
        snap[color.value] = get_player_snapshot(state, color)
    return snap


def print_snapshot(snap):
    print(f"\n--- Turn {snap['turn']} ---")
    for color in COLORS:
        p = snap[color.value]
        res = p["resources"]
        print(
            f"  {color.value:4s}: VP={p['victory_points']}  "
            f"Settlements={p['settlements']}  Cities={p['cities']}  "
            f"Road={p['road_length']}  "
            f"W={res['WOOD']} B={res['BRICK']} Sh={res['SHEEP']} Wh={res['WHEAT']} O={res['ORE']}"
        )
    if snap["winner"]:
        print(f"  >> WINNER: {snap['winner']} <<")


players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
game = Game(players)

snapshots = []
prev_turn = None
winner = None

while True:
    winner = game.winning_color()
    if winner is not None:
        snap = record_snapshot(game, winner=winner.value)
        snapshots.append(snap)
        print_snapshot(snap)
        break

    if game.state.num_turns >= MAX_TURNS:
        print(f"\nGame truncated at {MAX_TURNS} turns with no winner.")
        break

    game.play_tick()

    current_turn = game.state.num_turns
    if current_turn != prev_turn:
        snap = record_snapshot(game)
        snapshots.append(snap)
        print_snapshot(snap)
        prev_turn = current_turn

print(f"\nGame over! Total turns: {game.state.num_turns}")
print(f"Winner: {winner}")
print(f"Total snapshots recorded: {len(snapshots)}")
