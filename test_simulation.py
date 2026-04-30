from catanatron import Game, RandomPlayer, Color

players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
]

game = Game(players)
winner = game.play()
print("Winner:", winner)