from project.scripts.web_play import init_game
from project.scripts import web_play

# Start game and get env
init_game(True)
env = web_play.game_env
s = env.current_state_of_the_game

# Print initial gem counts
print('initial gems', s.active_players_hand().gems_possessed.to_dict())

# Add a lot of gems
hand = s.active_players_hand().gems_possessed
hand_dict = hand.to_dict()
for c in hand_dict:
    hand_dict[c] += 10
s.active_players_hand().gems_possessed = type(hand).from_dict(hand_dict)

web_play.legal_actions = env.action_space.list_of_actions
print('total actions', len(web_play.legal_actions))

buy = [(i,a) for i,a in enumerate(web_play.legal_actions) if a.action_type=='buy']
print('buy actions', len(buy))
if buy:
    i,a = buy[0]
    print('first buy', i, 'card id', a.card.id, 'price', a.card.price.to_dict())
    print('board cards', [c.id for c in s.board.cards_on_board])
