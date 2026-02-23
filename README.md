Forked from TomaszOdrzygozdz/gym-splendor

Environment set up: pip install -e .

Explanation feature:
1. State vector: 40-dim
     Dimension Dictionary:
    [Public Board] (0-5)
        [0-5]: Available Gems (Red, Blue, Green, White, Black, Gold)
    [Active Player] (6-19) - Always positioned first (Canonical perspective)
        [6]: Score / Victory Points
        [7-12]: Gems Possessed
        [13-18]: Card Discounts / Engines
        [19]: Number of Reserved Cards
    [Opponent] (20-33)
        [20]: Score / Victory Points
        [21-26]: Gems Possessed
        [27-32]: Card Discounts / Engines
        [33]: Number of Reserved Cards    
    [Padding] (34-39)
        [34-39]: Padding zeros for future feature expansion (e.g., Nobles)

2. Event vector: 9-dim
    Dimension Dictionary:
    [Base Actions]
        [0]: Is_Take_Gems -> Player collected gems from the board
        [1]: Is_Buy_Card  -> Player purchased a card
        [2]: Is_Reserve   -> Player reserved a card 
    [Goal Actions]
        [3]: Is_Score_Up  -> Action immediately increased Victory Points
        [4]: Is_Lethal    -> Action reached the winning threshold (>=15 VP)     
    [Defensive Tactics]
        [5]: Scarcity_Take -> Monopolized a scarce resource (took from <=2 remaining)
        [6]: Block_Reserve -> Reserved a card while opponent is close to winning (>=10 VP)
    [Offensive Tactics]
        [7]: Buy_Reserved  -> Played a previously reserved card (cleared hand space)
        [8]: Engine_Spike  -> Huge single-turn score jump (>=3 VP, implies Nobles or high-tier cards)
    

