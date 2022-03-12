def tic_tac_toe(board):
    for i in board:
        
        if i.count('X') == 3:
            return 'X'
        if i.count('O') == 3:
            return 'O'

    for i in [[row[col] for row in board] for col in range(3)]:
        if i.count('X') == 3:
            return 'X'
        if i.count('O') == 3:
            return 'O'

    diagonal = [[board[i][i] for i in range(3)], [board[i][-(i+1)] for i in range(3)]]
    for i in diagonal:
        if i.count('X') == 3:
            return 'X'
        if i.count('O') == 3:
            return 'O'

    return 'Draw'
    
          
print(tic_tac_toe([
  ["X", "O", "X"],
  ["O", "X",  "O"],
  ["O", "X",  "X"]
]))
