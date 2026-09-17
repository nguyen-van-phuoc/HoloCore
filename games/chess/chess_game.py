import chess

class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    def get_board(self):
        return self.board

    def get_piece_at(self, square_name: str):
        try:
            square = chess.parse_square(square_name)
            return self.board.piece_at(square)
        except ValueError:
            return None

    def get_legal_moves(self, square_name: str):
        try:
            square = chess.parse_square(square_name)
        except ValueError:
            return []

        return [chess.square_name(move.to_square) for move in self.board.legal_moves if move.from_square == square]

    def is_legal_move(self, from_square: str, to_square: str):
        try:
            move = chess.Move(chess.parse_square(from_square), chess.parse_square(to_square))
            return move in self.board.legal_moves
        except ValueError:
            return False

    def make_move(self, from_square: str, to_square: str):
        try:
            from_sq = chess.parse_square(from_square)
            to_sq = chess.parse_square(to_square)
        except ValueError:
            return False

        # Normal move
        move = chess.Move(from_sq, to_sq)
        if move in self.board.legal_moves:
            self.board.push(move)
            return True

        # Promotion (Auto-promote to Queen)
        for promotion in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            move = chess.Move(from_sq, to_sq, promotion=promotion)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
        return False

    def undo(self):
        if not self.board.move_stack:
            return False
        self.board.pop()
        return True

    def reset(self):
        self.board.reset()

    def get_fen(self):
        return self.board.fen()

    def is_white_turn(self):
        return self.board.turn == chess.WHITE

    def is_black_turn(self):
        return self.board.turn == chess.BLACK

    def is_check(self):
        return self.board.is_check()

    def is_checkmate(self):
        return self.board.is_checkmate()

    def is_stalemate(self):
        return self.board.is_stalemate()

    def is_game_over(self):
        return self.board.is_game_over()