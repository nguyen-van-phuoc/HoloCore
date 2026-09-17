"""
Caro (Gomoku) Core Game Controller
===================================
Provides core game mechanics for a 50x50 Caro (Gomoku) board, including:
- Move validation and turn execution.
- Win-condition checking with double-ended block rules.
- Fast heuristic cell evaluation for AI decision-making.
- Candidate move selection via local neighborhood search.
- Conditional LLM-driven dialogue generation using Ollama (character persona: Holo).
"""

import json
import re
import ollama


class CaroController:
    """Manages board state, game logic, move scoring, and LLM dialogue integration."""

    def __init__(self, board_size: int = 50):
        """Initializes the game board and state variables.

        Args:
            board_size: The width and height of the square grid (default: 50).
        """
        self.board_size = board_size
        # Grid representation: 0 = Empty, 1 = Player (X), 2 = AI (O)
        self.board = [[0] * board_size for _ in range(board_size)]

        # Game status: 0 = Ongoing, 1 = Player Won, 2 = AI Won, 3 = Draw
        self.winner = 0

        # Coordinates tracking for UI highlighting
        self.last_move_player = None
        self.last_move_ai = None

    def play_move(self, r: int, c: int, player: int) -> bool:
        """Executes a move on the board if the targeted position is valid and empty.

        Args:
            r: Row index.
            c: Column index.
            player: Player identifier (1 for Human, 2 for AI).

        Returns:
            True if the move was successfully placed; False otherwise.
        """
        if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == 0:
            self.board[r][c] = player

            # Store coordinates of the last valid move for UI highlighting
            if player == 1:
                self.last_move_player = (r, c)
            else:
                self.last_move_ai = (r, c)

            # Evaluate whether this move triggers a win condition
            if self.check_win(r, c, player):
                self.winner = player
            return True

        return False

    def check_win(self, r: int, c: int, player: int) -> bool:
        """Checks if placing a stone at (r, c) forms a winning line.

        Rules:
        - Requires a continuous sequence of at least 5 matching stones.
        - Exactly 5 stones blocked on BOTH ends by opponent or board edges does NOT win.
        - Sequences > 5 stones (overlines) win regardless of blocked ends.

        Args:
            r: Row index of the latest move.
            c: Column index of the latest move.
            player: Player identifier (1 or 2).

        Returns:
            True if the move produces a winning sequence; False otherwise.
        """
        opponent = 3 - player  # Invert player: 1 -> 2, 2 -> 1
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonals

        for dr, dc in directions:
            count = 1  # Count includes the newly placed stone at (r, c)

            # 1. Traversal in the positive direction
            step = 1
            while True:
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                    count += 1
                    step += 1
                else:
                    # Check if the positive ray end is blocked by an opponent stone or board boundary
                    head_blocked = (not (0 <= nr < self.board_size and 0 <= nc < self.board_size)) or (
                                self.board[nr][nc] == opponent)
                    break

            # 2. Traversal in the negative direction
            step = 1
            while True:
                nr, nc = r - dr * step, c - dc * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                    count += 1
                    step += 1
                else:
                    # Check if the negative ray end is blocked
                    tail_blocked = (not (0 <= nr < self.board_size and 0 <= nc < self.board_size)) or (
                                self.board[nr][nc] == opponent)
                    break

            # 3. Win Condition Evaluation
            if count >= 5:
                # Overlines (>5) always win; exact 5 wins unless blocked on BOTH ends
                if count > 5 or not (head_blocked and tail_blocked):
                    return True

        return False

    def _eval_direction(self, r: int, c: int, dr: int, dc: int, player: int) -> tuple:
        """Helper method: Scans bi-directionally along a line to count continuous pieces and open ends.

        Args:
            r: Starting row index.
            c: Starting column index.
            dr: Row direction vector step (-1, 0, or 1).
            dc: Column direction vector step (-1, 0, or 1).
            player: Target player whose pieces are counted.

        Returns:
            A tuple containing:
            - count (int): Total consecutive pieces owned by `player` along the axis.
            - open_ends (int): Number of unblocked open ends (0, 1, or 2).
        """
        count = 0
        open_ends = 0

        # Scan along the positive ray direction
        step = 1
        while 0 <= r + dr * step < self.board_size and 0 <= c + dc * step < self.board_size:
            val = self.board[r + dr * step][c + dc * step]
            if val == player:
                count += 1
                step += 1
            elif val == 0:
                open_ends += 1  # Unblocked open end found
                break
            else:
                break  # Blocked by opponent piece

        # Scan along the negative ray direction
        step = 1
        while 0 <= r - dr * step < self.board_size and 0 <= c - dc * step < self.board_size:
            val = self.board[r - dr * step][c - dc * step]
            if val == player:
                count += 1
                step += 1
            elif val == 0:
                open_ends += 1  # Unblocked open end found
                break
            else:
                break  # Blocked by opponent piece

        return count, open_ends

    def evaluate_cell(self, r: int, c: int, player: int = 2) -> int:
        """Heuristic scoring function evaluating the strategic value of an empty cell.

        Combines offensive value (AI alignment) and defensive value (blocking Human alignment).

        Args:
            r: Target row coordinate.
            c: Target column coordinate.
            player: The player evaluating the move (default: 2 for AI).

        Returns:
            int: Calculated heuristic value of the cell.
        """
        opponent = 3 - player
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            # Evaluate offensive potential (AI completing patterns)
            off_count, off_open = self._eval_direction(r, c, dr, dc, player)
            # Evaluate defensive potential (blocking human player patterns)
            def_count, def_open = self._eval_direction(r, c, dr, dc, opponent)

            # --- OFFENSIVE SCORING WEIGHTS ---
            if off_count >= 4:
                score += 1000000  # Instant victory move (creates 5)
            elif off_count == 3 and off_open == 2:
                score += 50000  # Creates an open-4 (guaranteed win)
            elif off_count == 3 and off_open == 1:
                score += 3000  # Creates a single-blocked 4
            elif off_count == 2 and off_open == 2:
                score += 2000  # Creates an open-3
            elif off_count == 1 and off_open == 2:
                score += 100  # Creates an open-2

            # --- DEFENSIVE SCORING WEIGHTS (Threat neutralization) ---
            if def_count >= 4:
                score += 800000  # CRITICAL BLOCK: Neutralize human's instant win
            elif def_count == 3 and def_open == 2:
                score += 40000  # CRITICAL BLOCK: Neutralize human's open-3
            elif def_count == 3 and def_open == 1:
                score += 2500  # Block single-blocked 3
            elif def_count == 2 and def_open == 2:
                score += 1500  # Block open-2
            elif def_count == 1 and def_open == 2:
                score += 50  # Low priority block

        return score

    def get_top_candidates(self, player: int = 2, top_k: int = 5) -> tuple:
        """Prunes search space and extracts top strategic candidate moves.

        Optimization Technique:
        Instead of scanning all empty board positions (up to 2500 cells), scans only
        empty cells within a 5x5 sub-grid around existing stones.

        Args:
            player: Player identifier (default: 2 for AI).
            top_k: Number of highest-scoring positions to return.

        Returns:
            A tuple containing:
            - List of (row, col) candidate tuples.
            - Highest heuristic score found among candidates.
        """
        candidates = []
        visited = set()
        has_moves = False

        # Scan active area surrounding placed pieces
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] != 0:
                    has_moves = True
                    # Check 5x5 local neighborhood around active stone
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                                if self.board[nr][nc] == 0 and (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    score = self.evaluate_cell(nr, nc, player)
                                    candidates.append(((nr, nc), score))

        # Handle opening turn on an empty board (play at grid center)
        if not has_moves:
            return [(self.board_size // 2, self.board_size // 2)], 0

        # Sort candidates descending by heuristic score
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_score = candidates[0][1]

        # Early exit: If a critical move exists (win/must-block), return immediately
        if top_score >= 800000:
            return [candidates[0][0]], top_score

        return [pos for pos, score in candidates[:top_k]], top_score

    def get_llm_move(self, model_name="qwen3:4b-instruct") -> tuple:
        """Determines the AI move and conditionally generates dialogue via Ollama LLM.

        Dialogue Generation Strategy:
        To optimize performance, the LLM is only queried during critical events
        (score >= 800000: winning move or crucial defense block). Routine moves return
        an empty dialogue string immediately.

        Args:
            model_name: Name of the local Ollama model to invoke.

        Returns:
            A tuple containing (best_row, best_col, dialogue_string).
        """
        top_moves, top_score = self.get_top_candidates(player=2, top_k=5)

        best_r, best_c = top_moves[0]
        dialogue = ""

        # Query LLM only for high-stakes/critical moves
        if top_score >= 800000:
            if top_score >= 1000000:
                context = "You (Holo) just played a decisive move and guaranteed victory. Taunt your opponent."
            else:
                context = "The opponent (Human) was close to winning, but you (Holo) just blocked their dangerous threat in time. Act smug and triumphant."

            # Prompt layout defining persona rules, context injection, JSON format, and animation markup tags
            prompt = f'''You are an arrogant Caro (Gomoku) AI master.
                Current context: {context}
                Generate ONE short, natural dialogue line without any extra explanations.
                Return strictly a single valid JSON string in the format: {{"dialogue": "your dialogue text"}}

                The motion marker must match the emotion of the spoken line:
                - <motion:idle> - calm, natural, composed.
                - <motion:flick> - light teasing, playful.
                - <motion:flick_down> - shy, confused, hesitant, or embarrassed.
                - <motion:flick_up> - confident, arrogant, smug, or proud of the move.
                - <motion:tap> - cheerful, excited, or seeking attention.
                - <motion:tap_body> - emphasizing the move, strong reaction, or obvious teasing.
                - <motion:flick_body> - heavy teasing, mischievous, or highly expressive reaction.

                Rule: EVERY dialogue string MUST begin with a motion marker at the very start-no exceptions. Verify the first character before outputting.
                PLEASE REPLY IN VIETNAMESE IN ALL CASES.
            '''

            try:
                # Query local Ollama API asynchronously/synchronously
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.6}
                )
                content = response["message"]["content"]

                # Parse JSON payload from LLM response content
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    dialogue = str(data.get("dialogue", "")).strip()
            except Exception as e:
                print("Lỗi kết nối LLM/Ollama:", e)
                # Fallback static dialogue string if Ollama request fails
                dialogue = "Chấp nhận thất bại đi!" if top_score >= 1000000 else "Đừng hòng qua mặt ta!"

        return best_r, best_c, dialogue