import re
import time
import chess
import random
import requests
import chess.engine
from typing import List
from dataclasses import dataclass


# DATA MODELS
@dataclass(frozen=True)
class MoveEvaluation:
    move_uci: str
    score_cp: int


@dataclass(frozen=True)
class TurnResult:
    tactic: str
    move_uci: str
    talk: str
    eval: list


# OLLAMA CLIENT
class OllamaClient:
    """Client wrapper for interacting with the Ollama local API."""

    def __init__(self, base_url: str, model_name: str, max_retries: int = 2, timeout: float = 60.0):
        self.base_url = base_url
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, json_mode: bool = False):
        payload = {
            "model": self.model_name,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }

        if json_mode:
            payload["format"] = "json"

        last_error = None
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self._session.post(self.base_url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                last_error = e
                # logger.warning(f"Ollama error (attempt {attempt}): {e}")
                print(f"Ollama error (attempt {attempt}): {e}")
                if attempt <= self.max_retries:
                    time.sleep(0.5 * attempt)

        # logger.error(f"Ollama failed: {last_error}")
        print(f"Ollama failed: {last_error}")
        return ""

    def close(self):
        self._session.close()


# STOCKFISH ANALYZER (The Brain)
class StockfishAnalyzer:
    """Handles Stockfish UCI engine process and evaluates legal board moves."""

    def __init__(self, engine_path: str, time_limit: float = 0.2, threads: int = 2, hash_mb: int = 128):
        self.engine_path = engine_path
        self.time_limit = time_limit
        self.threads = threads
        self.hash_mb = hash_mb
        self.engine = None

    def start(self):
        if self.engine is not None:
            return
        print("Starting Stockfish engine...")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        try:
            self.engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        except Exception as e:
            print(f"Could not configure Stockfish: {e}")

    def stop(self):
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None
            print("Stockfish engine stopped.")

    def get_top_moves(self, board: chess.Board, limit: int = 3) -> List[MoveEvaluation]:
        """
        Uses Stockfish MultiPV mode to fetch the top N moves for the given board state.
        Prevents the AI from selecting weak or illegal moves.
        """
        if self.engine is None:
            raise RuntimeError("Stockfish engine is not running.")

        evaluations = []
        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(time=self.time_limit),
                multipv=limit
            )

            for d in info:
                if "pv" in d and len(d["pv"]) > 0:
                    move_uci = d["pv"][0].uci()
                    # Convert score relative to the active player's point of view (in Centipawns)
                    score = d["score"].pov(board.turn).score(mate_score=10000) or 0
                    evaluations.append(MoveEvaluation(move_uci=move_uci, score_cp=score))

        except Exception as e:
            # logger.warning(f"Error fetching top moves from Stockfish: {e}")
            print(f"Error fetching top moves from Stockfish: {e}")
        return evaluations


# CHARACTER AGENT (The Persona)
class CharacterAgent:
    """Connects LLM output with move selection to roleplay a chess-playing persona."""

    def __init__(self, llm_client: OllamaClient, system_prompt: str):
        self.llm_client = llm_client
        self.system_prompt = system_prompt

    def decide_final_move(self, board: chess.Board, evaluations: List[MoveEvaluation]):
        # Fallback if no Stockfish evaluations are returned
        if not evaluations:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise ValueError("No legal moves available.")
            return TurnResult(
                tactic="I suppose I must resort to this move.",
                move_uci=random.choice(legal_moves).uci(),
                talk="",
                eval=[]
            )

        # Build evaluations string for the LLM prompt
        eval_text = "\n".join(f"- {ev.move_uci}: {ev.score_cp} cp (higher is better)" for ev in evaluations)

        prompt = f"""
        [MOVE DECISION & DIALOGUE]
        Current FEN: {board.fen()}
        Stockfish recommended TOP MOVES:
        {eval_text}

        Stay in character, analyze the position quickly, and pick 1 move from the list above.
        Return strictly in the following XML format (do not add extra formatting or explanations):
        <tactic>In-character monologue detailing your tactical thought process or mocking the opponent. Do not include extra tags.</tactic>
        <final_move>chosen_UCI_move</final_move>
        <talk>If the move is notable (blunder punisher, capture, check, checkmate, tactic...), write a short voice line to speak out loud. Otherwise, leave empty.</talk>
        """

        response_text = self.llm_client.generate(self.system_prompt, prompt, temperature=0.6)

        # Extract tactics/monologue
        talk_match = re.search(r"<tactic>(.*?)</tactic>", response_text, re.DOTALL)
        tactic = talk_match.group(1).strip() if talk_match else "My turn."

        # Extract voice lines
        talk_llm = re.search(r"<talk>(.*?)</talk>", response_text, re.DOTALL)
        talk = talk_llm.group(1).strip() if talk_llm else ""

        # Extract chosen move
        move_match = re.search(r"<final_move>(.*?)</final_move>", response_text, re.DOTALL)
        chosen_move = move_match.group(1).strip() if move_match else evaluations[0].move_uci

        # Strict validation: If LLM chooses a move outside top Stockfish suggestions, force Top 1
        if not any(ev.move_uci == chosen_move for ev in evaluations):
            chosen_move = evaluations[0].move_uci

        return TurnResult(tactic=tactic, move_uci=chosen_move, talk=talk, eval=evaluations)


# GAME CONTROLLER
class ChessGameController:
    """Manages board state, Stockfish synchronization, and turn progression."""

    def __init__(self, board: chess.Board, agent: CharacterAgent, stockfish_path: str, engine_time_limit: float = 0.2,
                 stockfish_threads: int = 2, stockfish_hash: int = 128):
        self.board = board
        self.agent = agent
        self.analyzer = StockfishAnalyzer(
            stockfish_path,
            time_limit=engine_time_limit,
            threads=stockfish_threads,
            hash_mb=stockfish_hash
        )
        self._engine_started = False

    def start(self):
        if not self._engine_started:
            self.analyzer.start()
            self._engine_started = True

    def stop(self):
        if self._engine_started:
            self.analyzer.stop()
            self._engine_started = False

    def play_agent_turn(self):
        if not self._engine_started:
            raise RuntimeError("Stockfish engine is not started.")

        if self.board.is_game_over():
            raise ValueError("The game is already over.")

        print("========== AI TURN ==========")

        # 1. Fetch top moves from Stockfish
        evaluations = self.analyzer.get_top_moves(self.board, limit=3)
        print(f"Phase 1: Stockfish Top Moves: {evaluations}")

        # 2. Ask LLM to pick a move and generate persona responses
        result = self.agent.decide_final_move(self.board, evaluations)
        print(f"Phase 2: LLM Selected: {result.move_uci}")

        # 3. Validate and execute move on board
        try:
            final_move = chess.Move.from_uci(result.move_uci)
            if final_move in self.board.legal_moves:
                self.board.push(final_move)
                # logger.info(f"AI executed: {result.move_uci}")
                print(f"Phase 3: AI played: {result.move_uci}")
                print("Phase 4: Move executed successfully.")
                return result
        except Exception:
            pass

        # Fallback mechanism if LLM output fails execution
        print("LLM output invalid move. Executing fallback.")
        backup_uci = evaluations[0].move_uci if evaluations else random.choice(list(self.board.legal_moves)).uci()

        self.board.push(chess.Move.from_uci(backup_uci))
        print(f"Fallback executed: {backup_uci}")
        return TurnResult(tactic=result.tactic, move_uci=backup_uci, talk=result.talk, eval=evaluations)