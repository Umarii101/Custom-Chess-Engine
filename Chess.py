"""
Complete GUI Chess Game with tkinter
Features: Full chess rules, timers, move history, undo, highlights, and more!
Python 3.8+ compatible - Standard library only
"""

import tkinter as tk
from tkinter import ttk, messagebox
import copy
import time
import random
from typing import List, Tuple, Optional, Dict

# Constants
BOARD_SIZE = 8
SQUARE_SIZE = 64
LIGHT_COLOR = "#F0D9B5"
DARK_COLOR = "#B58863"
HIGHLIGHT_COLOR = "#FFFF00"
VALID_MOVE_COLOR = "#90EE90"
CHECK_COLOR = "#FF6B6B"

# Unicode chess pieces (safe for tkinter)
PIECE_SYMBOLS = {
    'white': {
        'king': '♔', 'queen': '♕', 'rook': '♖',
        'bishop': '♗', 'knight': '♘', 'pawn': '♙'
    },
    'black': {
        'king': '♚', 'queen': '♛', 'rook': '♜',
        'bishop': '♝', 'knight': '♞', 'pawn': '♟'
    }
}
def unpack_move(move):
    return (*move[0], *move[1])

class Piece:
    """Represents a chess piece with color, type, and position"""
    
    def __init__(self, color: str, piece_type: str, row: int, col: int):
        self.color = color  # 'white' or 'black'
        self.piece_type = piece_type  # 'king', 'queen', etc.
        self.row = row
        self.col = col
        self.has_moved = False  # For castling and pawn double moves
        
    def get_symbol(self) -> str:
        """Returns Unicode symbol for the piece"""
        return PIECE_SYMBOLS[self.color][self.piece_type]
    
    def copy(self):
        """Create a copy of this piece"""
        new_piece = Piece(self.color, self.piece_type, self.row, self.col)
        new_piece.has_moved = self.has_moved
        return new_piece

class ChessBoard:
    """Manages the chess board state and piece positions"""
    
    def __init__(self):
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'white' 
        self.move_history = []
        self.captured_pieces = {'white': [], 'black': []}
        self.en_passant_target = None  # (row, col) for en passant capture
        self.king_positions = {'white': (7, 4), 'black': (0, 4)}
        self.setup_initial_position()
    
    def setup_initial_position(self):
        """Set up the standard chess starting position"""
        # Place pawns
        for col in range(BOARD_SIZE):
            self.board[1][col] = Piece('black', 'pawn', 1, col)
            self.board[6][col] = Piece('white', 'pawn', 6, col)
        
        # Place other pieces
        piece_order = ['rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook']
        
        for col in range(BOARD_SIZE):
            self.board[0][col] = Piece('black', piece_order[col], 0, col)
            self.board[7][col] = Piece('white', piece_order[col], 7, col)
    
    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        """Get piece at given position"""
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row][col]
        return None
    
    def set_piece(self, row: int, col: int, piece: Optional[Piece]):
        """Set piece at given position"""
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            self.board[row][col] = piece
            if piece:
                piece.row, piece.col = row, col
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within board bounds"""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
    
    def get_valid_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        """Get all valid moves for a piece"""
        if not piece:
            return []
        
        moves = []
        row, col = piece.row, piece.col
        
        if piece.piece_type == 'pawn':
            moves = self._get_pawn_moves(piece)
        elif piece.piece_type == 'rook':
            moves = self._get_rook_moves(piece)
        elif piece.piece_type == 'knight':
            moves = self._get_knight_moves(piece)
        elif piece.piece_type == 'bishop':
            moves = self._get_bishop_moves(piece)
        elif piece.piece_type == 'queen':
            moves = self._get_queen_moves(piece)
        elif piece.piece_type == 'king':
            moves = self._get_king_moves(piece)
        
        # Filter out moves that would put own king in check
        valid_moves = []
        for move_row, move_col in moves:
            if self._is_legal_move(piece, move_row, move_col):
                valid_moves.append((move_row, move_col))
        
        return valid_moves
    
    def _get_pawn_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        """Get valid pawn moves"""
        moves = []
        row, col = piece.row, piece.col
        direction = -1 if piece.color == 'white' else 1
        start_row = 6 if piece.color == 'white' else 1
        
        # Forward move
        new_row = row + direction
        if self.is_valid_position(new_row, col) and not self.get_piece(new_row, col):
            moves.append((new_row, col))
            
            # Double move from starting position
            if row == start_row:
                new_row = row + 2 * direction
                if self.is_valid_position(new_row, col) and not self.get_piece(new_row, col):
                    moves.append((new_row, col))
        
        # Diagonal captures
        for dc in [-1, 1]:
            new_row, new_col = row + direction, col + dc
            if self.is_valid_position(new_row, new_col):
                target_piece = self.get_piece(new_row, new_col)
                if target_piece and target_piece.color != piece.color:
                    moves.append((new_row, new_col))
                # En passant
                elif self.en_passant_target == (new_row, new_col):
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_rook_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        """Get valid rook moves"""
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            for i in range(1, BOARD_SIZE):
                new_row = piece.row + i * dr
                new_col = piece.col + i * dc
                
                if not self.is_valid_position(new_row, new_col):
                    break
                
                target_piece = self.get_piece(new_row, new_col)
                if target_piece:
                    if target_piece.color != piece.color:
                        moves.append((new_row, new_col))
                    break
                else:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_knight_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        """Get valid knight moves"""
        moves = []
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for dr, dc in knight_moves:
            new_row = piece.row + dr
            new_col = piece.col + dc
            
            if self.is_valid_position(new_row, new_col):
                target_piece = self.get_piece(new_row, new_col)
                if not target_piece or target_piece.color != piece.color:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_bishop_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        """Get valid bishop moves"""
        moves = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in directions:
            for i in range(1, BOARD_SIZE):
                new_row = piece.row + i * dr
                new_col = piece.col + i * dc
                
                if not self.is_valid_position(new_row, new_col):
                    break
                
                target_piece = self.get_piece(new_row, new_col)
                if target_piece:
                    if target_piece.color != piece.color:
                        moves.append((new_row, new_col))
                    break
                else:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_queen_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        """Get valid queen moves (combination of rook and bishop)"""
        return self._get_rook_moves(piece) + self._get_bishop_moves(piece)
    
    def _get_king_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        """Get valid king moves including castling"""
        moves = []
        
        # Regular king moves
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                new_row = piece.row + dr
                new_col = piece.col + dc
                
                if self.is_valid_position(new_row, new_col):
                    target_piece = self.get_piece(new_row, new_col)
                    if not target_piece or target_piece.color != piece.color:
                        moves.append((new_row, new_col))
        
        # Castling
        if not piece.has_moved and not self.is_in_check(piece.color):
            # Kingside castling
            rook = self.get_piece(piece.row, 7)
            if (rook and rook.piece_type == 'rook' and not rook.has_moved and
                not self.get_piece(piece.row, 5) and not self.get_piece(piece.row, 6)):
                if not self._would_be_in_check_after_move(piece, piece.row, 5):
                    moves.append((piece.row, 6))
            
            # Queenside castling
            rook = self.get_piece(piece.row, 0)
            if (rook and rook.piece_type == 'rook' and not rook.has_moved and
                not self.get_piece(piece.row, 1) and not self.get_piece(piece.row, 2) and
                not self.get_piece(piece.row, 3)):
                if not self._would_be_in_check_after_move(piece, piece.row, 3):
                    moves.append((piece.row, 2))
        
        return moves
    
    def _is_legal_move(self, piece: Piece, to_row: int, to_col: int) -> bool:
        """Check if move is legal (doesn't put own king in check)"""
        # Make temporary move
        original_piece = self.get_piece(to_row, to_col)
        original_row, original_col = piece.row, piece.col
        
        self.set_piece(to_row, to_col, piece)
        self.set_piece(original_row, original_col, None)
        
        # Update king position if king moved
        if piece.piece_type == 'king':
            self.king_positions[piece.color] = (to_row, to_col)
        
        # Check if own king is in check
        is_legal = not self.is_in_check(piece.color)
        
        # Restore original position
        self.set_piece(original_row, original_col, piece)
        self.set_piece(to_row, to_col, original_piece)
        
        if piece.piece_type == 'king':
            self.king_positions[piece.color] = (original_row, original_col)
        
        return is_legal
    
    def _would_be_in_check_after_move(self, piece: Piece, to_row: int, to_col: int) -> bool:
        """Check if king would be in check after a move"""
        return not self._is_legal_move(piece, to_row, to_col)
    
    def is_in_check(self, color: str) -> bool:
        """Check if the king of given color is in check"""
        king_row, king_col = self.king_positions[color]
        opponent_color = 'black' if color == 'white' else 'white'
        
        # Check if any opponent piece can attack the king
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.get_piece(row, col)
                if piece and piece.color == opponent_color:
                    if self._can_attack(piece, king_row, king_col):
                        return True
        
        return False
    
    def _can_attack(self, piece: Piece, target_row: int, target_col: int) -> bool:
        """Check if piece can attack target position (without legal move validation)"""
        if piece.piece_type == 'pawn':
            direction = -1 if piece.color == 'white' else 1
            if (piece.row + direction == target_row and 
                abs(piece.col - target_col) == 1):
                return True
        elif piece.piece_type == 'rook':
            if (piece.row == target_row or piece.col == target_col):
                return self._is_clear_path(piece.row, piece.col, target_row, target_col)
        elif piece.piece_type == 'knight':
            dr, dc = abs(piece.row - target_row), abs(piece.col - target_col)
            if (dr == 2 and dc == 1) or (dr == 1 and dc == 2):
                return True
        elif piece.piece_type == 'bishop':
            if abs(piece.row - target_row) == abs(piece.col - target_col):
                return self._is_clear_path(piece.row, piece.col, target_row, target_col)
        elif piece.piece_type == 'queen':
            if (piece.row == target_row or piece.col == target_col or
                abs(piece.row - target_row) == abs(piece.col - target_col)):
                return self._is_clear_path(piece.row, piece.col, target_row, target_col)
        elif piece.piece_type == 'king':
            if (abs(piece.row - target_row) <= 1 and abs(piece.col - target_col) <= 1):
                return True
        
        return False
    
    def _is_clear_path(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """Check if path between two positions is clear"""
        dr = 0 if from_row == to_row else (1 if to_row > from_row else -1)
        dc = 0 if from_col == to_col else (1 if to_col > from_col else -1)
        
        current_row, current_col = from_row + dr, from_col + dc
        
        while current_row != to_row or current_col != to_col:
            if self.get_piece(current_row, current_col):
                return False
            current_row += dr
            current_col += dc
        
        return True
    
    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """Make a move if it's valid"""
        piece = self.get_piece(from_row, from_col)
        if not piece or piece.color != self.current_player:
            return False
        
        valid_moves = self.get_valid_moves(piece)
        if (to_row, to_col) not in valid_moves:
            return False
        
        # Store move for history and undo
        move_data = {
            'from': (from_row, from_col),
            'to': (to_row, to_col),
            'piece': piece.copy(),
            'captured': None,
            'special': None,
            'en_passant_target': self.en_passant_target
        }
        
        captured_piece = self.get_piece(to_row, to_col)
        if captured_piece:
            move_data['captured'] = captured_piece.copy()
            self.captured_pieces[captured_piece.color].append(captured_piece)
        
        # Handle special moves
        self._handle_special_moves(piece, from_row, from_col, to_row, to_col, move_data)
        
        # Make the move
        self.set_piece(to_row, to_col, piece)
        self.set_piece(from_row, from_col, None)
        piece.has_moved = True
        
        # Update king position
        if piece.piece_type == 'king':
            self.king_positions[piece.color] = (to_row, to_col)
        
        # Add to move history
        self.move_history.append(move_data)
        
        # Switch players
        print(f"{self.current_player} made a move")
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        print(f"Now its turn of {self.current_player}")
        return True
    
    def _handle_special_moves(self, piece: Piece, from_row: int, from_col: int, 
                             to_row: int, to_col: int, move_data: dict):
        """Handle castling, en passant, and pawn promotion"""
        
        # Castling
        if piece.piece_type == 'king' and abs(to_col - from_col) == 2:
            move_data['special'] = 'castle'
            if to_col == 6:  # Kingside
                rook = self.get_piece(from_row, 7)
                self.set_piece(from_row, 5, rook)
                self.set_piece(from_row, 7, None)
                rook.has_moved = True
            else:  # Queenside
                rook = self.get_piece(from_row, 0)
                self.set_piece(from_row, 3, rook)
                self.set_piece(from_row, 0, None)
                rook.has_moved = True
        
        # En passant
        elif (piece.piece_type == 'pawn' and self.en_passant_target == (to_row, to_col)):
            move_data['special'] = 'en_passant'
            # Remove the captured pawn
            captured_row = from_row
            captured_pawn = self.get_piece(captured_row, to_col)
            move_data['captured'] = captured_pawn.copy()
            self.captured_pieces[captured_pawn.color].append(captured_pawn)
            self.set_piece(captured_row, to_col, None)
        
        # Set en passant target for next move
        self.en_passant_target = None
        if (piece.piece_type == 'pawn' and abs(to_row - from_row) == 2):
            self.en_passant_target = ((from_row + to_row) // 2, to_col)
    
    def undo_last_move(self) -> bool:
        """Undo the last move"""
        if not self.move_history:
            return False
        
        move = self.move_history.pop()
        from_row, from_col = move['from']
        to_row, to_col = move['to']
        piece = move['piece']
        
        # Restore piece to original position
        self.set_piece(from_row, from_col, piece)
        
        # Restore captured piece or clear destination
        if move['captured']:
            self.set_piece(to_row, to_col, move['captured'])
            # Remove from captured pieces list
            self.captured_pieces[move['captured'].color].remove(move['captured'])
        else:
            self.set_piece(to_row, to_col, None)
        
        # Handle special move undos
        if move['special'] == 'castle':
            if to_col == 6:  # Kingside
                rook = self.get_piece(from_row, 5)
                self.set_piece(from_row, 7, rook)
                self.set_piece(from_row, 5, None)
                rook.has_moved = False
            else:  # Queenside
                rook = self.get_piece(from_row, 3)
                self.set_piece(from_row, 0, rook)
                self.set_piece(from_row, 3, None)
                rook.has_moved = False
        elif move['special'] == 'en_passant':
            # Restore captured pawn
            captured_pawn = move['captured']
            self.set_piece(from_row, to_col, captured_pawn)
            self.captured_pieces[captured_pawn.color].remove(captured_pawn)
        
        # Restore en passant target
        self.en_passant_target = move['en_passant_target']
        
        # Update king position
        if piece.piece_type == 'king':
            self.king_positions[piece.color] = (from_row, from_col)
        
        # Switch back to previous player
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        
        return True
    
    def is_checkmate(self, color: str) -> bool:
        """Check if the given color is in checkmate"""
        if not self.is_in_check(color):
            return False
        
        # Check if any piece can make a legal move
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.get_piece(row, col)
                if piece and piece.color == color:
                    if self.get_valid_moves(piece):
                        return False
        
        return True
    
    def is_stalemate(self, color: str) -> bool:
        """Check if the given color is in stalemate"""
        if self.is_in_check(color):
            return False
        
        # Check if any piece can make a legal move
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.get_piece(row, col)
                if piece and piece.color == color:
                    if self.get_valid_moves(piece):
                        return False
        
        return True
    
    def get_algebraic_notation(self, move_data: dict) -> str:
        """Convert move to algebraic notation"""
        piece = move_data['piece']
        from_row, from_col = move_data['from']
        to_row, to_col = move_data['to']
        
        # Convert to chess notation (a1-h8)
        from_square = chr(ord('a') + from_col) + str(8 - from_row)
        to_square = chr(ord('a') + to_col) + str(8 - to_row)
        
        # Piece prefix
        piece_letter = piece.piece_type[0].upper() if piece.piece_type != 'pawn' else ''
        
        # Capture notation
        capture = 'x' if move_data['captured'] else ''
        if piece.piece_type == 'pawn' and capture:
            piece_letter = chr(ord('a') + from_col)
        
        # Special moves
        if move_data['special'] == 'castle':
            return 'O-O' if to_col == 6 else 'O-O-O'
        
        return f"{piece_letter}{capture}{to_square}"

    def copy(self):
        """Create a deep copy of the chess board"""
        new_board = ChessBoard()
        
        # Deep copy the board cells and pieces
        new_board.board = [[copy.deepcopy(piece) for piece in row] for row in self.board]
        
        # Copy current player
        new_board.current_player = self.current_player
        
        # Optional: copy other game state attributes
        if hasattr(self, 'move_history'):
            new_board.move_history = copy.deepcopy(self.move_history)
        if hasattr(self, 'en_passant_target'):
            new_board.en_passant_target = copy.deepcopy(self.en_passant_target)
        if hasattr(self, 'castling_rights'):
            new_board.castling_rights = copy.deepcopy(self.castling_rights)
        
        return new_board

class AdvancedChessBot:
    """
    Advanced chess bot with minimax search, tactical awareness, and sophisticated evaluation.
    Features:
    - Minimax with alpha-beta pruning (depth 3-4)
    - Tactical pattern recognition (pins, forks, skewers, discovered attacks)
    - Dynamic opening book with multiple lines
    - Advanced positional evaluation
    - Endgame awareness
    - Anti-repetition logic
    """

    # Enhanced piece values with positional context
    PIECE_VALUES = {
        'pawn': 100,
        'knight': 320,
        'bishop': 330,
        'rook': 500,
        'queen': 900,
        'king': 20000
    }

    # Positional piece-square tables (simplified)
    PAWN_TABLE = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5, 5, 10, 25, 25, 10, 5, 5],
        [0, 0, 0, 20, 20, 0, 0, 0],
        [5, -5, -10, 0, 0, -10, -5, 5],
        [5, 10, 10, -20, -20, 10, 10, 5],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]

    KNIGHT_TABLE = [
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20, 0, 0, 0, 0, -20, -40],
        [-30, 0, 10, 15, 15, 10, 0, -30],
        [-30, 5, 15, 20, 20, 15, 5, -30],
        [-30, 0, 15, 20, 20, 15, 0, -30],
        [-30, 5, 10, 15, 15, 10, 5, -30],
        [-40, -20, 0, 5, 5, 0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50]
    ]

    BISHOP_TABLE = [
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10, 0, 0, 0, 0, 0, 0, -10],
        [-10, 0, 5, 10, 10, 5, 0, -10],
        [-10, 5, 5, 10, 10, 5, 5, -10],
        [-10, 0, 10, 10, 10, 10, 0, -10],
        [-10, 10, 10, 10, 10, 10, 10, -10],
        [-10, 5, 0, 0, 0, 0, 5, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20]
    ]

    CENTER_SQUARES = [(3, 3), (3, 4), (4, 3), (4, 4)]
    EXTENDED_CENTER = [(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5), (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5, 5)]

    def __init__(self, board: ChessBoard, color: str = 'black', max_depth: int = 3):
        self.board = board
        self.color = color
        self.opponent_color = 'white' if color == 'black' else 'black'
        self.max_depth = max_depth
        self.thinking = False
        self.move_history = []
        self.position_history = []  # For repetition detection
        self.game_phase = 'opening'  # opening, middlegame, endgame
        
        # Opening book with multiple lines
        self.opening_book = self.initialize_opening_book()
        self.current_opening_line = None
        self.opening_move_count = 0
        
        # Tactical and strategic tracking
        self.last_evaluation = 0
        self.material_balance = 0
        
    def initialize_opening_book(self) -> Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Initialize opening book with multiple opening lines"""
        if self.color == 'black':
            return {
                'sicilian_dragon': [
                    ((1, 2), (3, 2)),  # c5
                    ((0, 6), (2, 5)),  # Nf6
                    ((1, 6), (2, 6)),  # g6
                    ((0, 5), (1, 6)),  # Bg7
                    ((0, 1), (2, 2)),  # Nc6
                ],
                'french_defense': [
                    ((1, 4), (3, 4)),  # e6
                    ((1, 3), (3, 3)),  # d5
                    ((0, 1), (2, 2)),  # Nc6
                    ((0, 6), (2, 5)),  # Nf6
                ],
                'caro_kann': [
                    ((1, 2), (3, 2)),  # c6
                    ((1, 3), (3, 3)),  # d5
                    ((0, 1), (2, 2)),  # Nc6
                    ((0, 2), (1, 3)),  # Bf5
                ],
                'kings_indian': [
                    ((1, 6), (2, 6)),  # g6
                    ((0, 5), (1, 6)),  # Bg7
                    ((0, 6), (2, 5)),  # Nf6
                    ((7, 4), (7, 6)),  # O-O (simplified)
                ]
            }
        else:  # white openings
            return {
                'italian_game': [
                    ((6, 4), (4, 4)),  # e4
                    ((7, 6), (5, 5)),  # Nf3
                    ((7, 5), (4, 2)),  # Bc4
                    ((7, 1), (5, 2)),  # Nc3
                ],
                'queens_gambit': [
                    ((6, 3), (4, 3)),  # d4
                    ((6, 2), (4, 2)),  # c4
                    ((7, 6), (5, 5)),  # Nf3
                    ((7, 1), (5, 2)),  # Nc3
                ],
                'ruy_lopez': [
                    ((6, 4), (4, 4)),  # e4
                    ((7, 6), (5, 5)),  # Nf3
                    ((7, 5), (4, 2)),  # Bb5
                    ((7, 1), (5, 2)),  # Nc3
                ]
            }

    def select_opening_line(self) -> Optional[str]:
        """Dynamically select opening line based on opponent's moves"""
        if not self.opening_book or len(self.move_history) > 8:
            return None
            
        # Simple heuristic: choose based on center control preference
        opponent_moves = [move for i, move in enumerate(self.move_history) if i % 2 == 1]
        
        # Prefer solid openings against aggressive play
        if any(self.is_aggressive_move(move) for move in opponent_moves):
            return 'french_defense' if self.color == 'black' else 'queens_gambit'
        else:
            return 'sicilian_dragon' if self.color == 'black' else 'italian_game'

    def is_aggressive_move(self, move) -> bool:
        """Check if a move appears aggressive (early attacks, sacrifices)"""
        (from_pos, to_pos) = move
        piece = self.board.get_piece(*from_pos)
        target = self.board.get_piece(*to_pos)
        
        # Early queen moves, early attacks on f7/f2
        if piece and piece.piece_type == 'queen' and len(self.move_history) < 6:
            return True
        if to_pos in [(1, 5), (6, 5)]:  # f7, f2 squares
            return True
        return False

    def get_best_move(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Main entry point for move selection"""
        print("[AdvancedBot] get_best_move called")
        if self.board.current_player != self.color:
            print("[AdvancedBot] Not my turn")
            return None

        self.thinking = True
        start_time = time.time()
        
        # Update game phase
        self.update_game_phase()
        
        # Try opening book first
        opening_move = self.try_opening_book()
        if opening_move:
            self.thinking = False
            return opening_move
        
        # Use minimax search
        best_move = self.minimax_search()
        
        elapsed = time.time() - start_time
        print(f"[AdvancedBot] Thinking time: {elapsed:.2f}s, Evaluation: {self.last_evaluation}")
        
        self.thinking = False
        return best_move

    def try_opening_book(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Try to use opening book move"""
        if self.opening_move_count >= 8:
            print("[OpeningBook] Opening phase completed")
            return None
            
        if not self.current_opening_line:
            self.current_opening_line = self.select_opening_line()
            print(f"[OpeningBook] Selected opening: {self.current_opening_line}")
            
        if not self.current_opening_line or self.current_opening_line not in self.opening_book:
            print("[OpeningBook] No valid opening line found")
            return None
            
        line = self.opening_book[self.current_opening_line]
        if self.opening_move_count >= len(line):
            print("[OpeningBook] All moves in line exhausted")
            return None
            
        move = line[self.opening_move_count]
        piece = self.board.get_piece(*move[0])
        print(f"[OpeningBook] Trying move {self.opening_move_count + 1}: {move}")
        
        if piece and piece.color == self.color:
            valid_moves = self.board.get_valid_moves(piece)
            print(f"[OpeningBook] Valid moves from {move[0]}: {valid_moves}")
            if move[1] in valid_moves and self.is_move_safe(*move[0], *move[1]):
                print(f"[AdvancedBot] Book move played: {move}")
                self.opening_move_count += 1
                return move
            else:
                print("[OpeningBook] Book move invalid or unsafe")
        
        print("[OpeningBook] Abandoning book")
        self.opening_move_count = 10
        return None


    def minimax_search(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Minimax search with alpha-beta pruning"""
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        moves = self.generate_ordered_moves(self.color)
        
        for move in moves:
            if not self.is_move_safe(*move[0], *move[1]):
                continue
                
            # Make move
            board_copy = self.make_temp_move(move)
            
            # Evaluate
            value = self.minimax(board_copy, self.max_depth - 1, alpha, beta, False)
            
            if value > best_value:
                best_value = value
                best_move = move
                
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Alpha-beta pruning
                
        self.last_evaluation = best_value
        return best_move

    def minimax(self, board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """Minimax algorithm with alpha-beta pruning"""
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_position(board)
            
        current_color = self.color if maximizing else self.opponent_color
        moves = self.generate_ordered_moves(current_color, board)
        
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                if not self.is_move_safe_on_board(*move[0], *move[1], board):
                    continue
                    
                new_board = self.make_temp_move_on_board(move, board)
                eval_score = self.minimax(new_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('+inf')
            for move in moves:
                if not self.is_move_safe_on_board(*move[0], *move[1], board):
                    continue
                    
                new_board = self.make_temp_move_on_board(move, board)
                eval_score = self.minimax(new_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def generate_ordered_moves(self, color: str, board=None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Generate moves ordered by likely strength (captures first, then tactical moves)"""
        if board is None:
            board = self.board
            
        captures = []
        tactical_moves = []
        quiet_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece and piece.color == color:
                    valid_moves = board.get_valid_moves(piece)
                    
                    for move_row, move_col in valid_moves:
                        move = ((row, col), (move_row, move_col))
                        target = board.get_piece(move_row, move_col)
                        
                        # Prioritize captures
                        if target:
                            capture_value = self.PIECE_VALUES.get(target.piece_type, 0)
                            attacker_value = self.PIECE_VALUES.get(piece.piece_type, 0)
                            # Good captures first (higher value target, lower value attacker)
                            captures.append((capture_value - attacker_value, move))
                        
                        # Tactical moves (checks, threats)
                        elif self.creates_threat(move, board):
                            tactical_moves.append(move)
                        
                        # Quiet moves
                        else:
                            quiet_moves.append(move)
        
        # Sort captures by value difference
        captures.sort(reverse=True, key=lambda x: x[0])
        ordered_moves = [move for _, move in captures] + tactical_moves + quiet_moves
        
        return ordered_moves

    def creates_threat(self, move: Tuple[Tuple[int, int], Tuple[int, int]], board) -> bool:
        """Check if move creates a threat (simplified)"""
        (from_pos, to_pos) = move
        temp_board = self.make_temp_move_on_board(move, board)
        
        # Check if move gives check
        opponent_color = 'white' if self.color == 'black' else 'black'
        if temp_board.is_in_check(opponent_color):
            return True
            
        # Check if move attacks undefended pieces
        piece = board.get_piece(*from_pos)
        if piece:
            attacking_moves = temp_board.get_valid_moves(temp_board.get_piece(*to_pos))
            for attack_row, attack_col in attacking_moves:
                target = temp_board.get_piece(attack_row, attack_col)
                if target and target.color != self.color:
                    if not self.is_piece_defended(attack_row, attack_col, temp_board, target.color):
                        return True
                        
        return False

    def evaluate_position(self, board) -> float:
        """Comprehensive position evaluation"""
        score = 0
        
        # Material balance
        material_score = self.evaluate_material(board)
        score += material_score
        
        # Positional factors
        positional_score = self.evaluate_positional_factors(board)
        score += positional_score
        
        # King safety
        king_safety_score = self.evaluate_king_safety(board)
        score += king_safety_score
        
        # Tactical patterns
        tactical_score = self.evaluate_tactical_patterns(board)
        score += tactical_score
        
        # Game phase specific evaluation
        if self.game_phase == 'endgame':
            endgame_score = self.evaluate_endgame_factors(board)
            score += endgame_score
            
        return score

    def evaluate_material(self, board) -> float:
        """Evaluate material balance with piece-square tables"""
        score = 0
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece:
                    piece_value = self.PIECE_VALUES.get(piece.piece_type, 0)
                    positional_bonus = self.get_positional_bonus(piece, row, col)
                    
                    total_value = piece_value + positional_bonus
                    
                    if piece.color == self.color:
                        score += total_value
                    else:
                        score -= total_value
                        
        return score

    def get_positional_bonus(self, piece, row: int, col: int) -> float:
        """Get positional bonus from piece-square tables"""
        if piece.piece_type == 'pawn':
            if piece.color == 'white':
                return self.PAWN_TABLE[row][col]
            else:
                return self.PAWN_TABLE[7-row][col]
        elif piece.piece_type == 'knight':
            return self.KNIGHT_TABLE[row][col]
        elif piece.piece_type == 'bishop':
            return self.BISHOP_TABLE[row][col]
        
        return 0

    def evaluate_positional_factors(self, board) -> float:
        """Evaluate positional factors"""
        score = 0
        
        # Center control
        center_control = self.evaluate_center_control(board)
        score += center_control * 10
        
        # Piece development
        development_score = self.evaluate_development(board)
        score += development_score
        
        # Pawn structure
        pawn_structure_score = self.evaluate_pawn_structure(board)
        score += pawn_structure_score
        
        return score

    def evaluate_center_control(self, board) -> float:
        """Evaluate center control"""
        score = 0
        
        for center_square in self.CENTER_SQUARES + self.EXTENDED_CENTER:
            row, col = center_square
            
            # Check who controls this square
            our_control = self.square_controlled_by_color(row, col, self.color, board)
            opponent_control = self.square_controlled_by_color(row, col, self.opponent_color, board)
            
            if our_control and not opponent_control:
                score += 2 if center_square in self.CENTER_SQUARES else 1
            elif opponent_control and not our_control:
                score -= 2 if center_square in self.CENTER_SQUARES else 1
                
        return score

    def square_controlled_by_color(self, row: int, col: int, color: str, board) -> bool:
        """Check if a square is controlled by pieces of given color"""
        for r in range(8):
            for c in range(8):
                piece = board.get_piece(r, c)
                if piece and piece.color == color:
                    valid_moves = board.get_valid_moves(piece)
                    if (row, col) in valid_moves:
                        return True
        return False

    def evaluate_development(self, board) -> float:
        """Evaluate piece development (mainly for opening/early middlegame)"""
        if self.game_phase == 'endgame':
            return 0
            
        score = 0
        
        # Knights and bishops developed
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece and piece.piece_type in ['knight', 'bishop']:
                    is_developed = False
                    
                    if piece.color == 'white':
                        is_developed = row < 6  # Moved from back rank
                    else:
                        is_developed = row > 1  # Moved from back rank
                        
                    if is_developed:
                        if piece.color == self.color:
                            score += 15
                        else:
                            score -= 15
                            
        return score

    def evaluate_pawn_structure(self, board) -> float:
        """Evaluate pawn structure"""
        score = 0
        
        our_pawns = []
        opponent_pawns = []
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece and piece.piece_type == 'pawn':
                    if piece.color == self.color:
                        our_pawns.append((row, col))
                    else:
                        opponent_pawns.append((row, col))
        
        # Doubled pawns penalty
        our_files = [col for _, col in our_pawns]
        opponent_files = [col for _, col in opponent_pawns]
        
        for file in range(8):
            our_count = our_files.count(file)
            opponent_count = opponent_files.count(file)
            
            if our_count > 1:
                score -= (our_count - 1) * 15  # Penalty for doubled pawns
            if opponent_count > 1:
                score += (opponent_count - 1) * 15
                
        return score

    def evaluate_king_safety(self, board) -> float:
        """Evaluate king safety"""
        score = 0
        
        our_king_pos = self.find_king(self.color, board)
        opponent_king_pos = self.find_king(self.opponent_color, board)
        
        if our_king_pos:
            our_safety = self.calculate_king_safety(our_king_pos, self.color, board)
            score += our_safety
            
        if opponent_king_pos:
            opponent_safety = self.calculate_king_safety(opponent_king_pos, self.opponent_color, board)
            score -= opponent_safety
            
        return score

    def find_king(self, color: str, board) -> Optional[Tuple[int, int]]:
        """Find king position"""
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece and piece.piece_type == 'king' and piece.color == color:
                    return (row, col)
        return None

    def calculate_king_safety(self, king_pos: Tuple[int, int], color: str, board) -> float:
        """Calculate king safety score"""
        safety_score = 0
        row, col = king_pos
        
        # Pawn shield
        pawn_shield_score = 0
        if color == 'white':
            shield_squares = [(row-1, col-1), (row-1, col), (row-1, col+1)]
        else:
            shield_squares = [(row+1, col-1), (row+1, col), (row+1, col+1)]
            
        for shield_row, shield_col in shield_squares:
            if 0 <= shield_row < 8 and 0 <= shield_col < 8:
                piece = board.get_piece(shield_row, shield_col)
                if piece and piece.piece_type == 'pawn' and piece.color == color:
                    pawn_shield_score += 10
                    
        safety_score += pawn_shield_score
        
        # King exposure penalty
        exposed_squares = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    if not board.get_piece(new_row, new_col):
                        exposed_squares += 1
                        
        safety_score -= exposed_squares * 5
        
        return safety_score

    def evaluate_tactical_patterns(self, board) -> float:
        """Evaluate tactical patterns (pins, forks, skewers)"""
        score = 0
        
        # Look for hanging pieces
        hanging_score = self.evaluate_hanging_pieces(board)
        score += hanging_score
        
        # Look for pins and skewers
        pin_skewer_score = self.evaluate_pins_and_skewers(board)
        score += pin_skewer_score
        
        return score

    def evaluate_hanging_pieces(self, board) -> float:
        """Evaluate hanging (undefended) pieces"""
        score = 0
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece:
                    is_defended = self.is_piece_defended(row, col, board, piece.color)
                    is_attacked = self.is_piece_attacked(row, col, board, piece.color)
                    
                    if is_attacked and not is_defended:
                        piece_value = self.PIECE_VALUES.get(piece.piece_type, 0)
                        if piece.color == self.color:
                            score -= piece_value * 0.5  # Our hanging piece
                        else:
                            score += piece_value * 0.5  # Opponent's hanging piece
                            
        return score

    def is_piece_defended(self, row: int, col: int, board, piece_color: str) -> bool:
        """Check if piece is defended by same color pieces"""
        for r in range(8):
            for c in range(8):
                piece = board.get_piece(r, c)
                if piece and piece.color == piece_color and (r != row or c != col):
                    valid_moves = board.get_valid_moves(piece)
                    if (row, col) in valid_moves:
                        return True
        return False

    def is_piece_attacked(self, row: int, col: int, board, piece_color: str) -> bool:
        """Check if piece is attacked by opponent pieces"""
        opponent_color = 'white' if piece_color == 'black' else 'black'
        return self.square_controlled_by_color(row, col, opponent_color, board)

    def evaluate_pins_and_skewers(self, board) -> float:
        """Evaluate pins and skewers (simplified)"""
        score = 0
        
        # This is a simplified implementation
        # A full implementation would check for pieces aligned with valuable targets
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece and piece.piece_type in ['bishop', 'rook', 'queen']:
                    pin_score = self.check_for_pins(row, col, piece, board)
                    if piece.color == self.color:
                        score += pin_score
                    else:
                        score -= pin_score
                        
        return score

    def check_for_pins(self, row: int, col: int, piece, board) -> float:
        """Check if piece is creating pins (simplified)"""
        # This would need more sophisticated implementation
        # For now, just a basic check
        directions = []
        if piece.piece_type in ['rook', 'queen']:
            directions.extend([(0, 1), (0, -1), (1, 0), (-1, 0)])
        if piece.piece_type in ['bishop', 'queen']:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            
        pin_score = 0
        for dr, dc in directions:
            # Check along direction for potential pins
            pieces_in_line = []
            for i in range(1, 8):
                new_row, new_col = row + i * dr, col + i * dc
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                target = board.get_piece(new_row, new_col)
                if target:
                    pieces_in_line.append(target)
                    if len(pieces_in_line) >= 2:
                        break
                        
            # Simple pin detection: enemy piece followed by valuable enemy piece
            if len(pieces_in_line) == 2:
                first, second = pieces_in_line
                if (first.color != piece.color and second.color != piece.color and
                    self.PIECE_VALUES.get(second.piece_type, 0) > self.PIECE_VALUES.get(first.piece_type, 0)):
                    pin_score += 20
                    
        return pin_score

    def evaluate_endgame_factors(self, board) -> float:
        """Evaluate endgame-specific factors"""
        score = 0
        
        # King activity in endgame
        our_king_pos = self.find_king(self.color, board)
        opponent_king_pos = self.find_king(self.opponent_color, board)
        
        if our_king_pos and opponent_king_pos:
            # Active king bonus
            our_king_row, our_king_col = our_king_pos
            opponent_king_row, opponent_king_col = opponent_king_pos
            
            # Kings closer to center are better in endgame
            our_king_centralization = 4 - max(abs(our_king_row - 3.5), abs(our_king_col - 3.5))
            opponent_king_centralization = 4 - max(abs(opponent_king_row - 3.5), abs(opponent_king_col - 3.5))
            
            score += (our_king_centralization - opponent_king_centralization) * 10
            
        # Pawn promotion potential
        pawn_promotion_score = self.evaluate_pawn_promotion(board)
        score += pawn_promotion_score
        
        return score

    def evaluate_pawn_promotion(self, board) -> float:
        """Evaluate pawn promotion potential"""
        score = 0
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece and piece.piece_type == 'pawn':
                    if piece.color == self.color:
                        # Our pawns - closer to promotion is better
                        if piece.color == 'white':
                            promotion_distance = row  # White pawns move up (row decreases)
                        else:
                            promotion_distance = 7 - row  # Black pawns move down (row increases)
                        
                        if promotion_distance <= 3:  # Advanced pawn
                            score += (4 - promotion_distance) * 25
                    else:
                        # Opponent pawns - closer to promotion is worse for us
                        if piece.color == 'white':
                            promotion_distance = row
                        else:
                            promotion_distance = 7 - row
                            
                        if promotion_distance <= 3:
                            score -= (4 - promotion_distance) * 25
                            
        return score

    def update_game_phase(self):
        """Update current game phase based on material"""
        total_material = 0
        queen_count = 0
        
        for row in range(8):
            for col in range(8):
                piece = self.board.get_piece(row, col)
                if piece and piece.piece_type != 'king':
                    total_material += self.PIECE_VALUES.get(piece.piece_type, 0)
                    if piece.piece_type == 'queen':
                        queen_count += 1
        
        # Simple phase detection
        if total_material > 6000:  # Opening/Early middlegame
            self.game_phase = 'opening'
        elif total_material > 2500 or queen_count >= 2:  # Middlegame
            self.game_phase = 'middlegame'
        else:  # Endgame
            self.game_phase = 'endgame'

    def is_move_safe(self, from_row, from_col, to_row, to_col):
        temp_board = self.board.copy()
        temp_board.make_move(from_row, from_col, to_row, to_col)
        return not temp_board.is_in_check(self.color)

    def is_move_safe_on_board(self, from_row, from_col, to_row, to_col, board):
        """Simulate a move on the given board and check if the king remains safe."""
        temp_board = board.copy()
        temp_board.make_move(from_row, from_col, to_row, to_col)
        return not temp_board.is_in_check(self.color)


    def make_temp_move(self, move: Tuple[Tuple[int, int], Tuple[int, int]]):
        """Make temporary move and return new board"""
        (from_pos, to_pos) = move
        temp_board = self.board.copy()
        temp_board.make_move(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
        return temp_board

    def make_temp_move_on_board(self, move: Tuple[Tuple[int, int], Tuple[int, int]], board):
        """Make temporary move on given board"""
        (from_pos, to_pos) = move
        temp_board = copy.deepcopy(board)
        temp_board.make_move(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
        return temp_board

    def is_game_over(self, board) -> bool:
        """Check if game is over (checkmate, stalemate, etc.)"""
        # This would need to be implemented based on your ChessBoard class
        # For now, simplified check
        try:
            # Try to find at least one legal move
            for row in range(8):
                for col in range(8):
                    piece = board.get_piece(row, col)
                    if piece and piece.color == board.current_player:
                        valid_moves = board.get_valid_moves(piece)
                        for move_row, move_col in valid_moves:
                            if self.is_move_safe_on_board((row, col), (move_row, move_col), board):
                                return False  # Found legal move
            return True  # No legal moves found
        except:
            return False

    def avoid_repetition(self, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """Check if move leads to threefold repetition"""
        if len(self.move_history) < 4:
            return True  # Not enough moves for repetition
            
        # Simple repetition check: avoid immediate move reversal
        if len(self.move_history) >= 2:
            last_move = self.move_history[-1] if self.move_history else None
            if last_move:
                # Check if current move reverses the last move
                if (move[0] == last_move[1] and move[1] == last_move[0]):
                    return False  # This would be an immediate reversal
                    
        return True

    def make_move(self) -> bool:
        """Make the best move"""
        move = self.get_best_move()
        if not move:
            return False
            
        # Anti-repetition check
        if not self.avoid_repetition(move):
            print("[AdvancedBot] Avoiding repetition, finding alternative move")
            # Try to find alternative move (simplified)
            moves = self.generate_ordered_moves(self.color)
            for alt_move in moves[:5]:  # Try top 5 alternatives
                if (alt_move != move and 
                    self.is_move_safe(alt_move[0][0], alt_move[0][1], alt_move[1][0], alt_move[1][1]) and
                    self.avoid_repetition(alt_move)):
                    move = alt_move
                    break
        
        (from_row, from_col), (to_row, to_col) = move
        self.move_history.append(move)
        
        # Store position for repetition detection
        position_key = self.get_position_key()
        self.position_history.append(position_key)
        
        success = self.board.make_move(from_row, from_col, to_row, to_col)
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        if success:
            print(f"[AdvancedBot] Move: {move}, Phase: {self.game_phase}, Eval: {self.last_evaluation}")
            
        return success

    def get_position_key(self) -> str:
        """Generate simple position key for repetition detection"""
        # This is a simplified implementation
        # A full implementation would include castling rights, en passant, etc.
        key = ""
        for row in range(8):
            for col in range(8):
                piece = self.board.get_piece(row, col)
                if piece:
                    key += f"{piece.color[0]}{piece.piece_type[0]}{row}{col}"
                else:
                    key += "."
        return key

    def get_move_explanation(self, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
        """Get human-readable explanation of why this move was chosen"""
        (from_pos, to_pos) = move
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        piece = self.board.get_piece(from_row, from_col)
        target = self.board.get_piece(to_row, to_col)
        
        explanations = []
        
        if target:
            target_value = self.PIECE_VALUES.get(target.piece_type, 0)
            piece_value = self.PIECE_VALUES.get(piece.piece_type, 0)
            if target_value >= piece_value:
                explanations.append(f"Captures {target.piece_type} (good trade)")
            else:
                explanations.append(f"Captures {target.piece_type}")
        
        if (to_row, to_col) in self.CENTER_SQUARES:
            explanations.append("Controls center")
            
        # Check if move gives check
        temp_board = self.make_temp_move(move)
        if temp_board.is_in_check(self.opponent_color):
            explanations.append("Gives check")
            
        # Check if move develops piece
        if piece and piece.piece_type in ['knight', 'bishop'] and self.game_phase == 'opening':
            explanations.append("Develops piece")
            
        if not explanations:
            explanations.append("Positional improvement")
            
        return " | ".join(explanations)

    def set_difficulty(self, difficulty: str):
        """Adjust bot difficulty"""
        difficulty_settings = {
            'easy': {'depth': 2, 'thinking_time': 0.5},
            'medium': {'depth': 3, 'thinking_time': 2.0},
            'hard': {'depth': 4, 'thinking_time': 5.0},
            'expert': {'depth': 5, 'thinking_time': 10.0}
        }
        
        if difficulty in difficulty_settings:
            settings = difficulty_settings[difficulty]
            self.max_depth = settings['depth']
            print(f"[AdvancedBot] Difficulty set to {difficulty}, depth: {self.max_depth}")

    def get_analysis(self) -> Dict:
        """Get current position analysis"""
        analysis = {
            'material_balance': self.evaluate_material(self.board),
            'positional_score': self.evaluate_positional_factors(self.board),
            'king_safety': self.evaluate_king_safety(self.board),
            'tactical_score': self.evaluate_tactical_patterns(self.board),
            'game_phase': self.game_phase,
            'total_evaluation': self.last_evaluation
        }
        return analysis

class GameModeSelector:
    """Startup window to choose game mode"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chess Game - Choose Mode")
        self.root.geometry("350x200")
        self.root.resizable(False, False)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (350 // 2)
        y = (self.root.winfo_screenheight() // 2) - (200 // 2)
        self.root.geometry(f"350x200+{x}+{y}")
        
        self.selected_mode = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the mode selection UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Chess Game", 
            font=('Arial', 18, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = ttk.Label(
            main_frame, 
            text="Choose Game Mode", 
            font=('Arial', 12)
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(expand=True)
        
        # Human vs Human button
        human_button = ttk.Button(
            buttons_frame,
            text="Play vs Human",
            command=lambda: self.select_mode('human'),
            width=20
        )
        human_button.pack(pady=5)
        
        # Human vs Bot button
        bot_button = ttk.Button(
            buttons_frame,
            text="Play vs Bot",
            command=lambda: self.select_mode('bot'),
            width=20
        )
        bot_button.pack(pady=5)
        
        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="In bot mode, you play as White and bot plays as Black",
            font=('Arial', 9),
            foreground='gray'
        )
        instructions.pack(pady=(20, 0))
    
    def select_mode(self, mode: str):
        """Select game mode and close window"""
        self.selected_mode = mode
        self.root.quit()
        self.root.destroy()
    
    def run(self) -> str:
        """Run the mode selector and return selected mode"""
        self.root.mainloop()
        return self.selected_mode

class Timer:
    def __init__(self, minutes: int = 10):
        self.white_time = minutes * 60
        self.black_time = minutes * 60
        self.active_color = 'white'
        self.last_update = time.time()
        self.running = False

    def start(self):
        self.running = True
        self.last_update = time.time()

    def stop(self):
        self.running = False

    def switch_player(self):
        self.update()
        self.active_color = 'black' if self.active_color == 'white' else 'white'
        self.last_update = time.time()

    def update(self):
        if not self.running:
            return
        elapsed = time.time() - self.last_update
        if self.active_color == 'white':
            self.white_time -= elapsed
        else:
            self.black_time -= elapsed
        self.last_update = time.time()

    def get_time_display(self, color: str) -> str:
        time_left = self.white_time if color == 'white' else self.black_time
        time_left = max(0, time_left)
        minutes = int(time_left // 60)
        seconds = int(time_left % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def is_time_up(self, color: str) -> bool:
        time_left = self.white_time if color == 'white' else self.black_time
        return time_left <= 0

class ChessGUI:
    """Main GUI class for the chess game"""
    
    def __init__(self, vs_bot: bool = False):
        self.root = tk.Tk()
        self.root.title("Chess Game")
        self.root.resizable(False, False)
        
        # Game components
        self.board = ChessBoard()
        self.timer = Timer(10)  # 10 minutes per player
        self.vs_bot = vs_bot
        chess_board = ChessBoard()  # ✅ create board instance
        if self.vs_bot:
            self.bot = AdvancedChessBot(self.board, color='black', max_depth=3)
            self.bot.set_difficulty('expert')
        else:
            self.bot = None
        # GUI state
        self.selected_piece = None
        self.selected_square = None
        self.valid_moves = []
        self.board_squares = []
        self.last_move = None  # For highlighting bot moves
        self.bot_thinking = False
        
        # Create GUI
        self.setup_gui()
        self.update_board_display()
        self.update_timer_display()
        
        # Start timer update loop
        self.timer.start()
        self.update_timer_loop()
    
    def setup_gui(self):
        """Set up the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Left panel (board)
        board_frame = ttk.Frame(main_frame)
        board_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        # Board canvas
        self.canvas = tk.Canvas(
            board_frame,
            width=BOARD_SIZE * SQUARE_SIZE,
            height=BOARD_SIZE * SQUARE_SIZE,
            bg='white'
        )
        self.canvas.pack()
        
        # Create board squares
        for row in range(BOARD_SIZE):
            square_row = []
            for col in range(BOARD_SIZE):
                x1 = col * SQUARE_SIZE
                y1 = row * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                
                color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
                square_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline=color
                )
                square_row.append(square_id)
            self.board_squares.append(square_row)
        
        # Bind click events
        self.canvas.bind("<Button-1>", self.on_square_click)
        
        # Right panel (controls and info)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Game info
        info_frame = ttk.LabelFrame(right_frame, text="Game Info")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        mode_text = "vs Bot" if self.vs_bot else "vs Human"
        self.mode_label = ttk.Label(info_frame, text=f"Mode: {mode_text}", font=('Arial', 10))
        self.mode_label.pack(pady=2)
        
        self.current_player_label = ttk.Label(info_frame, text="Current Player: White", font=('Arial', 12, 'bold'))
        self.current_player_label.pack(pady=5)
        
        self.status_label = ttk.Label(info_frame, text="Game in progress", font=('Arial', 10))
        self.status_label.pack(pady=5)
        
        # Timer display
        timer_frame = ttk.LabelFrame(right_frame, text="Timer")
        timer_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.white_timer_label = ttk.Label(timer_frame, text="White: 10:00", font=('Arial', 11))
        self.white_timer_label.pack()
        
        self.black_timer_label = ttk.Label(timer_frame, text="Black: 10:00", font=('Arial', 11))
        self.black_timer_label.pack()
        
        # Controls
        controls_frame = ttk.LabelFrame(right_frame, text="Controls")
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="Undo Move", command=self.undo_move).pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="New Game", command=self.new_game).pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="Pause/Resume", command=self.toggle_pause).pack(fill=tk.X, pady=2)
        
        # Move history
        history_frame = ttk.LabelFrame(right_frame, text="Move History")
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable text widget for move history
        history_scroll_frame = ttk.Frame(history_frame)
        history_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.history_text = tk.Text(
            history_scroll_frame,
            width=20,
            height=15,
            font=('Courier', 9),
            state=tk.DISABLED
        )
        history_scrollbar = ttk.Scrollbar(history_scroll_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Captured pieces display
        captured_frame = ttk.LabelFrame(right_frame, text="Captured Pieces")
        captured_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.white_captured_label = ttk.Label(captured_frame, text="White captured: ", font=('Arial', 9))
        self.white_captured_label.pack(anchor=tk.W)
        
        self.black_captured_label = ttk.Label(captured_frame, text="Black captured: ", font=('Arial', 9))
        self.black_captured_label.pack(anchor=tk.W)
    
    def on_square_click(self, event):
        """Handle square click events"""
        # Don't allow moves if bot is thinking
        if self.bot_thinking:
            return
            
        col = event.x // SQUARE_SIZE
        row = event.y // SQUARE_SIZE
        
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return
        
        # Check if game is over
        if self.is_game_over():
            return
        
        # In bot mode, only allow human (white) to make moves
        if self.vs_bot and self.board.current_player == 'black':
            return
        
        piece = self.board.get_piece(row, col)
        
        # If no piece is selected
        if not self.selected_piece:
            if piece and piece.color == self.board.current_player:
                self.select_piece(row, col)
        else:
            # If clicking on the same square, deselect
            if (row, col) == self.selected_square:
                self.deselect_piece()
            # If clicking on valid move
            elif (row, col) in self.valid_moves:
                self.make_move(row, col)
            # If clicking on own piece, select it
            elif piece and piece.color == self.board.current_player:
                self.select_piece(row, col)
            else:
                self.deselect_piece()
    
    def select_piece(self, row: int, col: int):
        """Select a piece and show valid moves"""
        piece = self.board.get_piece(row, col)
        if not piece:
            return
        
        self.selected_piece = piece
        self.selected_square = (row, col)
        self.valid_moves = self.board.get_valid_moves(piece)
        
        self.update_board_display()
    
    def deselect_piece(self):
        """Deselect the current piece"""
        self.selected_piece = None
        self.selected_square = None
        self.valid_moves = []
        self.update_board_display()
    
    def make_move(self, to_row: int, to_col: int):
        """Make a move"""
        if not self.selected_piece:
            return
        
        from_row, from_col = self.selected_square
        
        # Check for pawn promotion
        if (self.selected_piece.piece_type == 'pawn' and
            ((self.selected_piece.color == 'white' and to_row == 0) or
             (self.selected_piece.color == 'black' and to_row == 7))):
            self.handle_pawn_promotion(from_row, from_col, to_row, to_col)
        else:
            self.execute_move(from_row, from_col, to_row, to_col)

    def update_board_display(self):
        """Redraw the board and update piece positions"""
        self.canvas.delete("piece")
        self.canvas.delete("highlight")

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
                self.canvas.itemconfig(self.board_squares[row][col], fill=color, outline=color)

        for move_row, move_col in self.valid_moves:
            self.canvas.create_oval(
                move_col * SQUARE_SIZE + SQUARE_SIZE // 4,
                move_row * SQUARE_SIZE + SQUARE_SIZE // 4,
                move_col * SQUARE_SIZE + 3 * SQUARE_SIZE // 4,
                move_row * SQUARE_SIZE + 3 * SQUARE_SIZE // 4,
                fill=VALID_MOVE_COLOR,
                outline=VALID_MOVE_COLOR,
                tags="highlight"
            )

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board.get_piece(row, col)
                if piece:
                    x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                    y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                    self.canvas.create_text(
                        x, y,
                        text=piece.get_symbol(),
                        font=('Arial', 32),
                        fill='black',
                        tags="piece"
                    )

    def handle_pawn_promotion(self, from_row: int, from_col: int, to_row: int, to_col: int):
        """Handle pawn promotion dialog"""
        promotion_window = tk.Toplevel(self.root)
        promotion_window.title("Pawn Promotion")
        promotion_window.geometry("300x150")
        promotion_window.transient(self.root)
        promotion_window.grab_set()
        
        # Center the window
        promotion_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 50,
            self.root.winfo_rooty() + 50
        ))
        
        ttk.Label(promotion_window, text="Choose promotion piece:", font=('Arial', 12)).pack(pady=10)
        
        button_frame = ttk.Frame(promotion_window)
        button_frame.pack(pady=10)
        
        self.promotion_choice = None
        
        def promote_to(piece_type):
            self.promotion_choice = piece_type
            promotion_window.destroy()
        
        pieces = [('Queen', 'queen'), ('Rook', 'rook'), ('Bishop', 'bishop'), ('Knight', 'knight')]
        for i, (name, piece_type) in enumerate(pieces):
            ttk.Button(
                button_frame,
                text=name,
                command=lambda pt=piece_type: promote_to(pt)
            ).grid(row=0, column=i, padx=5)
        
        # Wait for user choice
        promotion_window.wait_window()
        
        if self.promotion_choice:
            # Manually handle promotion
            piece = self.board.get_piece(from_row, from_col)
            if self.board.make_move(from_row, from_col, to_row, to_col):
                # Change piece type after move
                promoted_piece = self.board.get_piece(to_row, to_col)
                promoted_piece.piece_type = self.promotion_choice
                self.post_move_actions()
    
    def execute_move(self, from_row: int, from_col: int, to_row: int, to_col: int):
        """Execute a move and update game state"""
        if self.board.make_move(from_row, from_col, to_row, to_col):
            self.post_move_actions()
    
    def post_move_actions(self):
        """Actions to perform after a successful move"""
        self.deselect_piece()
        self.timer.switch_player()
        self.update_board_display()
        self.update_move_history()
        self.update_captured_pieces()
        self.check_game_status()
        
        # If vs bot and now it's bot's turn, make bot move
        if (self.vs_bot and self.board.current_player == 'black' and 
            not self.is_game_over()):
            self.schedule_bot_move()
    
    def schedule_bot_move(self):
        """Schedule a bot move with thinking delay"""
        if self.bot_thinking or self.is_game_over():
            return
            
        self.bot_thinking = True
        self.status_label.config(text="Bot thinking...")
        self.root.after(500, self.make_bot_move)  # 500ms delay
    
    def make_bot_move(self):
        """Make a move for the bot"""
        if not self.vs_bot or self.board.current_player != 'black':
            self.bot_thinking = False
            return
        
        # Get bot's move
        move = self.bot.get_best_move()
        if not move:
            # Bot has no legal moves - resign
            self.bot_thinking = False
            self.end_game("White wins! Bot resigned (no legal moves)")
            return
        
        (from_row, from_col), (to_row, to_col) = move
        
        # Store the move for highlighting
        self.last_move = ((from_row, from_col), (to_row, to_col))
        
        # Check for pawn promotion
        piece = self.board.get_piece(from_row, from_col)
        if (piece and piece.piece_type == 'pawn' and to_row == 7):
            # Bot always promotes to queen
            if self.board.make_move(from_row, from_col, to_row, to_col):
                promoted_piece = self.board.get_piece(to_row, to_col)
                promoted_piece.piece_type = 'queen'
        else:
            self.board.make_move(from_row, from_col, to_row, to_col)
        
        self.bot_thinking = False
        
        # Update game state
        self.timer.switch_player()
        self.update_board_display()
        self.update_move_history()
        self.update_captured_pieces()
        self.check_game_status()
        """Update the visual board display"""
        # Clear all piece displays
        self.canvas.delete("piece")
        self.canvas.delete("highlight")
        
        # Reset square colors
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
                
                # Highlight selected square
                if self.selected_square == (row, col):
                    color = HIGHLIGHT_COLOR
                
                # Highlight king in check
                piece = self.board.get_piece(row, col)
                if (piece and piece.piece_type == 'king' and 
                    self.board.is_in_check(piece.color)):
                    color = CHECK_COLOR
                
                self.canvas.itemconfig(self.board_squares[row][col], fill=color, outline=color)
        
        # Highlight valid moves
        for move_row, move_col in self.valid_moves:
            self.canvas.create_oval(
                move_col * SQUARE_SIZE + SQUARE_SIZE // 4,
                move_row * SQUARE_SIZE + SQUARE_SIZE // 4,
                move_col * SQUARE_SIZE + 3 * SQUARE_SIZE // 4,
                move_row * SQUARE_SIZE + 3 * SQUARE_SIZE // 4,
                fill=VALID_MOVE_COLOR,
                outline=VALID_MOVE_COLOR,
                tags="highlight"
            )
        
        # Draw pieces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board.get_piece(row, col)
                if piece:
                    x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                    y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                    
                    self.canvas.create_text(
                        x, y,
                        text=piece.get_symbol(),
                        font=('Arial', 32),
                        fill='black',
                        tags="piece"
                    )
        
        # Update current player display
        self.current_player_label.config(
            text=f"Current Player: {self.board.current_player.capitalize()}"
        )
    
    def update_move_history(self):
        """Update the move history display"""
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        
        for i, move in enumerate(self.board.move_history):
            move_num = (i // 2) + 1
            notation = self.board.get_algebraic_notation(move)
            
            if i % 2 == 0:  # White's move
                self.history_text.insert(tk.END, f"{move_num}. {notation} ")
            else:  # Black's move
                self.history_text.insert(tk.END, f"{notation}\n")
        
        self.history_text.config(state=tk.DISABLED)
        self.history_text.see(tk.END)
    
    def update_captured_pieces(self):
        """Update captured pieces display"""
        white_captured = "".join([piece.get_symbol() for piece in self.board.captured_pieces['white']])
        black_captured = "".join([piece.get_symbol() for piece in self.board.captured_pieces['black']])
        
        self.white_captured_label.config(text=f"White captured: {white_captured}")
        self.black_captured_label.config(text=f"Black captured: {black_captured}")
    
    def update_timer_display(self):
        """Update timer display"""
        white_time = self.timer.get_time_display('white')
        black_time = self.timer.get_time_display('black')
        
        # Highlight active player's timer
        if self.timer.active_color == 'white':
            self.white_timer_label.config(text=f"White: {white_time}", font=('Arial', 11, 'bold'))
            self.black_timer_label.config(text=f"Black: {black_time}", font=('Arial', 11))
        else:
            self.white_timer_label.config(text=f"White: {white_time}", font=('Arial', 11))
            self.black_timer_label.config(text=f"Black: {black_time}", font=('Arial', 11, 'bold'))
    
    def update_timer_loop(self):
        """Update timer display in a loop"""
        self.timer.update()
        self.update_timer_display()
        
        # Check for time up
        if self.timer.is_time_up('white'):
            self.end_game("Black wins by timeout!")
        elif self.timer.is_time_up('black'):
            self.end_game("White wins by timeout!")
        
        # Schedule next update
        self.root.after(100, self.update_timer_loop)
    
    def check_game_status(self):
        """Check if the game is over"""
        current_player = self.board.current_player
        
        if self.board.is_checkmate(current_player):
            winner = 'Black' if current_player == 'white' else 'White'
            self.end_game(f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate(current_player):
            self.end_game("Stalemate! It's a draw!")
        elif self.board.is_in_check(current_player):
            self.status_label.config(text=f"{current_player.capitalize()} is in check!")
        else:
            self.status_label.config(text="Game in progress")
    
    def is_game_over(self) -> bool:
        """Check if game is over"""
        return (self.board.is_checkmate('white') or self.board.is_checkmate('black') or
                self.board.is_stalemate('white') or self.board.is_stalemate('black') or
                self.timer.is_time_up('white') or self.timer.is_time_up('black'))
    
    def end_game(self, message: str):
        """End the game with a message"""
        self.timer.stop()
        self.status_label.config(text=message)
        messagebox.showinfo("Game Over", message)
    
    def undo_move(self):
        """Undo the last move"""
        if self.board.undo_last_move():
            self.deselect_piece()
            self.timer.switch_player()  # Switch timer back
            self.update_board_display()
            self.update_move_history()
            self.update_captured_pieces()
            self.check_game_status()
    
    def new_game(self):
        """Start a new game"""
        if messagebox.askyesno("New Game", "Are you sure you want to start a new game?"):
            self.board = ChessBoard()
            self.timer = Timer(10)
            self.timer.start()
            chess_board = ChessBoard()  # ✅ create board instance
            if self.vs_bot:
                self.bot = AdvancedChessBot(self.board, color='black', max_depth=3)
                self.bot.set_difficulty('expert')
            else:
                self.bot = None
            self.deselect_piece()
            self.last_move = None
            self.bot_thinking = False
            self.update_board_display()
            self.update_move_history()
            self.update_captured_pieces()
            self.status_label.config(text="Game in progress")
    
    def toggle_pause(self):
        """Toggle timer pause"""
        if self.timer.running:
            self.timer.stop()
        else:
            self.timer.start()
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()

def main():
    """Main function to run the chess game"""
    try:
        # Step 1: Show game mode selector
        selector = GameModeSelector()
        selected_mode = selector.run()

        # Step 2: Determine if playing against bot
        vs_bot = selected_mode == 'bot'

        # Step 3: Start the game with selected mode
        game = ChessGUI(vs_bot=vs_bot)
        game.run()

    except Exception as e:
        print(f"Error running chess game: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()