from dataclasses import dataclass

@dataclass
class Card:
    value: int
    face_up: bool = False

    def reveal(self):
        self.face_up = True

    def hide(self):
        self.face_up = False

    def is_hidden(self) -> bool:
        return not self.face_up

    def get_value(self):
        if self.face_up:
            return self.value
        raise ValueError("Card is face down; value is not accessible.")

    def __repr__(self):
        if self.face_up:
            return f"[{self.value}]"
        return "[X]"