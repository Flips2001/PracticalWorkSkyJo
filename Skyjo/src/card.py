from dataclasses import dataclass


@dataclass(init=False)
class Card:
    __value: int
    face_up: bool = False

    def __init__(self, value: int, face_up: bool = False):
        self.__value = value
        self.face_up = face_up

    def reveal(self):
        self.face_up = True

    def hide(self):
        self.face_up = False

    def is_hidden(self) -> bool:
        return not self.face_up

    def get_value(self):
        if self.face_up:
            return self.__value
        raise ValueError("Card is face down; value is not accessible.")

    def _get_value_for_engine(self) -> int:
        return self.__value

    def __repr__(self):
        if self.face_up:
            return f"[{self.get_value()}]"
        return "[X]"
