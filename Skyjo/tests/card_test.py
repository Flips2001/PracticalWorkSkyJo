import pytest
from Skyjo.src.card import Card


@pytest.fixture
def card():
    return Card(3)


def test_print_card_face_down(card):
    assert repr(card) == "[X]"


def test_print_card_face_up(card):
    card.reveal()
    assert repr(card) == "[3]"


def test_reveal_card(card):
    card.reveal()
    assert not card.is_hidden()
    assert card.get_value() == 3


def test_get_value_face_down(card):
    with pytest.raises(ValueError):
        card.get_value()


def test_hide_card(card):
    card.reveal()
    card.hide()
    assert card.is_hidden()
    with pytest.raises(ValueError):
        card.get_value()
