from litecrawl import _compute_stable_hash


def test_compute_stable_hash_treats_string_and_bytes_equally():
    assert _compute_stable_hash("hello") == _compute_stable_hash(b"hello")


def test_compute_stable_hash_is_order_insensitive_for_dicts():
    first = {"a": 1, "b": 2}
    second = {"b": 2, "a": 1}
    changed = {"a": 2, "b": 1}

    assert _compute_stable_hash(first) == _compute_stable_hash(second)
    assert _compute_stable_hash(first) != _compute_stable_hash(changed)


def test_compute_stable_hash_handles_empty_values():
    assert _compute_stable_hash(None) == _compute_stable_hash(b"")
