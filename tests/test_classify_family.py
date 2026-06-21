from src.eval.classify_family import FAMILIES, classify_family


def test_bit_manipulation():
    prompt = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers..."
    )
    assert classify_family(prompt) == "bit_manipulation"


def test_gravity():
    prompt = "In Alice's Wonderland, the gravitational constant has been secretly changed..."
    assert classify_family(prompt) == "gravity"


def test_unit_conversion():
    prompt = "In Alice's Wonderland, a secret unit conversion is applied to measurements..."
    assert classify_family(prompt) == "unit_conversion"


def test_cipher():
    prompt = "In Alice's Wonderland, secret encryption rules are used on text..."
    assert classify_family(prompt) == "cipher"


def test_numeral():
    prompt = "In Alice's Wonderland, numbers are secretly converted into a different numeral system..."
    assert classify_family(prompt) == "numeral"


def test_transformation():
    prompt = "In Alice's Wonderland, a secret set of transformation rules is applied to equations..."
    assert classify_family(prompt) == "transformation"


def test_unknown_when_no_match():
    assert classify_family("totally unrelated text") == "unknown"


def test_all_families_in_constant():
    # If we ever add a new family, this fails loudly until FAMILIES is updated
    expected = {"bit_manipulation", "cipher", "gravity", "numeral", "transformation", "unit_conversion"}
    assert set(FAMILIES) == expected
