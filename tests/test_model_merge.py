from sdkit.train import merge_multiple_models


# section 1 - two models
def test_1_0__two_models__with_same_keys():
    actual = merge_multiple_models(
        models=[
            {"a": 1, "b": 2},
            {"a": 10, "b": 20},
        ],
        alphas=[0.5, 0.5],
    )
    expected = {"a": (1 * 0.5 + 10 * 0.5), "b": (2 * 0.5 + 20 * 0.5)}
    assert actual == expected


def test_1_1__two_models__with_different_keys():
    actual = merge_multiple_models(
        models=[
            {"a": 1, "b": 2},
            {"a": 10, "b": 20, "c": 30},
        ],
        alphas=[0.5, 0.5],
    )
    expected = {"a": (1 * 0.5 + 10 * 0.5), "b": (2 * 0.5 + 20 * 0.5), "c": 30}
    assert actual == expected


def test_1_2__two_models__with_different_keys():
    actual = merge_multiple_models(
        models=[
            {"a": 1, "b": 2, "c": 30},
            {"a": 10, "b": 20},
        ],
        alphas=[0.5, 0.5],
    )
    expected = {"a": (1 * 0.5 + 10 * 0.5), "b": (2 * 0.5 + 20 * 0.5), "c": 30}
    assert actual == expected


def test_1_3__two_models__with_different_keys():
    actual = merge_multiple_models(
        models=[
            {"a": 1, "b": 2, "c": 30},
            {"a": 10, "b": 20, "d": 40},
        ],
        alphas=[0.5, 0.5],
    )
    expected = {"a": (1 * 0.5 + 10 * 0.5), "b": (2 * 0.5 + 20 * 0.5), "c": 30, "d": 40}
    assert actual == expected


# section 2 - three models
def test_2_0__three_models__with_same_keys():
    actual = merge_multiple_models(
        models=[
            {"a": 1, "b": 2},
            {"a": 10, "b": 20},
            {"a": 100, "b": 200},
        ],
        alphas=[0.25, 0.5, 0.25],
    )
    expected = {"a": (1 * 0.25 + 10 * 0.5 + 100 * 0.25), "b": (2 * 0.25 + 20 * 0.5 + 200 * 0.25)}
    assert actual == expected


def test_2_1__three_models__with_different_keys():
    actual = merge_multiple_models(
        models=[
            {"a": 1, "b": 2},
            {"a": 10, "b": 20, "c": 30},
            {"a": 100, "b": 200},
        ],
        alphas=[0.25, 0.5, 0.25],
    )
    expected = {"a": (1 * 0.25 + 10 * 0.5 + 100 * 0.25), "b": (2 * 0.25 + 20 * 0.5 + 200 * 0.25), "c": 30}
    assert actual == expected


def test_2_2__three_models__with_different_keys():
    actual = merge_multiple_models(
        models=[
            {"a": 1, "b": 2, "c": 3},
            {"a": 10, "b": 20, "d": 30},
            {"a": 100, "b": 200},
        ],
        alphas=[0.25, 0.5, 0.25],
    )
    expected = {"a": (1 * 0.25 + 10 * 0.5 + 100 * 0.25), "b": (2 * 0.25 + 20 * 0.5 + 200 * 0.25), "c": 3, "d": 30}
    assert actual == expected


def test_2_3__three_models__with_different_keys():
    actual = merge_multiple_models(
        models=[
            {"a": 1, "b": 2, "c": 3},
            {"a": 10, "b": 20},
            {"a": 100, "b": 200, "c": 300},
        ],
        alphas=[0.1, 0.5, 0.4],
    )
    c_alpha = [0.1 / (0.1 + 0.4), 0, 0.4 / (0.1 + 0.4)]  # "c" will only normalize between the two models that have "c"

    expected = {
        "a": (1 * 0.1 + 10 * 0.5 + 100 * 0.4),
        "b": (2 * 0.1 + 20 * 0.5 + 200 * 0.4),
        "c": (3 * c_alpha[0] + 300 * c_alpha[2]),
    }
    assert actual == expected


# section 3 - misc
def test_3_0__misc__merges_only_if_key_pattern_matches():
    actual = merge_multiple_models(
        models=[
            {"model_a": 1, "model_b": 2, "model_c": 3},
            {"model_a": 10, "model_b": 20, "d": 30},
        ],
        alphas=[0.5, 0.5],
        key_name_pattern="model_",
    )
    expected = {"model_a": (1 * 0.5 + 10 * 0.5), "model_b": (2 * 0.5 + 20 * 0.5), "model_c": 3}
    assert actual == expected
