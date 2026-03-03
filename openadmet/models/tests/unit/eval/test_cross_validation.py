import numpy as np

from openadmet.models.eval.cross_validation import repeated_group_k_fold


def _make_grouped_data(n_groups=12, samples_per_group=2):
    groups = np.repeat(np.arange(n_groups), samples_per_group)
    X = np.arange(groups.size).reshape(-1, 1)
    y = np.arange(groups.size)
    return X, y, groups


def _repeat_signature(groups, test_inds, n_splits, repeat_idx):
    start = repeat_idx * n_splits
    fold_groups = [
        tuple(np.sort(np.unique(groups[test_inds[start + fold]])))
        for fold in range(n_splits)
    ]
    return tuple(sorted(fold_groups))


def test_repeated_group_k_fold_returns_expected_fold_count():
    X, y, groups = _make_grouped_data()
    n_splits = 4
    n_repeats = 3

    train_inds, test_inds = repeated_group_k_fold(
        X, y, groups, n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )

    assert len(train_inds) == n_splits * n_repeats
    assert len(test_inds) == n_splits * n_repeats


def test_repeated_group_k_fold_keeps_train_test_groups_disjoint():
    X, y, groups = _make_grouped_data()

    train_inds, test_inds = repeated_group_k_fold(
        X, y, groups, n_splits=4, n_repeats=2, random_state=7
    )

    for train_idx, test_idx in zip(train_inds, test_inds):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_repeated_group_k_fold_is_reproducible_for_same_random_state():
    X, y, groups = _make_grouped_data()

    train_inds_1, test_inds_1 = repeated_group_k_fold(
        X, y, groups, n_splits=4, n_repeats=2, random_state=13
    )
    train_inds_2, test_inds_2 = repeated_group_k_fold(
        X, y, groups, n_splits=4, n_repeats=2, random_state=13
    )

    assert all(np.array_equal(a, b) for a, b in zip(train_inds_1, train_inds_2))
    assert all(np.array_equal(a, b) for a, b in zip(test_inds_1, test_inds_2))


def test_repeated_group_k_fold_repeats_are_not_identical():
    X, y, groups = _make_grouped_data()
    n_splits = 4

    _, test_inds = repeated_group_k_fold(
        X, y, groups, n_splits=n_splits, n_repeats=2, random_state=99
    )

    repeat_0 = _repeat_signature(groups, test_inds, n_splits=n_splits, repeat_idx=0)
    repeat_1 = _repeat_signature(groups, test_inds, n_splits=n_splits, repeat_idx=1)

    assert repeat_0 != repeat_1
