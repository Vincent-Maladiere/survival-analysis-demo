import re

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ..metrics import BrierScoreSampler, brier_score, integrated_brier_score

X = load_regression_dataset()
X_train, X_test = X.iloc[:150], X.iloc[150:]
y_train = dict(
    event=X_train["E"],
    duration=X_train["T"],
)
y_test = dict(
    event=X_test["E"],
    duration=X_test["T"],
)
times = np.arange(
    y_test["duration"].min(),
    y_test["duration"].max() - 1,
)

est = CoxPHFitter().fit(X_train, duration_col="T", event_col="E")
y_pred = est.predict_survival_function(X_test, times)
y_pred = y_pred.T.values  # (n_samples, n_times)


@pytest.mark.parametrize("event_of_interest", [1, "any"])
def test_brier_score_computer(event_of_interest):
    times_, loss = brier_score(
        y_train,
        y_test,
        y_pred,
        times,
        event_of_interest,
    )

    # Check that 'times_' hasn't been changed
    assert_array_equal(times, times_)

    loss_expected = np.array(
        [
            0.01921016,
            0.08987548,
            0.11693115,
            0.18832202,
            0.21346599,
            0.24300206,
            0.24217776,
            0.21987924,
            0.19987174,
            0.16301318,
            0.07628881,
            0.05829176,
            0.0663998,
            0.04524901,
            0.04553689,
            0.02250038,
            0.02259133,
        ]
    )

    assert_array_almost_equal(loss, loss_expected)

    ibs = integrated_brier_score(
        y_train,
        y_test,
        y_pred,
        times,
        event_of_interest,
    )

    ibs_expected = 0.1257316251344779

    assert abs(ibs - ibs_expected) < 1e-6


def test_brier_score_sampler():
    sampler = BrierScoreSampler(y_train, random_state=0)
    times_, y_binary, sample_weights = sampler.draw()

    assert times_.shape == (y_train["event"].shape[0], 1)

    y_binary_expected = np.array(
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        dtype=np.int32,
    )

    assert_array_equal(y_binary[:10], y_binary_expected)

    sample_weights_expected = np.array(
        [
            1.0892326,
            1.02340426,
            1.0141844,
            1.05622555,
            1.02340426,
            1.0892326,
            1.0141844,
            1.05622555,
            1.0892326,
            1.03891038,
        ]
    )

    assert_array_almost_equal(sample_weights[:10], sample_weights_expected)


def test_wrong_parameters():
    msg = "event_of_interest must be a strictly positive integer or 'any'"
    for event_of_interest in [-10, 0, "wrong_event"]:
        with pytest.raises(ValueError, match=msg):
            brier_score(
                y_train,
                y_test,
                y_pred,
                times,
                event_of_interest,
            )

    msg = "event_of_interest must be an instance of"
    for event_of_interest in [None, [1], (2, 3)]:
        with pytest.raises(TypeError, match=msg):
            brier_score(
                y_train,
                y_test,
                y_pred,
                times,
                event_of_interest,
            )


def _dict_to_pd(y):
    return pd.DataFrame(y)


def _dict_to_recarray(y):
    y_out = np.empty(
        shape=y["event"].shape[0],
        dtype=[("event", np.int32), ("duration", np.float64)],
    )
    y_out["event"] = y["event"]
    y_out["duration"] = y["duration"]
    return y_out


@pytest.mark.parametrize("format_func", [_dict_to_pd, _dict_to_recarray])
def test_inputs_format(format_func):
    _, loss = brier_score(
        format_func(y_train),
        format_func(y_test),
        y_pred,
        times,
        event_of_interest="any",
    )

    loss_expected = np.array(
        [
            0.01921016,
            0.08987548,
            0.11693115,
            0.18832202,
            0.21346599,
            0.24300206,
            0.24217776,
            0.21987924,
            0.19987174,
            0.16301318,
            0.07628881,
            0.05829176,
            0.0663998,
            0.04524901,
            0.04553689,
            0.02250038,
            0.02259133,
        ]
    )

    assert_array_almost_equal(loss, loss_expected)


def test_wrong_inputs():
    y_train_wrong = dict(
        wrong_name=y_train["event"],
        duration=y_train["duration"],
    )
    msg = (
        "y must be a record array, a pandas DataFrame, or a dict whose dtypes, "
        "keys or columns are 'event' and 'duration'."
    )
    with pytest.raises(ValueError, match=msg):
        brier_score(
            y_train_wrong,
            y_test,
            y_pred,
            times,
            event_of_interest="any",
        )

    msg = "'times' length (5) must be equal to y_pred.shape[1] (17)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        brier_score(
            y_train,
            y_test,
            y_pred,
            times[:5],
            event_of_interest="any",
        )
