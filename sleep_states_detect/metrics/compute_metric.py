# https://github.com/yukiu00/kaggle-child-mind-institute-detect-sleep-states/blob/main/src/utils/metrics.py

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm


class ParticipantVisibleError(Exception):
    pass


# Set some placeholders for global parameters
# series_id_column_name = "series_id"
# time_column_name = "step"
# event_column_name = "event"
# score_column_name = "score"
# use_scoring_intervals = False
# tolerances = {
#     "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
#     "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
# }


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, List[float]] = None,
    series_id_column_name: str = "series_id",
    time_column_name: str = "step",
    event_column_name: str = "event",
    score_column_name: str = "score",
    use_scoring_intervals: bool = False,
    verbose: bool = False,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    if tolerances in None:
        tolerances = {
            "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
            "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
        }

    # Validate metric parameters
    assert len(tolerances) > 0, "Events must have defined tolerances."
    assert set(tolerances.keys()) == set(solution[event_column_name]).difference(
        {"start", "end"}
    ), (
        f"Solution column {event_column_name} must contain the same events "
        "as defined in tolerances."
    )
    assert pd.api.types.is_numeric_dtype(
        solution[time_column_name]
    ), f"Solution column {time_column_name} must be of numeric type."

    # Validate submission format
    for column_name in [
        series_id_column_name,
        time_column_name,
        event_column_name,
        score_column_name,
    ]:
        if column_name not in submission.columns:
            raise ParticipantVisibleError(
                f"Submission must have column '{column_name}'."
            )

    if not pd.api.types.is_numeric_dtype(submission[time_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{time_column_name}' must be of numeric type."
        )
    if not pd.api.types.is_numeric_dtype(submission[score_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{score_column_name}' must be of numeric type."
        )

    # Set these globally to avoid passing around a bunch of arguments
    globals()["series_id_column_name"] = series_id_column_name
    globals()["time_column_name"] = time_column_name
    globals()["event_column_name"] = event_column_name
    globals()["score_column_name"] = score_column_name
    globals()["use_scoring_intervals"] = use_scoring_intervals

    return event_detection_ap(
        solution,
        submission,
        tolerances,
        series_id_column_name,
        time_column_name,
        event_column_name,
        score_column_name,
        use_scoring_intervals,
        verbose=verbose,
    )


def find_nearest(xs: np.ndarray, value):
    """
    Find the index of the closest value to x in the array xs.
    """
    idx = np.searchsorted(xs, value, side="left")
    best_idx = None
    best_error = float("inf")
    best_diff = float("inf")

    range_min = max(0, idx - 1)
    range_max = min(len(xs), idx + 2)
    for check_idx in range(
        range_min, range_max
    ):  # Check the exact, one before, and one after
        error = abs(xs[check_idx] - value)
        if error < best_error:
            best_error = error
            best_idx = check_idx
            best_diff = xs[check_idx] - value

    return best_idx, best_error, best_diff


def find_nearest_time_idx(sorted_gt_times, det_time, excluded_indices: set):
    """
    search index of gt_times closest to det_time.

    assumes gt_times is sorted in ascending order.
    """
    # e.g. if gt_times = [0, 1, 2, 3, 4, 5] and det_time = 2.5, then idx = 3
    sorted_gt_times = np.asarray(sorted_gt_times)
    available_indices = np.asarray(
        sorted(set(range(len(sorted_gt_times))) - excluded_indices), dtype=int
    )
    sorted_gt_times = sorted_gt_times[available_indices]
    idx, error, diff = find_nearest(sorted_gt_times, det_time)
    best_idx = available_indices[idx] if idx is not None else None

    return best_idx, error, diff


def match_detections(
    tolerance: float,
    ground_truths: pd.DataFrame,
    detections: pd.DataFrame,
    time_column_name: str,
    event_column_name: str,
    score_column_name: str,
) -> pd.DataFrame:
    detections_sorted = detections.sort_values(
        score_column_name, ascending=False
    ).dropna()
    is_matched = np.full_like(detections_sorted[event_column_name], False, dtype=bool)
    diffs = np.full_like(
        detections_sorted[event_column_name], float("inf"), dtype=float
    )
    ground_truths_times = ground_truths.sort_values(time_column_name)[
        time_column_name
    ].to_list()
    matched_gt_indices: set[int] = set()

    for i, det in enumerate(detections_sorted.itertuples(index=False)):
        det_time = getattr(det, time_column_name)

        best_idx, best_error, best_diff = find_nearest_time_idx(
            ground_truths_times, det_time, matched_gt_indices
        )

        if (best_idx is not None) and (best_error < tolerance):
            is_matched[i] = True
            diffs[i] = best_diff
            matched_gt_indices.add(best_idx)

    detections_sorted["matched"] = is_matched
    detections_sorted["diff"] = diffs
    return detections_sorted


def precision_recall_curve(
    matches: np.ndarray, scores: np.ndarray, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matches) == 0:
        return [1], [0], []

    # Sort matches by decreasing confidence
    idxs = np.argsort(scores, kind="stable")[::-1]
    scores = scores[idxs]
    matches = matches[idxs]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    thresholds = scores[threshold_idxs]

    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = (
        tps / p
    )  # total number of ground truths might be different than total number of matches

    # Stop when full recall attained and reverse the outputs so recall is non-increasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    # Final precision is 1 and final recall is 0
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def event_detection_ap(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, List[float]],
    series_id_column_name: str,
    time_column_name: str,
    event_column_name: str,
    score_column_name: str,
    use_scoring_intervals: bool,
    progress_bar: bool = True,
    verbose: bool = True,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    # Ensure solution and submission are sorted properly
    solution = solution.sort_values([series_id_column_name, time_column_name])
    submission = submission.sort_values([series_id_column_name, time_column_name])

    # Extract scoring intervals.
    if use_scoring_intervals:
        raise NotImplementedError("Scoring intervals not implemented.")

    # Extract ground-truth events.
    ground_truths = solution.query("event not in ['start', 'end']").reset_index(
        drop=True
    )

    # Map each event class to its prevalence (needed for recall calculation)
    class_counts = ground_truths.value_counts(event_column_name).to_dict()

    # Create table for detections with a column indicating
    # a match to a ground-truth event
    detections = submission.assign(matched=False)

    # Remove detections outside of scoring intervals
    if use_scoring_intervals:
        raise NotImplementedError("Scoring intervals not implemented.")
    else:
        detections_filtered = detections

    # Create table of event-class x tolerance x series_id values
    aggregation_keys = pd.DataFrame(
        [
            (ev, tol, vid)
            for ev in tolerances.keys()
            for tol in tolerances[ev]
            for vid in ground_truths[series_id_column_name].unique()
        ],
        columns=[event_column_name, "tolerance", series_id_column_name],
    )

    # Create match evaluation groups: event-class x tolerance x series_id
    detections_grouped = aggregation_keys.merge(
        detections_filtered, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])
    ground_truths_grouped = aggregation_keys.merge(
        ground_truths, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])

    # Match detections to ground truth events by evaluation group
    pbars = aggregation_keys.itertuples(index=False)
    if progress_bar:
        pbars = tqdm(pbars, total=len(aggregation_keys), desc="Matching detections")
    detections_matched = []
    for key in pbars:
        dets = detections_grouped.get_group(key)
        gts = ground_truths_grouped.get_group(key)
        detections_matched.append(
            match_detections(
                dets["tolerance"].iloc[0],
                gts,
                dets,
                time_column_name,
                event_column_name,
                score_column_name,
            )
        )
    detections_matched = pd.concat(detections_matched)

    # Compute AP per event x tolerance group
    # event_classes = ground_truths[event_column_name].unique()
    ap_table = (
        detections_matched.query("event in @event_classes")
        .groupby([event_column_name, "tolerance"])
        .apply(
            lambda group: average_precision_score(
                group["matched"].to_numpy(),
                group[score_column_name].to_numpy(),
                class_counts[group[event_column_name].iat[0]],
            )
        )
        .reset_index()
        .pivot(index="tolerance", columns="event", values=0)
    )
    if verbose:
        display(ap_table)
    # Average over tolerances, then over event classes
    mean_ap = ap_table.mean().mean()

    return mean_ap, ap_table, detections_matched
