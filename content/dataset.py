from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import bernoulli
from scipy.stats import norm


total_years = 10
total_days = total_years * 365

truck_model_names = ["RA", "C1", "C2", "RB", "C3"]

brand_quality = pd.DataFrame({
    "brand": ["Robusta", "Cheapz"],
    "assembly_quality": [0.95, 0.30],
})

trucks = pd.DataFrame(
    {
        "truck_model": truck_model_names,
        "brand": [
            "Robusta" if m.startswith("R") else "Cheapz"
            for m in truck_model_names
        ],
        "ux": [.2, .5, .7, .9, 1.0],
        "material_quality": [.95, .92, .85, .7, .65],
    }
).merge(brand_quality)



def sample_usage_weights(n_datapoints, rng):
    rates_1 = norm.rvs(.5, .08, size=n_datapoints, random_state=rng)
    rates_2 = norm.rvs(.8, .05, size=n_datapoints, random_state=rng)
    usage_mixture_idxs = rng.choice(2, size=n_datapoints, p=[1/3, 2/3])    
    return np.where(usage_mixture_idxs, rates_1, rates_2).clip(0, 1)


def sample_driver_truck_pairs(n_datapoints, random_seed=None):
    rng = np.random.RandomState(random_seed)
    df = pd.DataFrame(
        {
            "driver_skill": rng.uniform(low=0.2, high=1.0, size=n_datapoints).round(decimals=1),
            "truck_model": rng.choice(truck_model_names, size=n_datapoints),
            "usage_rate": sample_usage_weights(n_datapoints, rng).round(decimals=2),
        }
    )
    return df


def sample_driver_truck_pairs_with_metadata(n_datapoints, random_seed):
    return (
        sample_driver_truck_pairs(
            n_datapoints, random_seed=random_seed
        )
        .reset_index()
        .merge(trucks, on="truck_model")
        # Sort by original index to avoid introducing an ordering
        # of the dataset based on the truck_model column.
        .sort_values("index")
        .drop("index", axis="columns")
    )


def weibull_hazard(t, k=1., s=1., t_shift=100, base_rate=1e2):
    # See: https://en.wikipedia.org/wiki/Weibull_distribution
    # t_shift is a trick to avoid avoid negative powers at t=0 when k < 1.
    # t_shift could be interpreted at the operation time at the factory for
    # quality assurance checks for instance.
    t = t + t_shift
    return base_rate * (k / s) * (t / s) ** (k - 1.)


def assembly_hazards(df, t):
    baseline = weibull_hazard(t, k=0.003)
    s = (df["usage_rate"] * (1 - df["assembly_quality"])).to_numpy()
    return s.reshape(-1, 1) * baseline.reshape(1, -1)


def operational_hazards(df, t):
    # Weibull hazards with k = 1 is just a constant over time:
    baseline = weibull_hazard(t, k=1, s=8e3)
    s = (
        ((1 - df["driver_skill"]) * (1 - df["ux"]) + .001) * df["usage_rate"]
    ).to_numpy()
    return s.reshape(-1, 1) * baseline.reshape(1, -1)


def fatigue_hazards(df, t):
    return np.vstack([
        0.5 * weibull_hazard(t, k=6 * material_quality, s=4e3) * usage_rate
        for material_quality, usage_rate in zip(df["material_quality"], df["usage_rate"])
    ])


def sample_events_by_type(hazards, random_state=None):
    rng = check_random_state(random_state)
    outcomes = bernoulli.rvs(hazards, random_state=rng)
    any_event_mask = np.any(outcomes, axis=1)
    duration = np.full(outcomes.shape[0], fill_value=total_days)
    occurrence_rows, occurrence_cols = np.where(outcomes)
    # Some individuals might have more than one event occurrence,
    # we only keep the first one.
    # ex: trials = [[0, 0, 1, 0, 1]] -> duration = 2
    _, first_occurrence_idxs = np.unique(occurrence_rows, return_index=True)
    duration[any_event_mask] = occurrence_cols[first_occurrence_idxs]
    jitter = rng.rand(duration.shape[0])
    return pd.DataFrame(dict(event=any_event_mask, duration=duration + jitter))


def first_event(event_frames, event_ids, random_seed=None):
    rng = check_random_state(random_seed)
    event = np.zeros(event_frames[0].shape[0], dtype=np.int32)
    max_duration = np.max([ef["duration"].max() for ef in event_frames])
    duration = np.full_like(event_frames[0]["duration"], fill_value=max_duration)
    
    out = pd.DataFrame(
        {
            "event": event,
            "duration": duration,
        }
    )
    for event_id, ef in zip(event_ids, event_frames):
        mask = ef["event"] & (ef["duration"] < out["duration"])
        out.loc[mask, "event"] = event_id
        out.loc[mask, "duration"] = ef.loc[mask, "duration"]
    return out

    
def uniform_censoring(occurrences, censoring_weight=0.5, offset=0, random_state=None):
    n_datapoints = occurrences.shape[0]
    rng = check_random_state(random_state)
    max_duration = occurrences["duration"].max()
    censoring_durations = rng.randint(
        low=offset, high=max_duration, size=n_datapoints
    )
    # reduce censoring randomly by setting durations back to the max,
    # effectively ensuring that a fraction of the datapoints will not
    # be censured.
    disabled_censoring_mask = rng.rand(n_datapoints) > censoring_weight
    censoring_durations[disabled_censoring_mask] = max_duration
    out = occurrences.copy()
    censor_mask = occurrences["duration"] > censoring_durations
    out.loc[censor_mask, "event"] = 0
    out.loc[censor_mask, "duration"] = censoring_durations[censor_mask]
    return out


def sample_competing_events(
    data,
    uniform_censoring_weight=1.0,
    max_observation_duration=2000,
    random_seed=None,
):
    rng = check_random_state(random_seed)
    t = np.linspace(0, total_days, total_days)
    hazard_funcs = [
        assembly_hazards,
        operational_hazards,
        fatigue_hazards,
    ]
    hazard_shape = (data.shape[0], total_days)
    all_hazards = np.zeros(hazard_shape)
    for hazard_func in hazard_funcs:
        hazards = hazard_func(data, t).reshape(hazard_shape)
        all_hazards += hazards
    return all_hazards  # shape = (n_observations, total_days)


def generate_hazards():
    print(f"Generating theoretical hazards...")
    truck_failure_10k = sample_driver_truck_pairs_with_metadata(10_000, random_seed=0)
    truck_failure_10k_all_hazards = sample_competing_events(truck_failure_10k, random_seed=0)
    return truck_failure_10k_all_hazards
