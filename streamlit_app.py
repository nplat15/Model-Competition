from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATASET_IDS = ("glucose", "sbp", "fev1")
SECTIONS = {
    "linear": {
        "label": "Push & Pull",
        "model": "Linear regression",
        "question": "Which features push one person's prediction up or down?",
        "takeaway": "Start at a reference prediction. Add each feature's signed push to get this person's prediction.",
    },
    "forest": {
        "label": "Wisdom of the Forest",
        "model": "Random forest",
        "question": "How do many different trees combine into one prediction?",
        "takeaway": "The forest's prediction is the average of 100 different tree guesses.",
    },
    "boosting": {
        "label": "One Correction at a Time",
        "model": "Gradient boosting",
        "question": "How does each new tree improve the model, and when should it stop?",
        "takeaway": "Start at the training average. Each small tree corrects part of the errors left by earlier rounds.",
    },
}

st.set_page_config(
    page_title="Explain the Model",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_datasets() -> dict[str, dict[str, Any]]:
    path = ROOT / "data" / "datasets.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_bundle(dataset_id: str) -> dict[str, Any]:
    path = ROOT / "data" / "outputs" / f"{dataset_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model output file was not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def rounded_number(value: float, digits: int = 1) -> float:
    return float(number(value, digits).replace(",", ""))


def signed(value: float) -> str:
    return f"{value:+,.1f}"


def grouped_pushes(
    bundle: dict[str, Any], values: dict[str, float]
) -> tuple[dict[str, float], float]:
    linear = bundle["linear"]
    feature_pushes = {
        feature["name"]: linear["coefficients"][feature["name"]]
        * (values[feature["name"]] - linear["reference"][feature["name"]])
        for feature in bundle["features"]
    }
    pushes: dict[str, float] = {}
    grouped_columns: set[str] = set()
    for group in bundle["categorical_groups"]:
        columns = [level["column"] for level in group["levels"] if level["column"]]
        pushes[group["name"]] = sum(feature_pushes[column] for column in columns)
        grouped_columns.update(columns)
    for feature in bundle["features"]:
        name = feature["name"]
        if name not in grouped_columns:
            pushes[name] = feature_pushes[name]
    prediction = linear["baseline"] + sum(feature_pushes.values())
    return pushes, prediction


def sample_tree_indices(values: list[float], count: int) -> list[int]:
    indices = {0}
    if count > 1:
        included = values[:count]
        indices.update(
            {
                count - 1,
                included.index(min(included)),
                included.index(max(included)),
            }
        )
        for index in range(count):
            if len(indices) >= min(7, count):
                break
            indices.add(round(index * (count - 1) / max(1, min(7, count) - 1)))
        for index in range(count):
            if len(indices) >= min(7, count):
                break
            indices.add(index)
    return sorted(indices)[: min(7, count)]


def initialize_linear_widgets(
    bundle: dict[str, Any], person: dict[str, Any], dataset_id: str
) -> tuple[dict[str, str], dict[str, str]]:
    context = (dataset_id, person["id"])
    feature_keys: dict[str, str] = {}
    group_keys: dict[str, str] = {}
    if st.session_state.get("_linear_context") != context:
        st.session_state["_linear_context"] = context
        for feature in bundle["features"]:
            key = f"linear_value_{feature['name']}"
            feature_keys[feature["name"]] = key
            st.session_state[key] = float(person["values"][feature["name"]])
        for group in bundle["categorical_groups"]:
            key = f"linear_group_{group['name']}"
            group_keys[group["name"]] = key
            selected = next(
                (
                    level["label"]
                    for level in group["levels"]
                    if level["column"] and person["values"][level["column"]] == 1
                ),
                group["reference"],
            )
            st.session_state[key] = selected
    else:
        feature_keys = {
            feature["name"]: f"linear_value_{feature['name']}"
            for feature in bundle["features"]
        }
        group_keys = {
            group["name"]: f"linear_group_{group['name']}"
            for group in bundle["categorical_groups"]
        }
    return feature_keys, group_keys


def render_linear(bundle: dict[str, Any], person: dict[str, Any], dataset_id: str) -> None:
    st.subheader("Try a different profile")
    st.write("Edit a value or choose a category to see how the linear prediction changes.")
    feature_keys, group_keys = initialize_linear_widgets(bundle, person, dataset_id)
    category_by_column = {
        level["column"]: group
        for group in bundle["categorical_groups"]
        for level in group["levels"]
        if level["column"]
    }

    values = {name: float(value) for name, value in person["values"].items()}
    seen_groups: set[str] = set()
    input_columns = st.columns(2)
    for index, feature in enumerate(bundle["features"]):
        group = category_by_column.get(feature["name"])
        with input_columns[index % 2]:
            if group:
                if group["name"] in seen_groups:
                    continue
                seen_groups.add(group["name"])
                selected = st.selectbox(
                    group["label"],
                    options=[level["label"] for level in group["levels"]],
                    key=group_keys[group["name"]],
                    help=f"Coding reference group: {group['reference']}.",
                )
                for level in group["levels"]:
                    if level["column"]:
                        values[level["column"]] = float(level["label"] == selected)
                st.caption(f"Reference group: {group['reference']}. One category push.")
                continue

            stats = bundle["feature_stats"][feature["name"]]
            step = 1.0 if feature["name"] == "age" else 0.01 if feature["units"] == "ratio" else 0.1
            values[feature["name"]] = st.number_input(
                f"{feature['label']} ({feature['units']})",
                step=step,
                key=feature_keys[feature["name"]],
                help=(
                    f"Training range: {number(stats['min'])}–"
                    f"{number(stats['max'])} {feature['units']}."
                ),
            )
            st.caption(
                f"Training range: {number(stats['min'])}–{number(stats['max'])} "
                f"{feature['units']}"
            )

    edited = any(
        values[name] != float(person["values"][name])
        for name in values
    )
    if edited:
        pushes, prediction = grouped_pushes(bundle, values)
    else:
        supplied = bundle["linear"]["people"][person["id"]]
        pushes = supplied["grouped_pushes"]
        prediction = supplied["prediction"]

    rows: list[dict[str, str]] = []
    group_by_column = {
        level["column"]: group
        for group in bundle["categorical_groups"]
        for level in group["levels"]
        if level["column"]
    }
    shown_groups: set[str] = set()
    for feature in bundle["features"]:
        group = group_by_column.get(feature["name"])
        if group:
            if group["name"] in shown_groups:
                continue
            shown_groups.add(group["name"])
            selected = next(
                (
                    level["label"]
                    for level in group["levels"]
                    if level["column"] and values[level["column"]] == 1
                ),
                group["reference"],
            )
            rows.append(
                {
                    "Feature": group["label"],
                    "Person's value": f"{selected} · one category push",
                    f"Push ({bundle['units']})": signed(pushes[group["name"]]),
                }
            )
        else:
            rows.append(
                {
                    "Feature": feature["label"],
                    "Person's value": f"{number(values[feature['name']])} {feature['units']}",
                    f"Push ({bundle['units']})": signed(pushes[feature["name"]]),
                }
            )

    baseline_shown = rounded_number(bundle["linear"]["baseline"])
    prediction_shown = rounded_number(prediction)
    total_push = prediction_shown - baseline_shown
    st.subheader("Build the prediction")
    equation = st.columns([1, 0.22, 1, 0.22, 1])
    equation[0].metric("Starting point", f"{number(baseline_shown)} {bundle['units']}")
    equation[1].markdown("**+**")
    equation[2].metric("Sum of pushes", f"{signed(total_push)} {bundle['units']}")
    equation[3].markdown("**=**")
    equation[4].metric("Prediction", f"{number(prediction_shown)} {bundle['units']}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "The displayed total uses the rounded starting point and prediction so the "
        "equation adds up exactly; individual pushes are rounded separately."
    )

    category_references = "; ".join(
        f"{group['label']}: {group['reference']}"
        for group in bundle["categorical_groups"]
    )
    if category_references:
        st.info(
            f"Category coding references: {category_references}. Category pushes are "
            "measured from the training-average mix, so someone in a reference group "
            "can still have a small push."
        )
    coefficient_feature = next(
        feature for feature in bundle["features"] if not feature["binary"]
    )
    coefficient_units = {
        "%": "percentage point",
        "years": "year",
        "cm": "centimeter",
        "ratio": "ratio unit",
    }.get(coefficient_feature["units"], coefficient_feature["units"])
    coefficient = bundle["linear"]["coefficients"][coefficient_feature["name"]]
    st.markdown(
        f"**Coefficient meaning:** holding other features fixed, a 1 "
        f"{coefficient_units} increase in {coefficient_feature['label']} is associated "
        f"with {signed(coefficient)} {bundle['units']} in the prediction. "
        f"This coefficient's units are {bundle['units']} per {coefficient_units}."
    )
    st.caption(
        "One person's pushes describe this prediction; they are not a ranking of which "
        "features are most important overall."
    )
    if edited:
        st.warning(
            "This edits the linear model only. Values outside the training range are "
            "extrapolations."
        )


def render_forest(
    bundle: dict[str, Any], person: dict[str, Any], dataset_id: str
) -> None:
    result = bundle["forest"]["people"][person["id"]]
    predictions = result["tree_predictions"]
    if len(predictions) != 100:
        raise ValueError(f"Expected 100 forest trees for {dataset_id}, got {len(predictions)}.")

    st.subheader("One person, many tree guesses")
    st.metric("Average of all 100 trees", f"{number(result['prediction'])} {bundle['units']}")
    count = st.slider(
        "Trees to include, in the supplied order",
        min_value=1,
        max_value=100,
        value=100,
        key="forest_tree_count",
    )
    included = predictions[:count]
    running_average = sum(included) / count
    current_range = max(included) - min(included)
    status = "Full forest" if count == 100 else "Partial forest"
    st.markdown(f"**{status}: {count} of 100 trees**")
    columns = st.columns(2)
    columns[0].metric(
        f"{status} running average",
        f"{number(running_average)} {bundle['units']}",
    )
    columns[1].metric(
        "Tree disagreement (range)",
        f"{number(current_range)} {bundle['units']}",
    )
    st.caption("The forest prediction is the mean of the individual tree guesses.")

    chart_data = pd.DataFrame(
        {
            "Tree": range(1, count + 1),
            "Individual tree guess": included,
            "Running average": [running_average] * count,
        }
    )
    st.line_chart(
        chart_data,
        x="Tree",
        y=["Individual tree guess", "Running average"],
        height=300,
    )
    st.caption(f"Each point on the first line is one tree's guess, in {bundle['units']}.")
    samples = sample_tree_indices(predictions, count)
    st.markdown("**Several individual tree guesses**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Tree": index + 1,
                    f"Individual guess ({bundle['units']})": number(predictions[index]),
                }
                for index in samples
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "Trees disagree with one another. This spread is not a confidence interval or "
        "prediction interval; it is not a calibrated range for the true outcome."
    )

    example_four = next(candidate for candidate in bundle["people"] if candidate["id"] == "P4")
    p4_predictions = bundle["forest"]["people"]["P4"]["tree_predictions"]
    p4_range = max(p4_predictions) - min(p4_predictions)
    st.markdown("**Explore a person whose trees disagree**")
    st.write(
        f"Example person 4 ({example_four['description']}) has a "
        f"{number(p4_range)} {bundle['units']} range across all 100 trees."
    )
    st.button(
        "Explore Example person 4",
        disabled=person["id"] == "P4",
        on_click=lambda: st.session_state.update(person_choice="P4"),
    )
    forest = bundle["forest"]
    unique_rows = forest["bootstrap_unique_training_rows"][0]
    st.markdown("### Why do the trees differ?")
    st.write(
        f"Each tree learns from a bootstrap sample: a random draw of training people "
        f"with replacement. The first tree used {unique_rows:,} distinct people out "
        f"of {forest['n_training_rows']:,}."
    )
    st.write(
        f"At each split, the tree considers a random subset of "
        f"{forest['features_tried_per_split']} of the {forest['n_features']} available "
        "features. Different samples and feature choices make different trees."
    )


def render_boosting(
    bundle: dict[str, Any], person: dict[str, Any]
) -> None:
    boosting = bundle["boosting"]
    result = boosting["people"][person["id"]]
    st.subheader("Round 0: the training average")
    st.metric("Starting prediction for everyone", f"{number(boosting['init'])} {bundle['units']}")

    round_number = st.slider(
        "Explore rounds and choose a stopping point",
        min_value=0,
        max_value=500,
        value=0,
        key="boosting_round",
    )
    staged = result["staged"]
    previous_raw = staged[max(0, round_number - 1)]
    updated_raw = staged[round_number]
    previous = rounded_number(previous_raw)
    updated = rounded_number(updated_raw)
    correction = updated - previous
    columns = st.columns([1, 0.18, 1, 0.18, 1])
    columns[0].metric("Previous prediction", f"{number(previous)} {bundle['units']}")
    columns[1].markdown("**+**")
    columns[2].metric(
        "Tree's correction",
        "—" if round_number == 0 else f"{signed(correction)} {bundle['units']}",
    )
    columns[3].markdown("**=**")
    columns[4].metric("Updated prediction", f"{number(updated)} {bundle['units']}")
    if round_number == 0:
        st.info("At round 0, no trees have been added yet. The prediction is the training average.")
    else:
        st.caption(
            f"Tree {round_number} was fitted to the remaining training residuals: the "
            "differences between training outcomes and predictions so far. The supplied "
            "correction includes the learning rate. Values are rounded to one decimal, "
            "so the displayed correction exactly matches the displayed change."
        )

    errors = boosting["rmse"]
    if len(errors["train"]) != 501 or len(errors["validation"]) != 501:
        raise ValueError("Expected training and validation RMSE for rounds 0 through 500.")
    best_round = boosting["best_validation_round"]
    best_rmse = boosting["best_validation_rmse"]
    chart_data = pd.DataFrame(
        {
            "Round": range(501),
            "Training RMSE": errors["train"],
            "Validation RMSE": errors["validation"],
        }
    )
    st.markdown("### Training and validation error by round")
    st.caption(
        f"X-axis: boosting round. Y-axis: RMSE ({bundle['units']}). "
        "Lower RMSE means smaller prediction errors."
    )
    st.line_chart(
        chart_data,
        x="Round",
        y=["Training RMSE", "Validation RMSE"],
        height=350,
    )
    metrics = st.columns(3)
    metrics[0].metric(
        f"Training RMSE · round {round_number}",
        f"{number(errors['train'][round_number])} {bundle['units']}",
    )
    metrics[1].metric(
        "Validation RMSE at your round",
        f"{number(errors['validation'][round_number])} {bundle['units']}",
    )
    metrics[2].metric(
        f"Lowest validation RMSE: round {best_round}",
        f"{number(best_rmse)} {bundle['units']}",
    )
    if round_number == best_round:
        st.success(
            f"Round {round_number} is your selected stopping point and also has the "
            f"lowest validation RMSE ({number(best_rmse)} {bundle['units']}). "
            "This is a validation-based choice, not a test result."
        )
    else:
        st.info(
            f"Your selected stopping point is round {round_number} "
            f"(validation RMSE {number(errors['validation'][round_number])} "
            f"{bundle['units']}). The lowest validation RMSE is "
            f"{number(best_rmse)} {bundle['units']} at round {best_round}. "
            "Compare these validation results to make a validation-based choice; "
            "neither is a test result."
        )
    st.markdown(
        "**Why stop?** Training error may keep falling while validation error starts "
        "to rise: later rounds can fit quirks of the training people. Use validation "
        "data to choose a stopping round; it is not a test result."
    )


def main() -> None:
    st.title("Explain the Model")
    st.write(
        "A hands-on guide to three regression models. Choose a section, dataset, and "
        "example person to follow how the prediction is made."
    )
    try:
        datasets = load_datasets()
        section_key = st.query_params.get("section", "linear")
        if section_key not in SECTIONS:
            section_key = "linear"
        if "section_selector" not in st.session_state:
            st.session_state["section_selector"] = section_key
        st.sidebar.header("Choose an example")
        st.sidebar.selectbox(
            "Section",
            options=list(SECTIONS),
            format_func=lambda key: SECTIONS[key]["label"],
            key="section_selector",
        )
        st.query_params["section"] = st.session_state["section_selector"]
        dataset_id = st.sidebar.selectbox(
            "Dataset",
            options=DATASET_IDS,
            format_func=lambda key: datasets[key]["target_label"],
            key="dataset_choice",
        )
        bundle = load_bundle(dataset_id)
        people = bundle["people"]
        person_ids = [person["id"] for person in people]
        st.sidebar.selectbox(
            "Example person",
            options=person_ids,
            format_func=lambda person_id: next(
                f"{person['label']} · {person['description']}"
                for person in people
                if person["id"] == person_id
            ),
            key="person_choice",
        )
        person = next(
            candidate
            for candidate in people
            if candidate["id"] == st.session_state["person_choice"]
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
        st.error(f"The app could not load the model data: {error}")
        st.stop()

    section = SECTIONS[st.session_state["section_selector"]]
    st.caption(section["model"].upper())
    st.header(section["label"])
    st.write(section["question"])
    st.info(section["takeaway"])
    st.caption(f"{person['label']} · {person['description']} · {bundle['name']}")

    if st.session_state["section_selector"] == "linear":
        render_linear(bundle, person, dataset_id)
    elif st.session_state["section_selector"] == "forest":
        render_forest(bundle, person, dataset_id)
    else:
        render_boosting(bundle, person)

    st.divider()
    st.caption(
        "Model numbers use the supplied precomputed outputs. These are associations, "
        "not evidence that a feature causes the outcome."
    )
    st.caption(
        f"Data source: [{bundle['source']}]({bundle['url']})"
    )


if __name__ == "__main__":
    main()
