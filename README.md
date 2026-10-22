# Explain the Model

**Public app:** Not deployed yet. Deploy `streamlit_app.py` on Streamlit Community Cloud, then put the generated app URL here.

Direct section links use the deployed URL with `?section=linear`, `?section=forest`, or `?section=boosting`.

A Streamlit interactive explainer for three supplied regression models. Visitors can choose any section, dataset, and example person; the app uses precomputed model outputs rather than retraining models.

The **Push & Pull** section also lets visitors edit numeric inputs and choose categorical values to see the linear prediction change. The forest and boosting sections remain tied to the supplied example people.

## Viewer takeaways

- **Push & Pull:** My viewer should walk away knowing that a linear model starts at a reference prediction and adds each feature's signed push to arrive at one person's prediction.
- **Wisdom of the Forest:** My viewer should walk away knowing that a random forest averages the guesses of many different trees, each learned from different samples and feature choices.
- **One Correction at a Time:** My viewer should walk away knowing that gradient boosting adds small corrections to earlier predictions and that validation error can guide when to stop.

## Run locally

Install the dependencies and run the Streamlit entry point from the repository root:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run streamlit_app.py
```

Streamlit opens the app at <http://localhost:8501>. The app loads `data/datasets.json` and the matching JSON file in `data/outputs/`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from this repository and select the `main` branch.
3. Set **Main file path** to `streamlit_app.py` (the file is at the repository root).
4. Deploy. Streamlit installs the packages listed in `requirements.txt`.

The existing `index.html`, `app.js`, and `style.css` are the separate static-site version; Streamlit runs `streamlit_app.py` instead.

## Data and model outputs

- The three NHANES CSV datasets and their data dictionary are in [`data/`](./data/README.md).
- Supplied model outputs are in [`data/outputs/`](./data/outputs/). They are from the [Explain the Model activity starter](https://github.com/snsie/regression-activity/tree/main/starter/outputs).
- The required Example person 1 checks are recorded in [`check.md`](./check.md).
- Source dataset citations and the model-output schema are documented in the data README and the activity's [output schema](https://github.com/snsie/regression-activity/blob/main/starter/outputs/SCHEMA.md).

All model figures shown by the app come from the supplied output JSON. In particular, the boosting corrections already include the learning rate; the app adds each supplied correction once.
