# Explain the Model

A static, interactive explainer for three supplied regression models. Visitors can choose any section, dataset, and example person; the app uses the precomputed model outputs rather than retraining models in the browser.

The **Push & Pull** section also lets visitors edit numeric inputs and choose categorical values to see the linear prediction change. The forest and boosting sections remain tied to the supplied example people.

## Run locally

Serve the repository root over HTTP, then open the local URL. For example:

```powershell
py -m http.server 8000
```

Open <http://localhost:8000>. The app loads `data/datasets.json` and the matching JSON file in `data/outputs/`. Opening `index.html` directly as a `file:` URL will not work because browsers restrict `fetch()` for local files.

## Publish

The app is plain HTML, CSS, and JavaScript and can be hosted from the repository root with GitHub Pages. In the repository's GitHub settings, enable Pages and select the `main` branch and `/ (root)` as the source.

## Data and model outputs

- The three NHANES CSV datasets and their data dictionary are in [`data/`](./data/README.md).
- Supplied model outputs are in [`data/outputs/`](./data/outputs/). They are from the [Explain the Model activity starter](https://github.com/snsie/regression-activity/tree/main/starter/outputs).
- The required Example person 1 checks are recorded in [`check.md`](./check.md).
- Source dataset citations and the model-output schema are documented in the data README and the activity's [output schema](https://github.com/snsie/regression-activity/blob/main/starter/outputs/SCHEMA.md).

All model figures shown by the app come from the supplied output JSON. In particular, the boosting corrections already include the learning rate; the app adds each supplied correction once.
