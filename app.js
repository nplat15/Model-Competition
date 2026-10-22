const DATASETS_URL = "./data/datasets.json";
const OUTPUTS_URL = (key) => `./data/outputs/${key}.json`;
const SECTION_IDS = ["linear", "forest", "boosting"];
const SECTION_COPY = {
  linear: {
    title: "Push & Pull",
    model: "LINEAR REGRESSION",
    question: "Which features push a prediction up or down, and by how much?",
    takeaway: "Start at a reference prediction. Add each feature's signed push to get this person's prediction.",
  },
  forest: {
    title: "Wisdom of the Forest",
    model: "RANDOM FOREST",
    question: "How do many different trees combine into one prediction?",
    takeaway: "Each tree makes a guess. The forest's prediction is the average of all 100 guesses.",
  },
  boosting: {
    title: "One Correction at a Time",
    model: "GRADIENT BOOSTING",
    question: "How does each new tree improve the model, and when should it stop?",
    takeaway: "Start at the training average. Each small tree corrects part of the errors left by earlier rounds.",
  },
};

const state = {
  datasets: null,
  key: "glucose",
  personId: "P1",
  section: "linear",
  linearEdit: null,
  bundles: new Map(),
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const content = $("#app-content");
const datasetSelect = $("#dataset-select");
const personSelect = $("#person-select");
const personDescription = $("#person-description");
let renderVersion = 0;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function number(value, digits = 1) {
  const rounded = Number(value.toFixed(digits));
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(rounded);
}

function signed(value) {
  return `${value >= 0 ? "+" : "−"}${number(Math.abs(value))}`;
}

function loadJson(url) {
  return fetch(url).then((response) => {
    if (!response.ok) {
      throw new Error(`Could not load ${url} (${response.status} ${response.statusText}).`);
    }
    return response.json();
  });
}

async function getBundle(key = state.key) {
  if (!state.bundles.has(key)) {
    state.bundles.set(key, loadJson(OUTPUTS_URL(key)));
  }
  return state.bundles.get(key);
}

function personFor(bundle) {
  const person = bundle.people.find((candidate) => candidate.id === state.personId);
  if (!person) throw new Error(`Example person ${state.personId} is missing from ${state.key}.json.`);
  return person;
}

function updatePersonDescription(bundle) {
  const person = personFor(bundle);
  const edit = state.linearEdit;
  const edited = state.section === "linear"
    && edit?.key === state.key
    && edit.personId === person.id
    && isEditedLinearProfile(bundle, person, edit.values);
  personDescription.textContent = edited
    ? `${person.description} · linear profile edited below`
    : `${person.description} · supplied validation example`;
}

function updatePersonSelector(bundle) {
  const currentPerson = bundle.people.find((person) => person.id === state.personId);
  if (!currentPerson) state.personId = bundle.people[0].id;
  personSelect.innerHTML = bundle.people.map((person) => `
    <option value="${escapeHtml(person.id)}" ${person.id === state.personId ? "selected" : ""}>
      ${escapeHtml(person.label)} · ${escapeHtml(person.description)}
    </option>
  `).join("");
  personSelect.disabled = false;
  updatePersonDescription(bundle);
}

function setActiveNavigation() {
  for (const link of $$(".model-nav a")) {
    if (link.dataset.section === state.section) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  }
}

function pageHeader(bundle) {
  const copy = SECTION_COPY[state.section];
  return `
    <section class="section-heading">
      <div>
        <p class="section-kicker">${copy.model}</p>
        <h2>${copy.title}</h2>
        <p class="section-question">${copy.question}</p>
      </div>
      <div class="learning-note">
        <span class="learning-note-icon" aria-hidden="true">↳</span>
        <div><span class="micro-label">THE IDEA</span><p>${copy.takeaway}</p></div>
      </div>
    </section>
  `;
}

function metric(label, value, unit, extraClass = "") {
  return `
    <div class="metric ${extraClass}">
      <span class="micro-label">${label}</span>
      <span class="metric-value">${value}</span>
      <span class="metric-unit">${unit}</span>
    </div>
  `;
}

function personCard(person, bundle, edited = false) {
  return `
    <div class="person-chip">
      <span class="person-chip-icon" aria-hidden="true">↗</span>
      <span><strong>${escapeHtml(person.label)}${edited ? " · edited" : ""}</strong><span class="person-chip-detail">${edited ? "Customized profile · based on " : ""}${escapeHtml(person.description)} · ${escapeHtml(bundle.name)}</span></span>
    </div>
  `;
}

function unitPhrase(units) {
  if (units === "years") return "year";
  if (units === "%") return "percentage point";
  if (units === "cm") return "centimeter";
  if (units === "ratio") return "ratio unit";
  return units;
}

function currentLinearValues(bundle, person) {
  if (
    !state.linearEdit
    || state.linearEdit.key !== state.key
    || state.linearEdit.personId !== person.id
  ) {
    state.linearEdit = {
      key: state.key,
      personId: person.id,
      values: { ...person.values },
    };
  }
  return state.linearEdit.values;
}

function isEditedLinearProfile(bundle, person, values) {
  return bundle.features.some((feature) => values[feature.name] !== person.values[feature.name]);
}

function explainLinear(bundle, person, values) {
  if (!isEditedLinearProfile(bundle, person, values)) {
    const supplied = bundle.linear.people[person.id];
    if (!supplied) throw new Error(`Linear prediction for ${person.id} is missing from ${state.key}.json.`);
    return supplied;
  }
  const pushes = {};
  for (const feature of bundle.features) {
    pushes[feature.name] = bundle.linear.coefficients[feature.name]
      * (values[feature.name] - bundle.linear.reference[feature.name]);
  }
  const groupedPushes = {};
  const groupedColumns = new Set();
  for (const group of bundle.categorical_groups) {
    const columns = group.levels.filter((level) => level.column).map((level) => level.column);
    groupedPushes[group.name] = columns.reduce((sum, column) => sum + pushes[column], 0);
    for (const column of columns) groupedColumns.add(column);
  }
  for (const feature of bundle.features) {
    if (!groupedColumns.has(feature.name)) groupedPushes[feature.name] = pushes[feature.name];
  }
  const prediction = bundle.linear.baseline
    + Object.values(pushes).reduce((sum, push) => sum + push, 0);
  return { pushes, grouped_pushes: groupedPushes, prediction };
}

function pushRows(bundle, person, values, pushes) {
  const groupByColumn = new Map();
  for (const group of bundle.categorical_groups) {
    for (const level of group.levels) {
      if (level.column) groupByColumn.set(level.column, group);
    }
  }
  const usedGroups = new Set();
  const rows = [];
  for (const feature of bundle.features) {
    const group = groupByColumn.get(feature.name);
    if (group) {
      if (usedGroups.has(group.name)) continue;
      usedGroups.add(group.name);
      const level = group.levels.find((candidate) =>
        candidate.column && values[candidate.column] === 1,
      );
      const category = level ? level.label : group.reference;
      rows.push({
        key: group.name,
        label: group.label,
        detail: `${category} · one category push`,
        value: pushes[group.name],
      });
      continue;
    }
    rows.push({
      key: feature.name,
      label: feature.label,
      detail: `${number(values[feature.name], 1)} ${feature.units}`,
      value: pushes[feature.name],
    });
  }
  return rows;
}

function renderLinearEditor(bundle, values) {
  const groupByColumn = new Map();
  for (const group of bundle.categorical_groups) {
    for (const level of group.levels) {
      if (level.column) groupByColumn.set(level.column, group);
    }
  }
  const addedGroups = new Set();
  const controls = [];
  for (const feature of bundle.features) {
    const group = groupByColumn.get(feature.name);
    if (group) {
      if (addedGroups.has(group.name)) continue;
      addedGroups.add(group.name);
      const selected = group.levels.find((level) =>
        level.column && values[level.column] === 1,
      )?.label ?? group.reference;
      controls.push(`
        <label class="linear-input-field">
          <span class="linear-input-label">${escapeHtml(group.label)}</span>
          <select data-linear-group="${escapeHtml(group.name)}" aria-label="${escapeHtml(group.label)}">
            ${group.levels.map((level) => `
              <option value="${escapeHtml(level.label)}" ${level.label === selected ? "selected" : ""}>
                ${escapeHtml(level.label)}
              </option>
            `).join("")}
          </select>
          <span class="linear-range-note">Reference group: ${escapeHtml(group.reference)}</span>
        </label>
      `);
      continue;
    }
    const stats = bundle.feature_stats[feature.name];
    if (!stats) throw new Error(`Training range for ${feature.name} is missing from ${state.key}.json.`);
    const step = feature.name === "age" ? "1" : feature.units === "ratio" ? "0.01" : "0.1";
    controls.push(`
      <label class="linear-input-field">
        <span class="linear-input-label">${escapeHtml(feature.label)}</span>
        <span class="linear-number-control">
          <input type="number" step="${step}" value="${escapeHtml(values[feature.name])}" data-linear-feature="${escapeHtml(feature.name)}" aria-label="${escapeHtml(feature.label)}" />
          <span>${escapeHtml(feature.units)}</span>
        </span>
        <span class="linear-range-note">Training range: ${number(stats.min)}–${number(stats.max)} ${escapeHtml(feature.units)}</span>
      </label>
    `);
  }
  return `
    <section class="linear-editor" aria-label="Edit example profile for linear regression">
      <div class="linear-editor-heading">
        <div>
          <span class="micro-label">TRY A DIFFERENT PROFILE</span>
          <p>Change a value or choose a category to see how this model's prediction responds.</p>
        </div>
        <button class="reset-button" id="reset-linear-profile" type="button">Reset to example</button>
      </div>
      <div class="linear-input-grid">${controls.join("")}</div>
      <p class="linear-edit-status" id="linear-edit-status" role="status">This edits the linear model only. Values outside the training range are extrapolations.</p>
    </section>
  `;
}

function renderLinearResult(bundle, person, values, result) {
  const rows = pushRows(bundle, person, values, result.grouped_pushes);
  const displayedBaseline = Number(bundle.linear.baseline.toFixed(1));
  const displayedPrediction = Number(result.prediction.toFixed(1));
  const totalPush = displayedPrediction - displayedBaseline;
  const scale = Math.max(1, ...rows.map((row) => Math.abs(row.value)));
  const positive = rows.filter((row) => row.value > 0).sort((a, b) => b.value - a.value)[0];
  const categoryNote = bundle.categorical_groups.length
    ? `<div class="category-reference-note"><strong>About category pushes:</strong> ${bundle.categorical_groups.map((group) => `${escapeHtml(group.label)} uses ${escapeHtml(group.reference)} as its reference group`).join("; ")}. These are coding references; category pushes are measured from the training-average mix, so even someone in a reference group can have a small push.</div>`
    : "";
  const shownRows = rows.map((row) => `
    <li class="push-row">
      <div class="push-description">
        <span class="push-name">${escapeHtml(row.label)}</span>
        <span class="push-detail">${escapeHtml(row.detail)}</span>
      </div>
      <span class="push-meter"><span class="push-meter-fill ${row.value < 0 ? "negative" : "positive"}" style="width:${Math.max(1, Math.abs(row.value) / scale * 50)}%"></span></span>
      <span class="push-value ${row.value < 0 ? "negative-text" : "positive-text"}">${signed(row.value)}</span>
    </li>
  `).join("");
  const biggestPush = positive
    ? `<div class="insight-line"><span class="insight-spark" aria-hidden="true">↗</span><span>For this person, <strong>${escapeHtml(positive.label)}</strong> is the biggest upward push at <strong>${signed(positive.value)} ${escapeHtml(bundle.units)}</strong>.</span></div>`
    : "";
  const operation = totalPush < 0 ? "−" : "+";

  return `
    <div class="equation">
      ${metric("STARTING POINT", number(bundle.linear.baseline), bundle.units)}
      <span class="equation-operator" aria-hidden="true">${operation}</span>
      <div class="equation-summary"><span class="micro-label">SUM OF PUSHES</span><span class="metric-value">${number(Math.abs(totalPush))}</span><span class="metric-unit">${escapeHtml(bundle.units)} · ${rows.length} feature pushes</span></div>
      <span class="equation-operator" aria-hidden="true">=</span>
      ${metric("PREDICTION", number(result.prediction), bundle.units, "prediction-metric")}
    </div>
    <div class="push-list-heading"><span>WHAT MOVED THE GUESS?</span><span>PUSH IN ${escapeHtml(bundle.units)}</span></div>
    <ul class="push-list">${shownRows}</ul>
    <div class="equation-footnote">The starting point is the prediction at training-average feature values, not the intercept. The displayed total makes the rounded equation add up exactly; individual pushes are rounded separately, and the model calculation uses full precision.</div>
    ${categoryNote}
    ${biggestPush}
  `;
}

function renderLinear(bundle, person) {
  const values = currentLinearValues(bundle, person);
  const result = explainLinear(bundle, person, values);
  const numericFeature = bundle.features.find((feature) => !feature.binary);
  const coefficient = bundle.linear.coefficients[numericFeature.name];
  const coefficientUnit = unitPhrase(numericFeature.units);
  const edited = isEditedLinearProfile(bundle, person, values);

  return `
    ${pageHeader(bundle)}
    <section class="explainer-card" aria-label="Linear regression prediction breakdown">
      <div class="card-topline"><span class="card-index">A PREDICTION, UNPACKED</span><div id="linear-profile">${personCard(person, bundle, edited)}</div></div>
      ${renderLinearEditor(bundle, values)}
      <div id="linear-result">${renderLinearResult(bundle, person, values, result)}</div>
    </section>
    <section class="concept-grid" aria-label="How to read the linear regression breakdown">
      <article class="concept-card coefficient-card">
        <span class="concept-number">01 / COEFFICIENT</span>
        <h3>Prediction change per unit</h3>
        <p><strong>${escapeHtml(numericFeature.label)}: ${signed(coefficient)} ${escapeHtml(bundle.units)} per ${escapeHtml(coefficientUnit)}</strong></p>
        <p>A ${escapeHtml(coefficientUnit)} difference in ${escapeHtml(numericFeature.label.toLowerCase())} is associated with this difference in predicted ${escapeHtml(bundle.target_label.toLowerCase())}, holding the other features fixed.</p>
      </div>
      <article class="concept-card">
        <span class="concept-number">02 / PUSH</span>
        <h3>This person's difference</h3>
        <p>The push combines the coefficient with how this person's feature value differs from its training average. A coefficient is not the same as a person's push, and a large push here is not a measure of overall feature importance.</p>
      </article>
      <article class="concept-card">
        <span class="concept-number">03 / INTERPRETATION</span>
        <h3>Pattern, not cause</h3>
        <p>These are associations in survey data. A feature's push describes how this fitted model's prediction differs; it doesn't show what would happen if a person's value changed.</p>
      </article>
    </section>
  `;
}

function updateLinearResult(bundle, person) {
  const values = currentLinearValues(bundle, person);
  const result = explainLinear(bundle, person, values);
  const edited = isEditedLinearProfile(bundle, person, values);
  $("#linear-profile").innerHTML = personCard(person, bundle, edited);
  updatePersonDescription(bundle);
  $("#linear-result").innerHTML = renderLinearResult(bundle, person, values, result);
}

function bindLinearEditor(bundle, person) {
  const inputs = $$("[data-linear-feature]", content);
  for (const input of inputs) {
    input.addEventListener("input", (event) => {
      const control = event.currentTarget;
      const value = control.valueAsNumber;
      const status = $("#linear-edit-status");
      if (!Number.isFinite(value) || !control.validity.valid) {
        control.setAttribute("aria-invalid", "true");
        status.textContent = "Enter a valid value to update the prediction.";
        return;
      }
      control.removeAttribute("aria-invalid");
      currentLinearValues(bundle, person)[control.dataset.linearFeature] = value;
      status.textContent = "Updated. This edits the linear model only.";
      updateLinearResult(bundle, person);
    });
  }
  for (const select of $$("[data-linear-group]", content)) {
    select.addEventListener("change", (event) => {
      const group = bundle.categorical_groups.find((candidate) =>
        candidate.name === event.currentTarget.dataset.linearGroup,
      );
      if (!group) throw new Error(`Category group ${event.currentTarget.dataset.linearGroup} is missing.`);
      const values = currentLinearValues(bundle, person);
      for (const level of group.levels) {
        if (level.column) values[level.column] = level.label === event.currentTarget.value ? 1 : 0;
      }
      $("#linear-edit-status").textContent = "Updated. This edits the linear model only.";
      updateLinearResult(bundle, person);
    });
  }
  $("#reset-linear-profile").addEventListener("click", () => {
    state.linearEdit = null;
    render();
  });
}

function forestSamples(values, count) {
  const sampleCount = Math.min(7, count);
  const indexes = new Set();
  indexes.add(0);
  if (count > 1) {
    const included = values.slice(0, count);
    indexes.add(count - 1);
    indexes.add(included.indexOf(Math.min(...included)));
    indexes.add(included.indexOf(Math.max(...included)));
    for (let i = 0; i < sampleCount && indexes.size < sampleCount; i += 1) {
      indexes.add(Math.round(i * (count - 1) / (sampleCount - 1)));
    }
    for (let i = 1; indexes.size < sampleCount && i < count - 1; i += 1) indexes.add(i);
  }
  return [...indexes]
    .sort((a, b) => a - b)
    .slice(0, sampleCount)
    .map((index) => ({ index, value: values[index] }));
}

function forestDisagreementCallout(bundle) {
  const person = bundle.people.find((candidate) => candidate.id === "P4");
  if (!person) throw new Error(`Example person 4 is missing from ${state.key}.json.`);
  const predictions = bundle.forest.people[person.id].tree_predictions;
  const range = Math.max(...predictions) - Math.min(...predictions);
  return `
    <div class="disagreement-example">
      <div>
        <span class="micro-label">SEE TREE DISAGREEMENT</span>
        <p><strong>Example person 4 · ${escapeHtml(person.description)}</strong> was chosen because this person's trees disagree the most. Across all 100 trees, their guesses span <strong>${number(range)} ${escapeHtml(bundle.units)}</strong>.</p>
      </div>
      <button class="disagreement-button" id="show-disagreement-person" type="button" ${state.personId === person.id ? "disabled" : ""}>
        ${state.personId === person.id ? "Viewing this example" : "Explore this example"}
      </button>
    </div>
  `;
}

function renderForest(bundle, person) {
  const result = bundle.forest.people[person.id];
  if (!result) throw new Error(`Forest prediction for ${person.id} is missing from ${state.key}.json.`);
  const values = result.tree_predictions;
  if (values.length !== 100) throw new Error(`Expected 100 supplied forest trees for ${state.key} ${person.id}; found ${values.length}.`);
  const initialSamples = forestSamples(values, 100);
  const initialAverage = result.prediction;
  const spread = Math.max(...values) - Math.min(...values);
  const firstTreeUniqueRows = bundle.forest.bootstrap_unique_training_rows[0];
  const sampleCount = bundle.forest.features_tried_per_split;
  const totalFeatures = bundle.forest.n_features;
  return `
    ${pageHeader(bundle)}
    <section class="explainer-card" aria-label="Random forest prediction">
      <div class="card-topline"><span class="card-index">ONE PERSON · MANY GUESSES</span>${personCard(person, bundle)}</div>
      <div class="forest-result">
        <div class="forest-result-label"><span class="micro-label">AVERAGE OF ALL 100 TREES</span><span class="forest-result-value">${number(initialAverage)} <small>${escapeHtml(bundle.units)}</small></span></div>
        <div class="forest-explainer">The forest's prediction is the mean of the trees' individual guesses.</div>
      </div>
      <div class="forest-control">
        <div class="slider-heading">
          <label for="tree-count"><span class="micro-label">HOW MANY TREES TO INCLUDE?</span><strong><span id="tree-count-value">100</span> <span id="forest-type-label">of 100 trees</span></strong></label>
          <span class="slider-readout" id="forest-average-label">Full forest</span>
        </div>
        <input id="tree-count" class="range-slider" type="range" min="1" max="100" value="100" step="1" aria-label="Number of trees to include from the supplied order" aria-valuetext="100 trees in the supplied order" />
        <p class="forest-order-note">The running average uses trees 1 through the selected number, in the supplied order.</p>
        <div class="range-limits"><span>1 tree</span><span>100 trees</span></div>
      </div>
      <div class="forest-live-metrics">
        <div><span class="micro-label" id="forest-average-heading">FULL FOREST AVERAGE</span><strong id="forest-average">${number(initialAverage)} ${escapeHtml(bundle.units)}</strong></div>
        <div><span class="micro-label">TREE DISAGREEMENT</span><strong id="forest-spread">${number(spread)} ${escapeHtml(bundle.units)} range</strong></div>
      </div>
      <div class="chart-heading">
        <div><span class="micro-label">A FEW OF THE INDIVIDUAL GUESSES</span><p>Different trees can land at different predictions.</p></div>
        <span class="chart-key"><i class="key-dot" aria-hidden="true"></i> tree guess <i class="key-line" aria-hidden="true"></i> current average</span>
      </div>
      <div id="forest-chart" class="forest-chart">
        ${forestChart(values, initialAverage, initialSamples, bundle.units)}
      </div>
      <div id="forest-samples" class="tree-samples" aria-live="polite">
        ${renderForestSampleRows(initialSamples, bundle.units)}
      </div>
      <p class="disagreement-note"><strong>Tree disagreement is not a confidence or prediction interval.</strong> The spread shows how much these trees differ from each other; it is not a calibrated range for where the true outcome falls.</p>
      ${forestDisagreementCallout(bundle)}
    </section>
    <section class="concept-grid forest-concepts" aria-label="Why random forest trees differ">
      <article class="concept-card">
        <span class="concept-number">01 / DIFFERENT PEOPLE</span>
        <h3>Each tree gets a different draw</h3>
        <p>Each tree learns from a bootstrap sample: a random draw from the training people, with replacement. The first tree used ${number(firstTreeUniqueRows, 0)} distinct people out of ${number(bundle.forest.n_training_rows, 0)} training people.</p>
      </article>
      <article class="concept-card">
        <span class="concept-number">02 / DIFFERENT QUESTIONS</span>
        <h3>Each split gets a feature shortlist</h3>
        <p>At each question, the tree considers a random subset of ${sampleCount} of the ${totalFeatures} available features. Different samples and choices create different trees.</p>
      </article>
      <article class="concept-card">
        <span class="concept-number">03 / AVERAGE THE GUESSES</span>
        <h3>Quirks can partly cancel</h3>
        <p>Averaging many different trees can make the final prediction steadier than one tree's guess. Steadier does not guarantee more accurate.</p>
      </article>
    </section>
  `;
}

function renderForestSampleRows(samples, units) {
  return samples.map(({ index, value }) => `
    <div class="tree-sample"><span>TREE ${index + 1}</span><strong>${number(value)} <small>${escapeHtml(units)}</small></strong></div>
  `).join("");
}

function forestChart(values, average, samples, units, count = values.length) {
  const width = 760;
  const height = 154;
  const left = 42;
  const right = 728;
  const top = 20;
  const bottom = 112;
  const included = values.slice(0, count);
  const low = Math.min(...included);
  const high = Math.max(...included);
  const domain = high === low ? 1 : high - low;
  const x = (value) => left + (value - low) / domain * (right - left);
  const positions = samples.map((sample, i) => ({
    ...sample,
    y: top + (samples.length <= 1 ? 0 : i / (samples.length - 1)) * (bottom - top),
  }));
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(samples.length)} individual tree predictions selected from the first ${count} trees; their running average is ${number(average)} ${escapeHtml(units)}">
      <line x1="${left}" y1="${top - 8}" x2="${left}" y2="${bottom + 8}" class="chart-axis" />
      <line x1="${left}" y1="${bottom + 8}" x2="${right}" y2="${bottom + 8}" class="chart-axis" />
      <line x1="${x(average)}" y1="${top - 9}" x2="${x(average)}" y2="${bottom + 8}" class="forest-average-line" />
      ${positions.map(({ index, value, y }) => `
        <line x1="${left}" y1="${y}" x2="${x(value)}" y2="${y}" class="forest-stem" />
        <circle cx="${x(value)}" cy="${y}" r="5.5" class="forest-dot"><title>Tree ${index + 1}: ${number(value)} ${escapeHtml(units)}</title></circle>
        <text x="4" y="${y + 4}" class="chart-small-label">T${index + 1}</text>
      `).join("")}
      <text x="${left}" y="${height - 6}" text-anchor="start" class="chart-tick">${number(low)} ${escapeHtml(units)}</text>
      <text x="${right}" y="${height - 6}" text-anchor="end" class="chart-tick">${number(high)} ${escapeHtml(units)}</text>
      <text x="${Math.min(right - 6, Math.max(left + 6, x(average)))}" y="12" text-anchor="${x(average) > (left + right) / 2 ? "end" : "start"}" class="chart-average-label">AVERAGE</text>
    </svg>
  `;
}

function renderBoosting(bundle, person) {
  const result = bundle.boosting.people[person.id];
  if (!result) throw new Error(`Boosting predictions for ${person.id} are missing from ${state.key}.json.`);
  const errors = bundle.boosting.rmse;
  if (errors.train.length !== 501 || errors.validation.length !== 501) {
    throw new Error(`Expected round 0 through 500 boosting errors for ${state.key}.`);
  }
  const selectedRound = 0;
  const bestRound = bundle.boosting.best_validation_round;
  return `
    ${pageHeader(bundle)}
    <section class="explainer-card" aria-label="Gradient boosting rounds and validation error">
      <div class="card-topline"><span class="card-index">500 SMALL STEPS · ONE PREDICTION</span>${personCard(person, bundle)}</div>
      <div class="boost-start">
        <span class="micro-label">ROUND 0 · BEFORE ANY TREES</span>
        <span class="boost-start-value">${number(bundle.boosting.init)} <small>${escapeHtml(bundle.units)}</small></span>
        <span class="boost-start-note">The training average is the starting prediction for everyone.</span>
      </div>
      <div class="round-control">
        <div class="slider-heading">
          <label for="boost-round"><span class="micro-label">EXPLORE ROUNDS &amp; CHOOSE WHEN TO STOP</span><strong>Round <span id="boost-round-value">0</span><span class="round-of"> / 500</span></strong></label>
          <span class="best-round-badge">LOWEST VALIDATION RMSE <strong>ROUND ${bestRound}</strong></span>
        </div>
        <input id="boost-round" class="range-slider boost-slider" type="range" min="0" max="500" value="${selectedRound}" step="1" aria-label="Choose a boosting round and stopping point" aria-valuetext="Round 0" />
        <div class="range-limits"><span>Round 0 · training average</span><span>Round 500</span></div>
      </div>
      <div class="correction-flow" aria-live="polite">
        <div class="correction-step"><span class="micro-label">PREVIOUS PREDICTION</span><strong id="previous-prediction">${number(result.staged[0])} <small>${escapeHtml(bundle.units)}</small></strong></div>
        <span class="flow-symbol" aria-hidden="true">+</span>
        <div class="correction-step correction-step-highlight"><span class="micro-label">TREE'S CORRECTION</span><strong id="round-correction">—</strong></div>
        <span class="flow-symbol" aria-hidden="true">=</span>
        <div class="correction-step correction-step-result"><span class="micro-label">UPDATED PREDICTION</span><strong id="updated-prediction">${number(result.staged[0])} <small>${escapeHtml(bundle.units)}</small></strong></div>
      </div>
      <p id="round-explanation" class="round-explanation">At round 0, no trees have been added yet. The prediction is the training average.</p>
      <div class="error-chart-header">
        <div><span class="micro-label">WHEN SHOULD THE MODEL STOP?</span><p>Choose a round based on how well it predicts held-out validation people.</p></div>
        <span class="chart-key"><i class="key-line train-key" aria-hidden="true"></i> training <i class="key-line validation-key" aria-hidden="true"></i> validation</span>
      </div>
      <div id="boost-chart" class="boost-chart">
        ${boostingChart(errors, selectedRound, bestRound, bundle.units)}
      </div>
      <div class="error-readout" aria-live="polite">
        <div class="error-readout-chosen"><span class="micro-label">YOUR CHOSEN ROUND <strong id="chosen-round">0</strong></span><span>Training RMSE <strong id="chosen-train-rmse">${number(errors.train[0])} ${escapeHtml(bundle.units)}</strong></span><span>Validation RMSE <strong id="chosen-validation-rmse">${number(errors.validation[0])} ${escapeHtml(bundle.units)}</strong></span></div>
        <div class="error-readout-best"><span class="micro-label">LOWEST VALIDATION RMSE</span><strong>${number(bundle.boosting.best_validation_rmse)} ${escapeHtml(bundle.units)}</strong><span>at round ${bestRound}</span></div>
      </div>
      <p id="round-comparison" class="round-comparison" aria-live="polite">${boostingRoundComparison(bundle.boosting, selectedRound, bundle.units)}</p>
      <div class="boosting-notes">
        <p><strong>Each new tree learns from the training residuals</strong>—the differences between the training outcomes and the predictions so far. It adds only a small part of the pattern it finds.</p>
        <p><strong>Validation error helps choose when to stop.</strong> Training error can keep falling while validation error starts to rise: later rounds may fit quirks of the training people. The validation minimum is a choice made using validation data, not a test result.</p>
      </div>
    </section>
  `;
}

function boostingRoundComparison(boosting, round, units) {
  const bestRound = boosting.best_validation_round;
  const chosenRmse = number(boosting.rmse.validation[round]);
  const bestRmse = number(boosting.best_validation_rmse);
  if (round === bestRound) {
    return `Round ${round} is your selected stopping point and also has the lowest validation RMSE (${bestRmse} ${units}). This is a validation-based choice, not a test result.`;
  }
  return `Your selected stopping point is round ${round} (validation RMSE ${chosenRmse} ${units}). The lowest validation RMSE is ${bestRmse} ${units} at round ${bestRound}. Compare these validation results to make a validation-based choice; neither is a test result.`;
}

function boostingChart(errors, selectedRound, bestRound, units) {
  const width = 760;
  const height = 274;
  const left = 62;
  const right = 728;
  const top = 24;
  const bottom = 218;
  const allValues = [...errors.train, ...errors.validation];
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const pad = (max - min) * 0.06 || 1;
  const yMin = Math.max(0, min - pad);
  const yMax = max + pad;
  const x = (round) => left + round / 500 * (right - left);
  const y = (value) => bottom - (value - yMin) / (yMax - yMin) * (bottom - top);
  const pathFor = (values) => values.map((value, index) =>
    `${index === 0 ? "M" : "L"}${x(index).toFixed(2)},${y(value).toFixed(2)}`,
  ).join(" ");
  const ticks = Array.from({ length: 5 }, (_, index) => yMin + (yMax - yMin) * index / 4);
  const selectedX = x(selectedRound);
  const selectedYTrain = y(errors.train[selectedRound]);
  const selectedYValidation = y(errors.validation[selectedRound]);
  const bestX = x(bestRound);
  const bestY = y(errors.validation[bestRound]);
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Training and validation RMSE by boosting round in ${escapeHtml(units)}. The lowest validation error is at round ${bestRound}; the selected round is ${selectedRound}.">
      ${ticks.map((tick) => `
        <line x1="${left}" y1="${y(tick)}" x2="${right}" y2="${y(tick)}" class="chart-gridline" />
        <text x="${left - 9}" y="${y(tick) + 4}" text-anchor="end" class="chart-tick">${number(tick)}</text>
      `).join("")}
      <line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" class="chart-axis" />
      <line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" class="chart-axis" />
      <path d="${pathFor(errors.train)}" class="error-path train-path" />
      <path d="${pathFor(errors.validation)}" class="error-path validation-path" />
      <line x1="${bestX}" y1="${top}" x2="${bestX}" y2="${bottom}" class="best-round-line" />
      <circle cx="${bestX}" cy="${bestY}" r="5" class="best-round-point"><title>Lowest validation RMSE: round ${bestRound}, ${number(errors.validation[bestRound])} ${escapeHtml(units)}</title></circle>
      <line x1="${selectedX}" y1="${top}" x2="${selectedX}" y2="${bottom}" class="selected-round-line" />
      <circle cx="${selectedX}" cy="${selectedYTrain}" r="4" class="selected-train-point"><title>Round ${selectedRound} training RMSE: ${number(errors.train[selectedRound])} ${escapeHtml(units)}</title></circle>
      <circle cx="${selectedX}" cy="${selectedYValidation}" r="4" class="selected-validation-point"><title>Round ${selectedRound} validation RMSE: ${number(errors.validation[selectedRound])} ${escapeHtml(units)}</title></circle>
      <text x="${left}" y="${bottom + 19}" text-anchor="middle" class="chart-tick">0</text>
      <text x="${x(100)}" y="${bottom + 19}" text-anchor="middle" class="chart-tick">100</text>
      <text x="${x(200)}" y="${bottom + 19}" text-anchor="middle" class="chart-tick">200</text>
      <text x="${x(300)}" y="${bottom + 19}" text-anchor="middle" class="chart-tick">300</text>
      <text x="${x(400)}" y="${bottom + 19}" text-anchor="middle" class="chart-tick">400</text>
      <text x="${right}" y="${bottom + 19}" text-anchor="middle" class="chart-tick">500</text>
      <text x="${(left + right) / 2}" y="${height - 6}" text-anchor="middle" class="chart-axis-label">BOOSTING ROUND</text>
      <text x="15" y="${(top + bottom) / 2}" text-anchor="middle" transform="rotate(-90 15 ${(top + bottom) / 2})" class="chart-axis-label">RMSE (${escapeHtml(units)})</text>
      <text x="${Math.min(right - 6, Math.max(left + 6, bestX))}" y="${Math.max(top + 12, bestY - 10)}" text-anchor="${bestX > (left + right) / 2 ? "end" : "start"}" class="best-round-label">LOWEST VALIDATION · ${bestRound}</text>
    </svg>
  `;
}

function updateForest(bundle, person, count) {
  const values = bundle.forest.people[person.id].tree_predictions;
  const samples = forestSamples(values, count);
  const total = values.slice(0, count).reduce((sum, value) => sum + value, 0);
  const average = total / count;
  const range = Math.max(...values.slice(0, count)) - Math.min(...values.slice(0, count));
  $("#tree-count-value").textContent = count;
  $("#forest-type-label").textContent = count === 100 ? "of 100 trees" : "of 100 trees · partial forest";
  $("#forest-average-label").textContent = count === 100 ? "Full forest" : "Partial forest";
  $("#forest-average-heading").textContent = count === 100 ? "FULL FOREST AVERAGE" : "PARTIAL FOREST AVERAGE";
  $("#forest-average").textContent = `${number(average)} ${bundle.units}`;
  $("#forest-spread").textContent = `${number(range)} ${bundle.units} range`;
  $("#tree-count").setAttribute("aria-valuetext", `${count} ${count === 100 ? "trees" : "trees in a partial forest"}, in the supplied order`);
  $("#forest-chart").innerHTML = forestChart(values, average, samples, bundle.units, count);
  $("#forest-samples").innerHTML = renderForestSampleRows(samples, bundle.units);
}

function updateBoosting(bundle, person, round) {
  const result = bundle.boosting.people[person.id];
  const staged = result.staged;
  const unit = bundle.units;
  const previous = Number(staged[Math.max(0, round - 1)].toFixed(1));
  const updated = Number(staged[round].toFixed(1));
  const correction = round === 0 ? null : Number((updated - previous).toFixed(1));
  $("#boost-round-value").textContent = round;
  $("#boost-round").setAttribute("aria-valuetext", `Round ${round}`);
  $("#previous-prediction").innerHTML = `${number(previous)} <small>${escapeHtml(unit)}</small>`;
  $("#round-correction").innerHTML = correction === null
    ? "—"
    : `${signed(correction)} <small>${escapeHtml(unit)}</small>`;
  $("#updated-prediction").innerHTML = `${number(updated)} <small>${escapeHtml(unit)}</small>`;
  $("#round-explanation").textContent = round === 0
    ? "At round 0, no trees have been added yet. The prediction is the training average."
    : `Tree ${round} was fitted to the remaining training residuals. Its correction includes the learning rate. Values are rounded to one decimal, and the displayed correction matches the change from ${number(previous)} to ${number(updated)} ${unit}.`;
  $("#chosen-round").textContent = round;
  $("#chosen-train-rmse").textContent = `${number(bundle.boosting.rmse.train[round])} ${unit}`;
  $("#chosen-validation-rmse").textContent = `${number(bundle.boosting.rmse.validation[round])} ${unit}`;
  $("#round-comparison").textContent = boostingRoundComparison(bundle.boosting, round, unit);
  $("#boost-chart").innerHTML = boostingChart(
    bundle.boosting.rmse,
    round,
    bundle.boosting.best_validation_round,
    unit,
  );
}

async function render() {
  const version = ++renderVersion;
  const key = state.key;
  const section = state.section;
  content.setAttribute("aria-busy", "true");
  setActiveNavigation();
  try {
    const bundle = await getBundle(key);
    if (version !== renderVersion) return;
    updatePersonSelector(bundle);
    const person = personFor(bundle);
    const renderer = {
      linear: renderLinear,
      forest: renderForest,
      boosting: renderBoosting,
    }[section];
    content.innerHTML = renderer(bundle, person);
    content.setAttribute("aria-busy", "false");
    if (section === "forest") {
      $("#tree-count").addEventListener("input", (event) => {
        updateForest(bundle, person, Number(event.currentTarget.value));
      });
      $("#show-disagreement-person").addEventListener("click", () => {
        state.personId = "P4";
        state.linearEdit = null;
        render();
      });
    }
    if (section === "linear") bindLinearEditor(bundle, person);
    if (section === "boosting") {
      $("#boost-round").addEventListener("input", (event) => {
        updateBoosting(bundle, person, Number(event.currentTarget.value));
      });
    }
  } catch (error) {
    if (version !== renderVersion) return;
    content.setAttribute("aria-busy", "false");
    content.innerHTML = `
      <section class="error-state" role="alert">
        <span class="error-icon" aria-hidden="true">!</span>
        <div><h2>This section could not load</h2><p>${escapeHtml(error.message)}</p><p>Check that the app is served over HTTP and that the files in <code>data/</code> are available.</p></div>
      </section>
    `;
    console.error(error);
  }
}

async function initialize() {
  try {
    state.datasets = await loadJson(DATASETS_URL);
    const entries = Object.values(state.datasets);
    datasetSelect.innerHTML = entries.map((dataset) => `
      <option value="${escapeHtml(dataset.id)}" ${dataset.id === state.key ? "selected" : ""}>
        ${escapeHtml(dataset.target_label)} · ${escapeHtml(dataset.units)}
      </option>
    `).join("");
    datasetSelect.disabled = false;
    const requestedSection = location.hash.slice(1);
    if (SECTION_IDS.includes(requestedSection)) state.section = requestedSection;
    datasetSelect.addEventListener("change", async (event) => {
      state.key = event.currentTarget.value;
      state.personId = "P1";
      state.linearEdit = null;
      await render();
    });
    personSelect.addEventListener("change", async (event) => {
      state.personId = event.currentTarget.value;
      state.linearEdit = null;
      await render();
    });
    for (const link of $$(".model-nav a")) {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        const section = link.dataset.section;
        if (state.section === section) return;
        state.section = section;
        history.pushState(null, "", `#${section}`);
        render();
      });
    }
    window.addEventListener("hashchange", () => {
      const section = location.hash.slice(1);
      if (SECTION_IDS.includes(section) && section !== state.section) {
        state.section = section;
        render();
      }
    });
    await render();
  } catch (error) {
    content.setAttribute("aria-busy", "false");
    content.innerHTML = `
      <section class="error-state" role="alert">
        <span class="error-icon" aria-hidden="true">!</span>
        <div><h2>The app could not start</h2><p>${escapeHtml(error.message)}</p><p>Check that <code>data/datasets.json</code> is available and serve the app over HTTP.</p></div>
      </section>
    `;
    console.error(error);
  }
}

initialize();
