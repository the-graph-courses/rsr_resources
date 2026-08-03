"""Build index.html and the per-dataset detail pages for the full catalogue.

Row content (blurbs, outcomes, predictors) comes from build/meta.json, which was
exported from the hand-authored metadata in the fable5_d3n8 build. Row counts,
column counts and missingness percentages are recomputed from the shipped CSVs
every time, so the page can never drift from the data.

    python3 build/build_index.py
"""

import json
import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = json.load(open(os.path.join(HERE, "build", "meta.json")))

# Display order: logistic-friendly and LMIC datasets first, small/linear-only last.
ORDER = [
    "serbia_antibiotics", "workplace_protection", "sciatica_primary_care",
    "nepal_hypertension", "kenya_hypertension", "peru_child_anaemia",
    "bangladesh_growth_monitoring", "medstudent_qol", "patient_activation",
    "nhanes_adults", "us_births_2014",
    "ggt_arterial_stiffness", "cimt_rheumatoid", "hiv_lung_6mwt",
    "famuss_strength", "tibial_bone_strength", "nigeria_malaria_households",
    "ageing_qol_europe", "essens_adolescents",
]

# These are public data extracts, not datasets deposited alongside one research paper.
NON_PAPER_SOURCES = {"nhanes_adults", "us_births_2014"}

# Variables that are identifiers or otherwise not for modelling.
NOT_A_VARIABLE = {"id", "number", "subject_code", "survey_weight"}

# Variables the upstream curation left out of the suggested lists, mostly because
# they are incomplete. This catalogue keeps them and marks the missingness instead,
# so students can make the trade-off themselves.
EXTRA_PREDICTORS = {
    "patient_activation": ["disease_duration", "support_family", "support_friends",
                           "support_significant_other", "ethnicity_dutch", "care_allowance"],
    "medstudent_qol": ["whoqol_social", "dreem_academic_self", "dreem_social_self"],
    "nhanes_adults": ["general_health", "days_phys_health_bad", "days_ment_health_bad",
                      "pulse", "testosterone"],
    "hiv_lung_6mwt": ["fev1_fvc_ratio", "mmrc_dyspnoea", "cd4_count",
                      "on_antiretroviral", "viral_load_detectable"],
    "cimt_rheumatoid": ["hypertension_or_hyperchol", "prednisone", "apo_a", "apo_b"],
    "us_births_2014": ["father_age_yrs"],
    "nepal_hypertension": ["marital_status", "cigarettes_per_week"],
    "kenya_hypertension": ["monthly_income"],
    "peru_child_anaemia": ["multiple_birth", "father_education"],
}

# Outcomes a student has to build themselves, so they match no column in the CSV.
# Outcomes withdrawn after fitting them: each either explains almost nothing, is
# dominated by a continuous version already on offer, or produces no movement
# between the crude and adjusted models. Reason kept alongside so the decision is
# reviewable rather than folklore.
EXCLUDED_OUTCOMES = {
    "nhanes_adults": {
        "total_cholesterol": "adjusted R-squared of 0.02; the diabetes coefficient is "
                             "an artefact of unrecorded statin use",
        "obese": "a lossy recode of bmi, and two of its three significant coefficients "
                 "come from a 25-person stratum holding one obese participant",
        "hypertensive": "an exact recode of the blood-pressure variables, dominated by "
                        "systolic_bp, and the worst listwise-deletion loss in the file",
    },
    "peru_child_anaemia": {
        "low_birthweight": "one dominant predictor plus one reverse-causal one, "
                           "everything else null; outcome is maternal recall years later",
    },
    "bangladesh_growth_monitoring": {
        "stunted": "no predictor changes significance after adjustment and most move "
                   "under 3%; use height_for_age_z instead",
    },
}

DERIVED_OUTCOMES = {
    "patient_activation": [("logistic", "Low patient activation", "low_activation",
        'Create from <code>pam_level</code>: classify levels 1 and 2 as low activation.')],
    "peru_child_anaemia": [("logistic", "Low birth weight", "low_birthweight",
        'Create from <code>birth_weight</code>, which is already categorical: '
        'classify <code>low</code> as 1 and <code>normal</code>/<code>macrosomia</code> as 0.')],
    "bangladesh_growth_monitoring": [
        ("logistic", "Underweight", "underweight", 'Create from <code>weight_for_age_z</code>: '
         'classify values of −2 or below as underweight.'),
        ("logistic", "Wasting", "wasting", 'Create from <code>weight_for_height_z</code>: '
         'classify values of −2 or below as wasting.')],
}

# Extra outcomes beyond the upstream list.
EXTRA_OUTCOMES = {
    "bangladesh_growth_monitoring": [("logistic", "can_explain_chart"),
                                     ("logistic", "explains_chart_colours")],
}

OUTCOME_LABELS = {
    "hypertension": "Hypertension",
    "systolic_bp": "Systolic blood pressure",
    "diastolic_bp": "Diastolic blood pressure",
    "bmi": "Body mass index",
    "bp_awareness": "Hypertension awareness",
    "prior_htn_diagnosis": "Previous hypertension diagnosis",
    "anaemia": "Childhood anaemia",
    "stunting": "Childhood stunting",
    "haemoglobin_g_dl": "Haemoglobin concentration",
    "height_for_age_z": "Height-for-age Z-score",
    "received_gmp_card": "Receipt of a growth-monitoring card",
    "heard_of_gmp": "Awareness of growth monitoring",
    "stunted": "Childhood stunting",
    "weight_for_age_z": "Weight-for-age Z-score",
    "weight_for_height_z": "Weight-for-height Z-score",
    "can_explain_chart": "Understanding of the growth chart",
    "explains_chart_colours": "Understanding of growth-chart colours",
    "whoqol_psychological": "Psychological quality of life",
    "whoqol_physical": "Physical quality of life",
    "whoqol_environment": "Environmental quality of life",
    "bdi_depression": "Depression severity",
    "medical_school_qol": "Medical-school quality of life",
    "overall_qol": "Overall quality of life",
    "pam_score": "Patient activation",
    "hads_depression": "Depression severity",
    "sf12_physical": "Physical health",
    "sf12_mental": "Mental health",
    "systolic_bp": "Systolic blood pressure",
    "total_cholesterol": "Total cholesterol",
    "hdl_cholesterol": "HDL cholesterol",
    "diabetes": "Diabetes",
    "sleep_trouble": "Sleep trouble",
    "phys_active": "Physical activity",
    "hypertensive": "Hypertension",
    "obese": "Obesity",
    "birthweight_lb": "Birthweight",
    "low_birthweight": "Low birthweight",
    "premature": "Premature birth",
    "gestation_weeks": "Gestational age",
    "pwv_max": "Arterial stiffness",
    "fatty_liver": "Fatty liver",
    "log2_ggt": "Gamma-glutamyl transferase",
    "abi_max": "Ankle-brachial index",
    "cimt_total": "Carotid intima-media thickness",
    "carotid_plaque": "Carotid plaque",
    "crp": "C-reactive protein",
    "six_min_walk_m": "Six-minute walk distance",
    "sgrq_total": "Respiratory quality of life",
    "fev1_pct_predicted": "Forced expiratory volume",
    "dlco_pct_predicted": "Diffusing capacity",
    "nondom_arm_change_pct": "Strength change in the trained arm",
    "dom_arm_change_pct": "Strength change in the untrained arm",
    "bone_strength_index": "Trabecular bone strength",
    "polar_strength_strain_index": "Cortical bone strength",
    "adequate_knowledge": "Adequate antibiotic knowledge",
    "self_medication": "Antibiotic self-medication",
    "feels_protected": "Feeling protected at work",
    "disability_score": "Back-related disability",
    "mri_nerve_root_compression": "Nerve-root compression on MRI",
    "clinician_sciatica": "Clinician-diagnosed sciatica",
    "malaria_parasitaemia": "Asymptomatic malaria parasitaemia",
    "quality_of_life_score": "Quality of life",
    "depression_diagnosis": "Diagnosed depression",
    "active_5plus_days": "Physical activity on at least five days",
    "high_weekday_tv": "High weekday TV or movie streaming",
    "high_weekday_gaming": "High weekday electronic gaming",
}

CODEBOOK_OVERRIDES = {
    "id": ("Participant identifier", "identifier"),
    "smoking_status": ("Smoking status", "categorical"),
    "alcohol_g_per_week": ("Alcohol intake (g/week)", "continuous"),
}


def unwrap(x):
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x


def missingness(slug):
    """Percent missing per column, recomputed from the shipped CSV."""
    d = pd.read_csv(os.path.join(HERE, "data", f"{slug}.csv"))
    return d, (100 * d.isna().mean()).round(1)


def flag(var, miss):
    """Append a missingness warning to a variable name when it is worth knowing."""
    pct = miss.get(var)
    if pct is None or pct < 10:
        return ""
    shown = f"{float(pct):.1f}".rstrip("0").rstrip(".")
    return f' <span class="miss">({shown}% missing)</span>'


def outcome_missingness(var, miss):
    pct = float(miss.get(var, 0))
    if pct == 0:
        return "No missing"
    shown = f"{pct:.1f}".rstrip("0").rstrip(".")
    return f"{shown}% missing"


def display_number(value):
    """Compact, readable numbers for the generated codebooks."""
    value = float(value)
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def codebook_values(series, kind):
    observed = series.dropna()
    if kind == "identifier":
        return f"{observed.nunique():,} distinct values"
    if kind.startswith("binary"):
        ones = int((observed == 1).sum())
        zeros = int((observed == 0).sum())
        pct = 100 * ones / len(observed) if len(observed) else 0
        return f"1 = {ones:,} ({pct:.1f}%), 0 = {zeros:,}"
    if kind == "categorical":
        counts = observed.astype(str).value_counts()
        return ", ".join(f"{level} ({count:,})" for level, count in counts.items())
    if not len(observed):
        return "No observed values"
    return (f"median {display_number(observed.median())} | range "
            f"{display_number(observed.min())} to {display_number(observed.max())}")


def rebuild_codebooks():
    """Refresh counts, ranges and completeness after rebuilding a CSV."""
    for slug in ORDER:
        data_path = os.path.join(HERE, "data", f"{slug}.csv")
        codebook_path = os.path.join(HERE, "codebooks", f"{slug}_codebook.csv")
        d = pd.read_csv(data_path)
        old = pd.read_csv(codebook_path).set_index("variable")
        rows = []
        for var in d.columns:
            if var in CODEBOOK_OVERRIDES:
                description, kind = CODEBOOK_OVERRIDES[var]
            else:
                assert var in old.index, f"{slug}: no codebook entry for {var}"
                description = old.loc[var, "description"]
                kind = old.loc[var, "type"]
            rows.append({
                "variable": var,
                "description": description,
                "type": kind,
                "values": codebook_values(d[var], kind),
                "pct_complete": round(100 * d[var].notna().mean(), 1),
            })
        pd.DataFrame(rows).to_csv(codebook_path, index=False)


def outcome_li(o, miss):
    var = unwrap(o["var"])
    name = OUTCOME_LABELS.get(var, var.replace("_", " ").capitalize())
    return (f'<li><span class="outcome-name">{name}</span>'
            f'<code class="outcome-var">{var}</code>'
            f'<span class="detail">{outcome_missingness(var, miss)}</span></li>')


def derived_li(name, var, how):
    return (f'<li><span class="outcome-name">{name}</span>'
            f'<code class="outcome-var">{var}</code>'
            f'<span class="detail">{how}</span></li>')


def build_row(slug, number):
    m = META[slug]
    d, miss = missingness(slug)
    title, blurb = unwrap(m["title"]), unwrap(m["blurb"])
    region = unwrap(m["region"])

    # EXTRA_PREDICTORS is hand-written here, so a name that does not exist is a typo.
    for p in EXTRA_PREDICTORS.get(slug, []):
        assert p in d.columns, f"{slug}: extra predictor {p} is not a column"
    preds = [unwrap(p) for p in m["predictors"]] + EXTRA_PREDICTORS.get(slug, [])
    dropped = [p for p in preds if p not in d.columns]
    if dropped:
        print(f"  {slug}: upstream lists {', '.join(dropped)}, not in this CSV — skipped")
    preds = [p for p in preds if p in d.columns and p not in NOT_A_VARIABLE]
    pred_html = ", ".join(f"<code>{p}</code>{flag(p, miss)}" for p in preds)

    outs = m["outcomes"] if isinstance(m["outcomes"], list) else list(m["outcomes"].values())
    outs = [o for o in outs if unwrap(o["var"]) in d.columns]   # upstream lists a few
    outs += [{"var": v, "type": t, "detail": "", "note": ""}
             for t, v in EXTRA_OUTCOMES.get(slug, [])]
    # An excluded outcome may be a real column or one of the derived recipes below.
    dropped = EXCLUDED_OUTCOMES.get(slug, {})
    derivable = {v for _, _, v, _ in DERIVED_OUTCOMES.get(slug, [])}
    for v in dropped:
        assert v in d.columns or v in derivable, \
            f"{slug}: excluded outcome {v} is neither a column nor a derived outcome"
    outs = [o for o in outs if unwrap(o["var"]) not in dropped]
    lin = [o for o in outs if unwrap(o["type"]) in ("linear", "either")]
    log = [o for o in outs if unwrap(o["type"]) == "logistic"]

    derived = [x for x in DERIVED_OUTCOMES.get(slug, []) if x[2] not in dropped]
    lin_html = "".join(outcome_li(o, miss) for o in lin) \
        + "".join(derived_li(n, v, h) for t, n, v, h in derived if t == "linear")
    log_html = "".join(outcome_li(o, miss) for o in log) \
        + "".join(derived_li(n, v, h) for t, n, v, h in derived if t == "logistic")

    if slug in NON_PAPER_SOURCES:
        source_html = (f'<span class="source-status">No linked research paper</span>'
                       f'<a href="{unwrap(m["paper_url"])}" target="_blank" rel="noreferrer">View data source</a>')
    else:
        source_html = f'<a href="{unwrap(m["paper_url"])}" target="_blank" rel="noreferrer">View source paper</a>'

    return f'''      <tr>
        <td class="dataset"><a class="dataset-title" href="datasets/{slug}.html"><span class="dataset-number">{number}.</span> {title}</a><div class="meta">{len(d):,} rows x {d.shape[1]} columns</div><div class="links"><a class="details" href="datasets/{slug}.html">More details about this dataset</a><a class="csv" href="data/{slug}.csv" download>Download CSV dataset</a>{source_html}</div></td>
        <td class="sample">{f'<p><q>{blurb}</q></p>' if blurb else ''}</td>
        <td class="variables"><div class="vars">{pred_html}</div></td>
        <td class="outcomes"><ul class="outcome-list">{lin_html or '<li class="none">None suitable</li>'}</ul></td>
        <td class="outcomes"><ul class="outcome-list">{log_html or '<li class="none">None suitable</li>'}</ul></td>
      </tr>'''


DETAIL_TEMPLATE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Final project dataset</title><link rel="stylesheet" href="detail.css"></head><body data-dataset="{slug}"><header><div class="wrap"><a class="back" href="../index.html">← All datasets</a><h1></h1><div class="meta"></div></div></header><main class="wrap"><section><h2>Variables</h2><div class="table-shell"><table class="codebook-table"><thead><tr><th>Variable</th><th>Meaning</th><th>Type</th><th>Values or range</th><th>Complete</th></tr></thead><tbody></tbody></table></div></section><section><h2>First 20 rows</h2><div class="table-shell sample-shell"><table class="sample-table"><thead></thead><tbody></tbody></table></div></section><section><h2>Variable distributions</h2><p class="section-note">Generated with <code>inspect_cat()</code> and <code>inspect_num()</code>.</p><figure><h3>Categorical variables</h3><img class="inspect-figure" data-inspect="cat" alt="Categorical variable distributions"></figure><figure><h3>Numeric variables</h3><img class="inspect-figure" data-inspect="num" alt="Numeric variable distributions"></figure></section></main><footer><div class="wrap">Research Statistics with R · Final project dataset</div></footer><script src="detail.js"></script></body></html>
'''


def build_detail_js():
    entries = []
    for slug in ORDER:
        m = META[slug]
        d, _ = missingness(slug)
        complete = 100 * d.dropna().shape[0] / len(d)
        entries.append(f'''  {slug}: {{
    title: "{unwrap(m["title"])}", place: "{unwrap(m["region"])}", rows: "{len(d):,}", variables: "{d.shape[1]}", completeness: "{complete:.0f}% of rows complete on every variable",
    lede: "{unwrap(m["blurb"])}",
    csv: "{slug}.csv", codebook: "{slug}_codebook.csv", paper: "{unwrap(m["paper_url"])}"
  }}''')
    js = open(os.path.join(HERE, "datasets", "detail.js")).read()
    head, sep, tail = js.partition("\n};\n")
    assert sep, "could not find the dataset table in detail.js"
    return "const datasets = {\n" + ",\n".join(entries) + sep + tail


if __name__ == "__main__":
    rebuild_codebooks()
    rows = "\n\n".join(build_row(s, number) for number, s in enumerate(ORDER, start=1))
    shell = open(os.path.join(HERE, "build", "index_shell.html")).read()
    open(os.path.join(HERE, "index.html"), "w").write(shell.replace("<!--ROWS-->", rows))

    for slug in ORDER:
        with open(os.path.join(HERE, "datasets", f"{slug}.html"), "w") as fh:
            fh.write(DETAIL_TEMPLATE.format(slug=slug))

    js = build_detail_js()          # reads detail.js, so build it before truncating
    with open(os.path.join(HERE, "datasets", "detail.js"), "w") as fh:
        fh.write(js)

    print(f"built index.html and {len(ORDER)} detail pages")
