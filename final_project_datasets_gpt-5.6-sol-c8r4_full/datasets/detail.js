const datasets = {
  serbia_antibiotics: {
    title: "Antibiotic knowledge and self-medication in Serbia", place: "Serbia", rows: "500", variables: "12", completeness: "100% of rows complete on every variable",
    lede: "The study sample consisted of adult subjects who consulted general practitioners at four health centers ... 500 respondents completed the entire questionnaire.",
    csv: "serbia_antibiotics.csv", codebook: "serbia_antibiotics_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180799"
  },
  workplace_protection: {
    title: "COVID-19 workplace protection survey", place: "Multiple countries", rows: "557", variables: "9", completeness: "100% of rows complete on every variable",
    lede: "Target participants were recruited ... through non-probability convenience sampling techniques ... A total of 890 surveys were completed.",
    csv: "workplace_protection.csv", codebook: "workplace_protection_codebook.csv", paper: "https://link.springer.com/article/10.1186/s12889-022-12500-w"
  },
  sciatica_primary_care: {
    title: "Clinical signs, MRI findings and sciatica in primary care", place: "United Kingdom", rows: "395", variables: "19", completeness: "94% of rows complete on every variable",
    lede: "Patients completed questionnaires, underwent a standardised clinical assessment ... and had a lumbar spine MRI within two weeks of their assessment.",
    csv: "sciatica_primary_care.csv", codebook: "sciatica_primary_care_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0191852"
  },
  nepal_hypertension: {
    title: "Hypertension in mid-western Nepal (WHO STEPS survey)", place: "Surkhet, Nepal", rows: "1,159", variables: "21", completeness: "100% of rows complete on every variable",
    lede: "We enrolled 1159 subjects aged 30 years and above ... Trained enumerator collected socio-demographic, anthropometric, and clinical data using standard STEPS questionnaires.",
    csv: "nepal_hypertension.csv", codebook: "nepal_hypertension_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0185806"
  },
  kenya_hypertension: {
    title: "Hypertension among clients at Kenyan health facilities", place: "Kenya", rows: "1,444", variables: "18", completeness: "67% of rows complete on every variable",
    lede: "The general adult public visiting the outpatient clinics were recruited from 8 healthcare facilities in Kenya ... A total of 1444 clients were recruited.",
    csv: "kenya_hypertension.csv", codebook: "kenya_hypertension_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0334255"
  },
  peru_child_anaemia: {
    title: "Childhood anaemia and stunting in Peru (ENDES 2022)", place: "Peru", rows: "19,191", variables: "23", completeness: "90% of rows complete on every variable",
    lede: "The source of information was data from the 2022 Demographic and Family Health Survey ... face-to-face interviews conducted in selected households.",
    csv: "peru_child_anaemia.csv", codebook: "peru_child_anaemia_codebook.csv", paper: "https://journals.plos.org/globalpublichealth/article?id=10.1371/journal.pgph.0002914"
  },
  bangladesh_growth_monitoring: {
    title: "Growth-monitoring services and child nutrition in Bangladesh", place: "Bangladesh", rows: "3,038", variables: "19", completeness: "100% of rows complete on every variable",
    lede: "A descriptive mixed-method study was conducted across six sub-districts ... A total of 3038 randomly selected mothers and children under one year old were included.",
    csv: "bangladesh_growth_monitoring.csv", codebook: "bangladesh_growth_monitoring_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0324918"
  },
  medstudent_qol: {
    title: "Resilience, learning environment and quality of life in Brazilian medical students", place: "Brazil", rows: "1,350", variables: "22", completeness: "100% of rows complete on every variable",
    lede: "We evaluated data from a random sample of 1,350 medical students from 22 Brazilian medical schools.",
    csv: "medstudent_qol.csv", codebook: "medstudent_qol_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131535"
  },
  patient_activation: {
    title: "Patient activation for self-management in chronic disease", place: "Netherlands", rows: "1,154", variables: "27", completeness: "76% of rows complete on every variable",
    lede: "A cross-sectional survey study was conducted in primary and secondary care ... We included 1154 patients.",
    csv: "patient_activation.csv", codebook: "patient_activation_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0126400"
  },
  nhanes_adults: {
    title: "NHANES adults - US national health and nutrition survey", place: "United States", rows: "488", variables: "31", completeness: "35% of rows complete on every variable",
    lede: "",
    csv: "nhanes_adults.csv", codebook: "nhanes_adults_codebook.csv", paper: "https://www.cdc.gov/nchs/nhanes/"
  },
  us_births_2014: {
    title: "US births 2014 - birthweight and prematurity", place: "United States", rows: "1,000", variables: "13", completeness: "79% of rows complete on every variable",
    lede: "",
    csv: "us_births_2014.csv", codebook: "us_births_2014_codebook.csv", paper: "https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm"
  },
  ggt_arterial_stiffness: {
    title: "Liver enzymes and arterial stiffness in a Japanese health-check cohort", place: "Japan", rows: "912", variables: "23", completeness: "93% of rows complete on every variable",
    lede: "912 Japanese men and women aged 24–84 years ... received a medical health check-up programme with ... an automatic waveform analyser to measure baPWV.",
    csv: "ggt_arterial_stiffness.csv", codebook: "ggt_arterial_stiffness_codebook.csv", paper: "https://bmjopen.bmj.com/content/4/10/e005413"
  },
  cimt_rheumatoid: {
    title: "Cardiovascular risk and carotid artery thickness in rheumatoid arthritis", place: "Netherlands", rows: "470", variables: "25", completeness: "86% of rows complete on every variable",
    lede: "Subjects with RA and healthy controls without RA ... underwent a standard physical examination and laboratory measurements ... cIMT was measured semi-automatically by ultrasound.",
    csv: "cimt_rheumatoid.csv", codebook: "cimt_rheumatoid_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140844"
  },
  hiv_lung_6mwt: {
    title: "HIV infection and exercise capacity (6-minute walk distance)", place: "United States", rows: "427", variables: "21", completeness: "40% of rows complete on every variable",
    lede: "PLWH and HIV-uninfected individuals were enrolled from 2 clinical centers and completed a 6-MWD, spirometry, diffusing capacity ... and St. George’s Respiratory Questionnaire.",
    csv: "hiv_lung_6mwt.csv", codebook: "hiv_lung_6mwt_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0212975"
  },
  famuss_strength: {
    title: "FAMuSS - genotype and muscle response to resistance training", place: "United States", rows: "595", variables: "9", completeness: "100% of rows complete on every variable",
    lede: "The muscle phenotypes examined were baseline elbow flexor muscle size and strength and the response ... to 12 wk of progressive resistance exercise training.",
    csv: "famuss_strength.csv", codebook: "famuss_strength_codebook.csv", paper: "https://journals.physiology.org/doi/full/10.1152/japplphysiol.01139.2004"
  },
  tibial_bone_strength: {
    title: "Muscle power and tibial bone strength in healthy adults", place: "United States", rows: "142", variables: "8", completeness: "100% of rows complete on every variable",
    lede: "A convenience sample of 142 participants ... was recruited for this observational, cross-sectional study, from the faculty, staff, and students at a mid-sized regional university.",
    csv: "tibial_bone_strength.csv", codebook: "tibial_bone_strength_codebook.csv", paper: "https://pmc.ncbi.nlm.nih.gov/articles/PMC9186462/"
  },
  nigeria_malaria_households: {
    title: "Asymptomatic malaria in households of confirmed cases in Abuja", place: "Abuja, Nigeria", rows: "602", variables: "15", completeness: "100% of rows complete on every variable",
    lede: "Overall, we recruited 602 participants from 107 households linked to 107 malaria patients attending the health facilities.",
    csv: "nigeria_malaria_households.csv", codebook: "nigeria_malaria_households_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0203686"
  },
  ageing_qol_europe: {
    title: "Quality of life among adults in Finland, Poland and Spain", place: "Finland, Poland and Spain", rows: "5,341", variables: "38", completeness: "100% of rows complete on every variable",
    lede: "COURAGE in Europe is an observational, cross-sectional study of the general community dwelling adult population.",
    csv: "ageing_qol_europe.csv", codebook: "ageing_qol_europe_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0159293"
  },
  essens_adolescents: {
    title: "Screen time and physical activity among Norwegian adolescents", place: "Norway", rows: "742", variables: "20", completeness: "82% of rows complete on every variable",
    lede: "A cross-sectional study including 742 adolescents was conducted in 2016. Data were collected at school through an online questionnaire.",
    csv: "essens_adolescents.csv", codebook: "essens_adolescents_codebook.csv", paper: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0241887"
  }
};

function parseCsv(text) {
  const rows = []; let row = []; let cell = ""; let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(cell); cell = ""; }
    else if (c === '\n') { row.push(cell.replace(/\r$/, "")); rows.push(row); row = []; cell = ""; }
    else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function render() {
  const key = document.body.dataset.dataset;
  const d = datasets[key];
  if (!d) return;
  document.title = `${d.title} | Final project dataset`;
  document.querySelector("h1").textContent = d.title;
  document.querySelector(".meta").textContent = `${d.rows} rows x ${d.variables} columns`;
  document.querySelectorAll("[data-inspect]").forEach(image => {
    image.src = `../figures/${key}_inspect_${image.dataset.inspect}.png`;
  });

  try {
    const response = await fetch(`../codebooks/${d.codebook}`);
    if (!response.ok) throw new Error("Codebook unavailable");
    const rows = parseCsv(await response.text());
    const body = rows.slice(1).filter(r => r.length >= 5).map(r => `<tr><td>${escapeHtml(r[0])}</td><td>${escapeHtml(r[1])}</td><td>${escapeHtml(r[2])}</td><td>${escapeHtml(r[3])}</td><td>${escapeHtml(r[4])}%</td></tr>`).join("");
    document.querySelector(".codebook-table tbody").innerHTML = body;
  } catch (error) {
    document.querySelector(".codebook-table").closest(".table-shell").innerHTML = '<p class="error">The codebook could not be loaded. Open this page through the course website rather than directly from the file system.</p>';
  }

  try {
    const response = await fetch(`../data/${d.csv}`);
    if (!response.ok) throw new Error("Dataset unavailable");
    const rows = parseCsv(await response.text());
    const headers = rows[0] || [];
    document.querySelector(".sample-table thead").innerHTML = `<tr>${headers.map(header => `<th>${escapeHtml(header)}</th>`).join("")}</tr>`;
    document.querySelector(".sample-table tbody").innerHTML = rows.slice(1, 21).map(row => `<tr>${headers.map((_, index) => `<td>${escapeHtml(row[index] ?? "")}</td>`).join("")}</tr>`).join("");
  } catch (error) {
    document.querySelector(".sample-shell").innerHTML = '<p class="error">The dataset preview could not be loaded. Open this page through the course website rather than directly from the file system.</p>';
  }
}
render();
