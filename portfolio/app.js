const FEATURE_NAMES = [
  "Name token overlap", "Address token overlap", "Name spelling overlap",
  "Address spelling overlap", "Name containment", "Address containment",
  "Exact normalized name", "Exact normalized address", "Postal overlap", "Country match"
];
const SUFFIXES = new Set(["inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "pvt", "private"]);
const ADDRESS_ALIASES = { street:"st", avenue:"ave", road:"rd", boulevard:"blvd", drive:"dr", lane:"ln", court:"ct", highway:"hwy", suite:"ste", apartment:"apt", north:"n", south:"s", east:"e", west:"w", northeast:"ne", northwest:"nw", southeast:"se", southwest:"sw" };
const US_STATES = { al:"alabama", ak:"alaska", az:"arizona", ar:"arkansas", ca:"california", co:"colorado", ct:"connecticut", de:"delaware", fl:"florida", ga:"georgia", hi:"hawaii", id:"idaho", il:"illinois", in:"indiana", ia:"iowa", ks:"kansas", ky:"kentucky", la:"louisiana", me:"maine", md:"maryland", ma:"massachusetts", mi:"michigan", mn:"minnesota", ms:"mississippi", mo:"missouri", mt:"montana", ne:"nebraska", nv:"nevada", nh:"new hampshire", nj:"new jersey", nm:"new mexico", ny:"new york", nc:"north carolina", nd:"north dakota", oh:"ohio", ok:"oklahoma", or:"oregon", pa:"pennsylvania", ri:"rhode island", sc:"south carolina", sd:"south dakota", tn:"tennessee", tx:"texas", ut:"utah", vt:"vermont", va:"virginia", wa:"washington", wv:"west virginia", wi:"wisconsin", wy:"wyoming" };
const ADDRESS_IGNORED = new Set(["ste", "apt", "unit", "floor", "fl", "building", "bldg", "room", "rm"]);
const GENERIC_NAME_TOKENS = new Set(["shop", "store", "services", "service", "center", "centre", "company", "business", "group", "the"]);
// Editable category vocabulary: aliases map to a business type, never to an
// identity. Shared category + same normalized address prompts human review.
const BUSINESS_CATEGORY_ALIASES = {
  coffee_cafe: ["coffee", "cafe", "café", "cappuccino", "espresso", "coffeehouse", "coffee house", "roastery", "roaster", "tea room", "tea house", "bubble tea", "boba"],
  nightlife: ["pub", "bar", "nightclub", "night club", "tavern", "saloon", "lounge", "taproom", "brewpub"],
  restaurant: ["restaurant", "diner", "bistro", "eatery", "brasserie", "grill", "trattoria"],
  bakery: ["bakery", "baker", "patisserie", "pastry shop", "bakehouse"],
  automotive: ["car mechanic", "auto mechanic", "automotive mechanic", "mechanic", "mechanics", "garage", "auto repair", "car repair", "automotive repair", "vehicle repair", "auto shop", "car shop", "automotive service", "body shop", "tire shop", "tyre shop", "dealership", "auto dealer", "car dealer", "auto service", "vehicle service"],
  apparel: ["tailor", "tailoring", "suit shop", "clothing store", "apparel", "boutique", "garment shop"],
  beauty: ["hair salon", "salon", "hairdresser", "barber", "barbershop", "nail salon", "spa"],
  lodging: ["hotel", "motel", "inn", "hostel", "bed and breakfast", "guesthouse"],
  pharmacy: ["pharmacy", "drugstore", "chemist", "apothecary"],
  dental: ["dentist", "dental", "dental clinic"],
  healthcare: ["doctor", "physician", "medical clinic", "health clinic", "hospital"],
  grocery: ["grocery", "grocer", "supermarket", "convenience store", "market"],
  bookstore: ["bookstore", "bookshop", "bookseller"],
  florist: ["florist", "flower shop", "flower store"],
  fitness: ["gym", "fitness center", "fitness club", "health club"],
  banking: ["bank", "credit union"],
  legal: ["lawyer", "attorney", "law firm", "legal services"],
  education: ["school", "college", "university", "academy"]
};
const fmt = new Intl.NumberFormat("en-US");
let model = null;
let evaluation = null;

const $ = (id) => document.getElementById(id);
const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const fixed = (value, digits = 3) => Number(value).toFixed(digits);

function normalize(value) {
  return String(value || "").toLowerCase().replace(/ß/g, "ss")
    .normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ");
}
function tokenSet(value, dropSuffix = false) {
  const tokens = new Set(normalize(value).match(/[a-z0-9]+/g) || []);
  if (dropSuffix) for (const suffix of SUFFIXES) tokens.delete(suffix);
  return tokens;
}
function meaningfulNameTokens(value) {
  return new Set([...tokenSet(value, true)].filter(token => !GENERIC_NAME_TOKENS.has(token)));
}
function gramSet(value) {
  const s = normalize(value), result = new Set();
  for (let i = 0; i <= s.length - 3; i++) result.add(s.slice(i, i + 3));
  return result;
}
function jaccard(a, b) {
  let shared = 0;
  for (const item of a) if (b.has(item)) shared++;
  const union = a.size + b.size - shared;
  return union ? shared / union : 0;
}
function containment(a, b) {
  if (!Math.min(a.size, b.size)) return 0;
  let shared = 0;
  for (const item of a) if (b.has(item)) shared++;
  return Math.min(shared / Math.min(a.size, b.size), 1);
}
function addressTokens(value) {
  const tokens = normalize(value).match(/[a-z0-9]+/g) || [];
  const postalIndex = tokens.findIndex(token => /^\d{5}$/.test(token));
  if (postalIndex > 0) {
    const state = tokens[postalIndex - 1];
    if (US_STATES[state]) tokens[postalIndex - 1] = `state_${state}`;
    else {
      const abbreviation = Object.keys(US_STATES).find(code => US_STATES[code] === state);
      if (abbreviation) tokens[postalIndex - 1] = `state_${abbreviation}`;
    }
  }
  return new Set(tokens
    .filter(token => !/^\d+$/.test(token))
    .map(token => ADDRESS_ALIASES[token] || token)
    .filter(token => !ADDRESS_IGNORED.has(token)));
}
function unitIdentifier(value) {
  return normalize(value).match(/\b(?:suite|ste|unit|apt|apartment|floor|fl)\s+([a-z0-9-]+)/)?.[1] || "";
}
function businessConcepts(value) {
  const normalized = ` ${normalize(value)} `;
  return new Set(Object.entries(BUSINESS_CATEGORY_ALIASES)
    .filter(([, aliases]) => aliases.some(alias => normalized.includes(` ${normalize(alias)} `)))
    .map(([category]) => category));
}
function categoryOnlyName(value) {
  const nameTokens = meaningfulNameTokens(value);
  if (!nameTokens.size || !businessConcepts(value).size) return false;
  const categoryTokens = new Set(Object.values(BUSINESS_CATEGORY_ALIASES)
    .flatMap(aliases => aliases.flatMap(alias => normalize(alias).split(" ")))
    .filter(token => token && !GENERIC_NAME_TOKENS.has(token) && !SUFFIXES.has(token)));
  return [...nameTokens].every(token => categoryTokens.has(token));
}
function canonicalAddress(value) {
  const normalized = normalize(value).replace(/\b(\d{5})\s+(\d{4})\b/g, "$1");
  const tokens = normalized.match(/[a-z0-9]+/g) || [];
  const postalIndex = tokens.findIndex(token => /^\d{5}$/.test(token));
  if (postalIndex > 0) {
    const state = tokens[postalIndex - 1];
    if (US_STATES[state]) tokens[postalIndex - 1] = `state_${US_STATES[state]}`;
    else {
      const abbreviation = Object.keys(US_STATES).find(code => US_STATES[code] === state);
      if (abbreviation) tokens[postalIndex - 1] = `state_${US_STATES[abbreviation]}`;
    }
  }
  return tokens.map(token => ADDRESS_ALIASES[token] || token).join(" ");
}
function featuresFromForm() {
  const nameA = normalize($("nameA").value), nameB = normalize($("nameB").value);
  const addressA = normalize($("addressA").value), addressB = normalize($("addressB").value);
  const nameTokensA = tokenSet(nameA, true), nameTokensB = tokenSet(nameB, true);
  const addressTokensA = tokenSet(addressA), addressTokensB = tokenSet(addressB);
  const postalA = new Set([...addressTokensA].filter(x => /^\d{5,6}$/.test(x)));
  const postalB = new Set([...addressTokensB].filter(x => /^\d{5,6}$/.test(x)));
  const countryA = normalize($("countryA").value);
  const countryB = normalize($("countryB").value);
  return [
    jaccard(nameTokensA, nameTokensB), jaccard(addressTokensA, addressTokensB),
    jaccard(gramSet(nameA), gramSet(nameB)), jaccard(gramSet(addressA), gramSet(addressB)),
    containment(nameTokensA, nameTokensB), containment(addressTokensA, addressTokensB),
    Number(nameA === nameB), Number(addressA === addressB),
    Number([...postalA].some(x => postalB.has(x))), Number(countryA === countryB)
  ];
}
function modelScore(features) {
  let z = model.w[0];
  for (let i = 0; i < features.length; i++) {
    const sd = Math.abs(model.sd[i]) < 1e-8 ? 1 : model.sd[i];
    z += model.w[i + 1] * ((features[i] - model.mu[i]) / sd);
  }
  return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
}
function renderFeatures(values) {
  $("featureList").innerHTML = values.map((value, index) => `
    <div class="feature-row"><span>${FEATURE_NAMES[index]}</span><span class="feature-track"><i class="feature-fill" style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></span><span class="feature-value">${value.toFixed(2)}</span></div>
  `).join("");
}
function scorePair() {
  if (!model) return;
  const values = featuresFromForm();
  const score = modelScore(values);
  const nameA = normalize($("nameA").value), nameB = normalize($("nameB").value);
  const addressA = normalize($("addressA").value), addressB = normalize($("addressB").value);
  const threshold = Number($("displayThreshold").value);
  const passes = score >= threshold;
  const countryConflict = Boolean(normalize($("countryA").value) && normalize($("countryB").value) && values[9] === 0);
  const meaningfulNameA = meaningfulNameTokens($("nameA").value), meaningfulNameB = meaningfulNameTokens($("nameB").value);
  const meaningfulNameTextA = [...meaningfulNameA].sort().join(" "), meaningfulNameTextB = [...meaningfulNameB].sort().join(" ");
  const meaningfulNameConflict = meaningfulNameA.size > 0 && meaningfulNameB.size > 0
    && jaccard(meaningfulNameA, meaningfulNameB) === 0
    && jaccard(gramSet(meaningfulNameTextA), gramSet(meaningfulNameTextB)) < 0.12;
  const nameConflict = meaningfulNameConflict;
  const sharedBusinessConcept = [...businessConcepts($("nameA").value)].some(concept => businessConcepts($("nameB").value).has(concept));
  const sameCanonicalAddress = canonicalAddress($("addressA").value) === canonicalAddress($("addressB").value);
  const categoryOnlyNamePair = categoryOnlyName($("nameA").value) || categoryOnlyName($("nameB").value);
  const needsMatchReview = nameA !== nameB && sharedBusinessConcept && sameCanonicalAddress
    && (nameConflict || categoryOnlyNamePair);
  const houseA = normalize($("addressA").value).match(/^\d+[a-z]?/)?.[0] || "";
  const houseB = normalize($("addressB").value).match(/^\d+[a-z]?/)?.[0] || "";
  const houseNumberConflict = Boolean(houseA && houseB && houseA !== houseB);
  const unitA = unitIdentifier($("addressA").value), unitB = unitIdentifier($("addressB").value);
  const unitConflict = Boolean(unitA && unitB && unitA !== unitB);
  const addrA = addressTokens($("addressA").value), addrB = addressTokens($("addressB").value);
  const uniqueAddressA = [...addrA].filter(token => !addrB.has(token));
  const uniqueAddressB = [...addrB].filter(token => !addrA.has(token));
  const localityConflict = uniqueAddressA.length > 0 && uniqueAddressB.length > 0;
  const canonicalAddressA = new Set(canonicalAddress($("addressA").value).split(" ").filter(Boolean));
  const canonicalAddressB = new Set(canonicalAddress($("addressB").value).split(" ").filter(Boolean));
  const addressIsSubset = (shorter, longer) => shorter.size > 0 && shorter.size < longer.size
    && [...shorter].every(token => longer.has(token));
  const addressDetailGap = addressIsSubset(canonicalAddressA, canonicalAddressB)
    || addressIsSubset(canonicalAddressB, canonicalAddressA);
  const addressMissing = !addressA || !addressB;
  const postalA = new Set(normalize($("addressA").value).match(/\b\d{5,6}\b/g) || []);
  const postalB = new Set(normalize($("addressB").value).match(/\b\d{5,6}\b/g) || []);
  const postalConflict = postalA.size > 0 && postalB.size > 0 && ![...postalA].some(code => postalB.has(code));
  const addressConflict = houseNumberConflict || unitConflict || localityConflict || postalConflict;
  const conflict = countryConflict || nameConflict || addressConflict || addressDetailGap || addressMissing;
  const needsAddressReview = passes && (addressConflict || addressDetailGap || addressMissing)
    && !countryConflict && !nameConflict;
  const finalPossibleMatch = passes && !conflict && !needsMatchReview;
  const decisionText = countryConflict ? "NOT EVALUATED" : needsMatchReview ? "REVIEW MATCH" : nameConflict ? "NO MATCH" : needsAddressReview ? "REVIEW ADDRESS" : finalPossibleMatch ? "POSSIBLE MATCH" : "NO MATCH";
  const addressDifferences = [...new Set([...uniqueAddressA, ...uniqueAddressB])].slice(0, 6).join(" / ");
  const decisionDetail = countryConflict
    ? "This pair is outside the production blocker's country rule and cannot be matched."
    : needsMatchReview
      ? "The names point to the same business category and the normalized addresses match. They could be alternate descriptions or separate co-located businesses; verify identity before matching."
    : nameConflict && houseNumberConflict
      ? `No meaningful name overlap and street numbers differ (${houseA} vs ${houseB}). The pair fails the demo's consistency checks.`
    : nameConflict
      ? "No meaningful business-name overlap. Shared address tokens inflated the raw score, so the pair fails the demo's consistency checks."
      : addressDetailGap || addressMissing
        ? `${addressMissing ? "At least one address is missing." : "One address contains less location detail than the other."} ${categoryOnlyNamePair ? "One business name only identifies a broad business category." : ""} The score is not enough to establish that these records describe the same business; review the records.`.trim()
      : postalConflict
        ? "The postal codes differ. This may be a data error or a different location; verify it before deciding."
      : localityConflict
        ? `The addresses contain different place terms (${addressDifferences}). This could be a formatting/data issue or a different location; verify it before deciding.`
      : houseNumberConflict
        ? `Street numbers differ (${houseA} vs ${houseB}). This could be a typo or neighboring storefront; review rather than auto-matching or rejecting.`
      : unitConflict
        ? `Unit identifiers differ (${unitA} vs ${unitB}). This may be an address typo or separate suite; review before deciding.`
      : passes
        ? "No contradiction was detected. This is a possible match for further review, not proof of identity."
        : "The raw classifier score is below its cutoff.";
  const overrideReason = countryConflict ? "country blocker" : needsMatchReview ? "related-name review" : nameConflict ? "name conflict check" : needsAddressReview ? "address review" : "";
  const rawContext = countryConflict
    ? `Raw classifier output: ${score.toFixed(3)}; not actionable because this pair is excluded by the ${overrideReason}.`
      : overrideReason
      ? `Raw classifier output: ${score.toFixed(3)} (${passes ? "above" : "below"} cutoff ${threshold.toFixed(2)}); ${needsMatchReview ? "related business terms need identity review" : nameConflict ? "the name conflict blocks a match" : "the address discrepancy needs review before deciding"}.`
      : `Raw classifier output: ${score.toFixed(3)} (${passes ? "above" : "below"} cutoff ${threshold.toFixed(2)}). This is a model score, not a calibrated probability.`;
  $("scoreValue").textContent = countryConflict ? "—" : finalPossibleMatch ? "✓" : needsAddressReview || needsMatchReview ? "?" : "×";
  $("scoreRing").style.background = "transparent";
  $("scoreRing").style.borderColor = finalPossibleMatch ? "var(--mint)" : "var(--amber)";
  $("scoreRing").style.color = finalPossibleMatch ? "var(--mint)" : "var(--amber)";
  $("decisionLabel").textContent = decisionText;
  $("decisionLabel").style.color = finalPossibleMatch ? "var(--mint)" : "var(--amber)";
  $("decisionDetail").textContent = decisionDetail;
  $("rawScoreContext").textContent = rawContext;
  $("decisionThreshold").textContent = threshold.toFixed(2);
  renderFeatures(values);
}

function renderCaps(selectedCap = 20) {
  if (!evaluation) return;
  const records = evaluation.candidate_cap_sweep;
  const selected = records.find(row => row.max_candidates_per_source === selectedCap) || records[0];
  const effectiveCap = selected.max_candidates_per_source;
  $("capOutput").value = effectiveCap;
  $("capSlider").value = effectiveCap;
  $("selectedLinks").textContent = fmt.format(selected.candidate_links);
  $("selectedRecall").textContent = pct(selected.candidate_recall_for_known_links, 1);
  $("selectedF05").textContent = fixed(selected.macro_f05, 3);
  $("selectedMicro").textContent = `${pct(selected.micro_precision)} / ${pct(selected.micro_recall)}`;
  $("selectedCi").textContent = `${fixed(selected.macro_f05_bootstrap_95pct_ci[0], 3)}–${fixed(selected.macro_f05_bootstrap_95pct_ci[1], 3)}`;
  $("headlineF05").textContent = fixed(selected.macro_f05, 3);
  $("headlineRecall").textContent = pct(selected.candidate_recall_for_known_links, 1);
  $("headlineAvg").textContent = fixed(selected.average_candidates_per_s1, 1);
  $("headlineReduction").textContent = pct(selected.candidate_reduction_ratio, 4);
  $("thresholdLabel").textContent = fixed(evaluation.threshold, 2);
  $("truthLinks").textContent = fmt.format(evaluation.ground_truth_links);
  $("singletonCount").textContent = fmt.format(evaluation.singleton_entities);
  $("ciLabel").textContent = `${fixed(selected.macro_f05_bootstrap_95pct_ci[0], 3)}–${fixed(selected.macro_f05_bootstrap_95pct_ci[1], 3)}`;
  renderTable(records, effectiveCap);
  renderChart(records, effectiveCap);
  updateRangeFill();
}
function updateRangeFill() {
  const slider = $("capSlider");
  if (!slider) return;
  const min = Number(slider.min), max = Number(slider.max);
  const ratio = max > min ? (Number(slider.value) - min) / (max - min) : 0;
  slider.style.background = `linear-gradient(to right, var(--mint) 0%, var(--mint) ${ratio * 100}%, #2a4148 ${ratio * 100}%)`;
}
function renderTable(records, selectedCap) {
  $("capTable").innerHTML = records.map(row => {
    const ci = row.macro_f05_bootstrap_95pct_ci;
    return `<tr class="${row.max_candidates_per_source === selectedCap ? "selected-row" : ""}">
      <td>${row.max_candidates_per_source} / source</td>
      <td>${fixed(row.average_candidates_per_s1, 1)}</td>
      <td>${fmt.format(row.candidate_links)}</td>
      <td>${pct(row.candidate_recall_for_known_links, 1)}</td>
      <td>${fixed(row.macro_f05, 3)} <span class="table-ci">[${fixed(ci[0], 3)}–${fixed(ci[1], 3)}]</span></td>
    </tr>`;
  }).join("");
}
function renderChart(records, selectedCap) {
  const host = $("capChart");
  const width = Math.max(320, host.clientWidth || 560), height = 250;
  const m = { left: 44, right: 14, top: 18, bottom: 34 };
  const innerW = width - m.left - m.right, innerH = height - m.top - m.bottom;
  // Derive the value domain from the data so the chart stays correct when the
  // evaluation report is regenerated with different scores.
  const values = records.flatMap(row => [row.macro_f05, ...row.macro_f05_bootstrap_95pct_ci]);
  const low = Math.min(...values), high = Math.max(...values);
  const pad = Math.max(0.005, (high - low) * 0.15);
  const yMin = Math.max(0, low - pad), yMax = high + pad;
  const x = i => m.left + i * innerW / Math.max(1, records.length - 1);
  const y = value => m.top + (yMax - value) * innerH / (yMax - yMin || 1);
  const gridValues = Array.from({ length: 6 }, (_, index) => yMin + (yMax - yMin) * index / 5);
  const grid = gridValues.map(value => `<line class="chart-grid" x1="${m.left}" x2="${width-m.right}" y1="${y(value)}" y2="${y(value)}"/><text class="chart-axis" x="${m.left-8}" y="${y(value)+3}" text-anchor="end">${value.toFixed(2)}</text>`).join("");
  const path = records.map((row, i) => `${i ? "L" : "M"}${x(i)},${y(row.macro_f05)}`).join(" ");
  const area = `${path} L${x(records.length-1)},${height-m.bottom} L${x(0)},${height-m.bottom} Z`;
  const errors = records.map((row, i) => {
    const [lo, hi] = row.macro_f05_bootstrap_95pct_ci;
    return `<line class="ci-line" x1="${x(i)}" x2="${x(i)}" y1="${y(hi)}" y2="${y(lo)}"/><line class="ci-cap" x1="${x(i)-4}" x2="${x(i)+4}" y1="${y(hi)}" y2="${y(hi)}"/><line class="ci-cap" x1="${x(i)-4}" x2="${x(i)+4}" y1="${y(lo)}" y2="${y(lo)}"/>`;
  }).join("");
  const points = records.map((row, i) => {
    const active = row.max_candidates_per_source === selectedCap;
    return `<circle class="chart-point ${active ? "selected" : ""}" cx="${x(i)}" cy="${y(row.macro_f05)}" r="${active ? 5.5 : 3.8}"><title>Cap ${row.max_candidates_per_source}: F0.5 ${row.macro_f05.toFixed(3)}, candidate recall ${(row.candidate_recall_for_known_links*100).toFixed(1)}%</title></circle>`;
  }).join("");
  const xLabels = records.map((row, i) => `<text class="chart-axis" x="${x(i)}" y="${height-10}" text-anchor="middle">${row.max_candidates_per_source}</text>`).join("");
  const selectedIndex = records.findIndex(row => row.max_candidates_per_source === selectedCap);
  const selectedLine = selectedIndex >= 0 ? `<line class="selected-guide" x1="${x(selectedIndex)}" x2="${x(selectedIndex)}" y1="${m.top}" y2="${height-m.bottom}"/>` : "";
  host.innerHTML = `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><defs><linearGradient id="areaFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#67e2b2" stop-opacity=".18"/><stop offset="1" stop-color="#67e2b2" stop-opacity="0"/></linearGradient></defs>${grid}<path class="chart-area" d="${area}"/>${selectedLine}${errors}<path class="chart-line" d="${path}"/>${points}${xLabels}<text class="chart-axis" x="${width-m.right}" y="${height-10}" text-anchor="end">cap / source</text></svg>`;
}

async function init() {
  try {
    const [modelResponse, evaluationResponse] = await Promise.all([fetch("model.json"), fetch("evaluation.json")]);
    if (!modelResponse.ok || !evaluationResponse.ok) throw new Error("Could not load the model or evaluation report.");
    model = await modelResponse.json();
    evaluation = await evaluationResponse.json();
    $("modelStatus").textContent = `weights loaded · threshold ${model.threshold.toFixed(2)}`;
    // Keep the display cutoff on the model's calibrated operating point; the
    // other options are for exploring the review bands below it.
    const tuned = $("displayThreshold").querySelector(`option[value="${model.threshold}"]`);
    if (tuned) $("displayThreshold").value = tuned.value;
    renderCaps(20);
    scorePair();
  } catch (error) {
    $("modelStatus").textContent = "could not load model files";
    $("decisionLabel").textContent = "Start a local web server";
    const caveat = $("scoreCaveat");
    if (caveat) caveat.textContent = error.message;
    console.error(error);
  }
  $("pairForm").addEventListener("submit", event => { event.preventDefault(); scorePair(); });
  $("pairForm").addEventListener("input", scorePair);
  $("displayThreshold").addEventListener("change", scorePair);
  $("capSlider").addEventListener("input", event => renderCaps(Number(event.target.value)));
  updateRangeFill();
  window.addEventListener("resize", () => evaluation && renderChart(evaluation.candidate_cap_sweep, Number($("capSlider").value)));
}
init();
