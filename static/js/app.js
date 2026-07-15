const form = document.getElementById("approvalForm");
const sampleButton = document.getElementById("sampleButton");
const resultPanel = document.getElementById("resultPanel");
const decisionTitle = document.getElementById("decisionTitle");
const decisionText = document.getElementById("decisionText");
const confidenceValue = document.getElementById("confidenceValue");

const sampleApplicant = {
  gender: "Female",
  income_type: "Working",
  education: "Higher education",
  family_status: "Married",
  housing_type: "House",
  employment_status: "Employed",
  annual_income: 86000,
  employment_years: 8,
  age: 38,
  credit_history_years: 7,
  existing_loan_balance: 9000,
  credit_inquiries: 1,
  past_due_count: 0,
};

function readPayload() {
  const data = new FormData(form);
  const payload = {};
  data.forEach((value, key) => {
    payload[key] = Number.isNaN(Number(value)) || value === "" ? value : Number(value);
  });
  return payload;
}

function fillSample() {
  Object.entries(sampleApplicant).forEach(([key, value]) => {
    const field = form.elements[key];
    if (field) {
      field.value = value;
    }
  });
}

function showDecision(result) {
  const approved = result.prediction === "Approved";
  resultPanel.classList.toggle("approved", approved);
  resultPanel.classList.toggle("rejected", !approved);
  decisionTitle.textContent = result.prediction;
  decisionText.textContent = approved
    ? "The applicant profile is likely eligible for credit card approval."
    : "The applicant profile should be rejected or routed for manual compliance review.";
  confidenceValue.textContent =
    typeof result.confidence === "number" ? `${Math.round(result.confidence * 100)}%` : "Unavailable";
}

async function submitPrediction(event) {
  event.preventDefault();
  decisionTitle.textContent = "Scoring...";
  decisionText.textContent = "Evaluating the applicant profile.";
  confidenceValue.textContent = "--";

  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(readPayload()),
  });

  const result = await response.json();
  if (!response.ok) {
    decisionTitle.textContent = "Unable to score";
    decisionText.textContent = result.error || "Prediction failed.";
    return;
  }

  showDecision(result);
}

sampleButton.addEventListener("click", fillSample);
form.addEventListener("submit", submitPrediction);
