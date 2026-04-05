const configNode = document.getElementById('dashboard-config');
const transactionNode = document.getElementById('transaction-data');

const dashboardConfig = configNode ? JSON.parse(configNode.textContent || '{}') : {};
const transactionData = transactionNode ? JSON.parse(transactionNode.textContent || '{}') : null;

const searchForm = document.getElementById('search-form');
const searchInput = document.getElementById('txn-input');
const searchMessage = document.getElementById('search-message');
const historyBody = document.getElementById('history-body');

const transactionFields = document.getElementById('transaction-fields');
const transactionTitle = document.getElementById('transaction-title');
const transactionScore = document.getElementById('txn-score');
const transactionLabel = document.getElementById('txn-label');
const transactionStatus = document.getElementById('txn-status');
const transactionRisk = document.getElementById('txn-risk');
const transactionIdText = document.getElementById('transactionIdText');
const transactionStatusText = document.getElementById('transactionStatusText');
const fraudLabelText = document.getElementById('fraudLabelText');
const fraudTypeText = document.getElementById('fraudTypeText');
const transactionMessage = document.getElementById('transaction-message');
const riskPill = document.getElementById('risk-pill');
const scoreText = document.getElementById('fraudScoreText');
const modelBarRows = Array.from(document.querySelectorAll('#model-bars .bar-row'));
const actionButtons = {
  block: document.getElementById('action-block'),
  flag: document.getElementById('action-flag'),
  review: document.getElementById('action-review'),
  safe: document.getElementById('action-safe'),
};

let gaugeChart = null;

function readApiJSON(response) {
  return response.json().catch(() => ({}));
}

function riskBand(score) {
  if (score > 70) return 'high';
  if (score >= 40) return 'medium';
  return 'low';
}

function riskLabel(score) {
  if (score > 70) return 'High';
  if (score >= 40) return 'Medium';
  return 'Low';
}

function riskTier(score) {
  if (score > 70) return 'FRAUD';
  if (score >= 40) return 'SUSPICIOUS';
  return 'SAFE';
}

function scoreColor(score) {
  if (score > 70) return '#ef4444';
  if (score >= 40) return '#f59e0b';
  return '#22c55e';
}

function updateSvgGauge(score) {
  const arc = document.getElementById('gaugeArc');
  if (!arc) return;

  const safeScore = Math.max(0, Math.min(100, Number(score || 0)));
  const radius = 90;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (safeScore / 100) * circumference;

  arc.style.stroke = scoreColor(safeScore);
  arc.style.strokeDasharray = `${circumference}`;
  arc.style.strokeDashoffset = `${offset}`;
}

function statusBand(status) {
  const normalized = String(status || 'ACTIVE').toUpperCase();
  if (normalized === 'BLOCKED') return 'blocked';
  if (normalized === 'FLAGGED') return 'flagged';
  if (normalized === 'REVIEW') return 'review';
  return 'active';
}

function formatValue(value) {
  if (value === null || value === undefined || value === '') return '-';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
  }
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  return String(value);
}

function buildFieldCard(label, value) {
  const wrapper = document.createElement('div');
  wrapper.className = 'detail-item';
  const labelEl = document.createElement('span');
  labelEl.textContent = label;
  const valueEl = document.createElement('strong');
  valueEl.textContent = formatValue(value);
  wrapper.append(labelEl, valueEl);
  return wrapper;
}

function updateGauge(score) {
  const safeScore = Math.max(0, Math.min(100, Number(score || 0)));
  if (scoreText) scoreText.textContent = String(Math.round(safeScore));

  updateSvgGauge(safeScore);

  const color = scoreColor(safeScore);
  document.documentElement.style.setProperty('--risk-accent', color);
  if (riskPill) {
    const band = riskBand(safeScore);
    riskPill.className = `pill pill-${band}`;
    riskPill.textContent = riskTier(safeScore);
  }
}

function updateModelBars(modelScores) {
  if (!modelBarRows.length) return;
  const values = {
    xgboost: Number(modelScores?.xgboost ?? 0),
    lightgbm: Number(modelScores?.lightgbm ?? 0),
    ensemble: Number(modelScores?.ensemble ?? 0),
  };

  modelBarRows.forEach((row) => {
    const label = String(row.querySelector('span')?.textContent || '').trim().toLowerCase();
    const fill = row.querySelector('.bar i');
    const value = row.querySelector('strong');
    const score = values[label] ?? 0;
    if (fill) fill.style.width = `${Math.max(0, Math.min(100, score))}%`;
    if (value) value.textContent = `${Math.round(score)}%`;
  });
}

function renderTransactionFields(record) {
  if (!transactionFields || !record) return;

  transactionFields.innerHTML = '';

  const keyBlock = document.createElement('div');
  keyBlock.className = 'detail-block';
  const keyTitle = document.createElement('h4');
  keyTitle.textContent = 'Key fields';
  keyBlock.appendChild(keyTitle);

  const keyFields = document.createElement('div');
  keyFields.className = 'detail-grid';
  const keyEntries = [
    ['Transaction ID', record.TransactionID || record.transaction_id],
    ['Transaction Amount', record.TransactionAmt ?? record.amount],
    ['ProductCD', record.ProductCD],
    ['Card Network', record.card_network],
    ['Card ID', record.card_id],
    ['isFraud (Dataset)', record.isFraud],
    ['Timestamp', record.TransactionDT ?? record.timestamp],
    ['Status', record.status],
  ];
  keyEntries.forEach(([label, value]) => keyFields.appendChild(buildFieldCard(label, value)));
  keyBlock.appendChild(keyFields);

  const analysisBlock = document.createElement('div');
  analysisBlock.className = 'detail-block';
  const analysisTitle = document.createElement('h4');
  analysisTitle.textContent = 'Analysis summary';
  analysisBlock.appendChild(analysisTitle);

  const analysisFields = document.createElement('div');
  analysisFields.className = 'detail-grid';
  const analysisEntries = [
    ['Fraud score', `${Math.round(Number(record.fraud_score || 0))}%`],
    ['Risk tier', riskTier(Number(record.fraud_score || 0))],
    ['Fraud label', record.fraud_label],
    ['Fraud type', record.fraud_type],
    ['Probability', record.raw_probability != null ? Number(record.raw_probability).toFixed(6) : '-'],
    ['Latency', record.latency_ms != null ? `${record.latency_ms} ms` : '-'],
  ];
  analysisEntries.forEach(([label, value]) => analysisFields.appendChild(buildFieldCard(label, value)));
  analysisBlock.appendChild(analysisFields);

  transactionFields.append(keyBlock, analysisBlock);
}

function syncTransactionHeader(record) {
  if (!record) return;
  const score = Number(record.fraud_score || 0);
  const label = record.fraud_label || riskLabel(score);
  const status = String(record.status || 'ACTIVE').toUpperCase();
  const tier = riskTier(score);
  const transactionId = record.TransactionID || record.transaction_id || '-';

  if (transactionTitle) transactionTitle.textContent = `Transaction ${transactionId}`;
  if (transactionScore) transactionScore.textContent = `${Math.round(score)}%`;
  if (transactionLabel) transactionLabel.textContent = label;
  if (transactionStatus) {
    transactionStatus.textContent = status;
    transactionStatus.className = `status-pill status-${statusBand(status)}`;
  }
  if (transactionRisk) transactionRisk.textContent = tier;
  if (transactionIdText) transactionIdText.textContent = transactionId;
  if (transactionStatusText) {
    transactionStatusText.textContent = status;
    transactionStatusText.className = `status-pill status-${statusBand(status)}`;
  }
  if (fraudLabelText) fraudLabelText.textContent = label;
  if (fraudTypeText) fraudTypeText.textContent = record.fraud_type || 'Suspicious';

  updateGauge(score);
  updateModelBars(record.model_scores || {});
  renderTransactionFields(record);
}

async function updateStatus(action) {
  const transactionId = transactionData?.TransactionID || transactionData?.transaction_id;
  if (!transactionId) return;

  const response = await fetch('/update_status', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transaction_id: transactionId, action }),
  });

  const data = await readApiJSON(response);
  if (!response.ok) {
    throw new Error(data.error || 'Unable to update status');
  }

  transactionData.status = data.status || transactionData.status;
  syncTransactionHeader({ ...transactionData, status: transactionData.status });
}

function bindRowNavigation() {
  const rows = document.querySelectorAll('#history-body tr[data-transaction-id]');
  rows.forEach((row) => {
    row.addEventListener('click', () => {
      const transactionId = row.dataset.transactionId;
      if (transactionId) {
        window.location.href = `/transaction/${encodeURIComponent(transactionId)}`;
      }
    });
  });
}

function initDashboardSearch() {
  if (!searchForm) return;

  searchForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const transactionId = String(searchInput?.value || '').trim();
    if (!transactionId) {
      if (searchMessage) searchMessage.textContent = 'Enter a TransactionID.';
      return;
    }

    if (searchMessage) searchMessage.textContent = 'Opening transaction...';

    try {
      const response = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_id: transactionId }),
      });

      const data = await readApiJSON(response);
      if (!response.ok) {
        throw new Error(data.error || 'Transaction not found');
      }

      if (response.redirected || response.url) {
        window.location.href = response.url;
        return;
      }

      if (data.redirect_url) {
        window.location.href = data.redirect_url;
        return;
      }

      window.location.href = `/transaction/${encodeURIComponent(transactionId)}`;
    } catch (error) {
      if (searchMessage) searchMessage.textContent = error.message;
    }
  });
}

function initTransactionPage() {
  if (!transactionData) return;
  const seeded = { ...transactionData };
  if (typeof FRAUD_SCORE !== 'undefined' && seeded.fraud_score == null) {
    seeded.fraud_score = Number(FRAUD_SCORE);
  }
  syncTransactionHeader(seeded);

  actionButtons.safe?.addEventListener('click', () => updateStatus('SAFE').catch((error) => {
    if (transactionMessage) transactionMessage.textContent = error.message;
  }));
  actionButtons.flag?.addEventListener('click', () => updateStatus('FLAG').catch((error) => {
    if (transactionMessage) transactionMessage.textContent = error.message;
  }));
  actionButtons.review?.addEventListener('click', () => updateStatus('REVIEW').catch((error) => {
    if (transactionMessage) transactionMessage.textContent = error.message;
  }));
  actionButtons.block?.addEventListener('click', () => updateStatus('BLOCK').catch((error) => {
    if (transactionMessage) transactionMessage.textContent = error.message;
  }));
}

function ensureEmptyState() {
  if (!historyBody) return;
  if (!historyBody.querySelector('tr')) {
    const row = document.createElement('tr');
    row.innerHTML = '<td colspan="9">No transactions yet</td>';
    historyBody.appendChild(row);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  bindRowNavigation();
  initDashboardSearch();
  initTransactionPage();
  ensureEmptyState();
});
