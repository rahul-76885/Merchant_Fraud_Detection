/* FraudShield AI | dashboard.js */

// Live clock
const clockEl = document.getElementById('live-clock');
function tick() {
  if (clockEl) clockEl.textContent = new Date().toLocaleTimeString('en-GB');
}
setInterval(tick, 1000); tick();

// Filter buttons
document.querySelectorAll('.tf').forEach(btn => {
  btn.addEventListener('click', function () {
    document.querySelectorAll('.tf').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
    const f = this.dataset.f;
    document.querySelectorAll('#txn-tbody tr').forEach(row => {
      row.style.display = f === 'all' ? '' : row.dataset.fraud === f ? '' : 'none';
    });
  });
});

// Search — demo mode (remove preventDefault when Flask route is live)
const searchForm = document.getElementById('search-form');
const demoResult = document.getElementById('demo-result');
if (searchForm && demoResult) {
  searchForm.addEventListener('submit', e => {
    const val = document.getElementById('txn-input').value.trim();
    if (val.length > 2) {
      e.preventDefault(); // REMOVE THIS LINE when url_for('search_transaction') is set on the form action
      demoResult.classList.add('show');
      demoResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  });
}

// Staggered table row reveal
document.querySelectorAll('#txn-tbody tr').forEach((row, i) => {
  row.style.opacity = '0';
  row.style.transform = 'translateX(-6px)';
  row.style.transition = `opacity 0.3s ease ${i * 0.04}s, transform 0.3s ease ${i * 0.04}s`;
  setTimeout(() => { row.style.opacity = '1'; row.style.transform = 'none'; }, 80 + i * 40);
});
