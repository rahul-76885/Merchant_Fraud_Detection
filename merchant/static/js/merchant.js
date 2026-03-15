/* FraudShield AI | merchant.js
   FRAUD_SCORE, IS_FRAUD, RISK_TIER injected by Jinja2 in <script> block */

// Gauge animation — animate from 0 to FRAUD_SCORE on load
window.addEventListener('load', () => {
  const arc = document.getElementById('gauge-arc');
  const scoreEl = document.getElementById('gauge-score-display');
  if (!arc) return;

  const circumference = 452.4;
  const finalOffset = circumference * (1 - FRAUD_SCORE / 100);

  // Start from empty
  arc.setAttribute('stroke-dashoffset', circumference);
  arc.style.transition = 'none';

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      arc.style.transition = 'stroke-dashoffset 1.8s cubic-bezier(0.4, 0, 0.2, 1)';
      arc.setAttribute('stroke-dashoffset', finalOffset);
    });
  });

  // Animate score number
  const target = FRAUD_SCORE;
  const duration = 1800;
  const start = performance.now();
  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }
  function animScore(now) {
    const p = Math.min((now - start) / duration, 1);
    const cur = Math.round(easeOut(p) * target);
    if (scoreEl) {
      const unit = scoreEl.querySelector('.g-unit');
      scoreEl.textContent = cur;
      if (unit) scoreEl.appendChild(unit);
      else { const s = document.createElement('span'); s.className = 'g-unit'; s.textContent = '%'; scoreEl.appendChild(s); }
    }
    if (p < 1) requestAnimationFrame(animScore);
  }
  requestAnimationFrame(animScore);
});

// Mini stat bars
document.querySelectorAll('.ms-fill').forEach(bar => {
  const w = bar.style.width; bar.style.width = '0%';
  setTimeout(() => { bar.style.transition = 'width 1s ease 0.2s'; bar.style.width = w; }, 200);
});

// Model bars
document.querySelectorAll('.mb-fill').forEach(bar => {
  const w = bar.style.width; bar.style.width = '0%';
  setTimeout(() => { bar.style.transition = 'width 1.2s cubic-bezier(.4,0,.2,1) 0.4s'; bar.style.width = w; }, 200);
});

// Indicator conf bars
document.querySelectorAll('.cfill').forEach(bar => {
  const w = bar.style.width; bar.style.width = '0%';
  setTimeout(() => { bar.style.transition = 'width 1s ease 0.6s'; bar.style.width = w; }, 200);
});

// Action button confirm dialogs
document.querySelectorAll('.act-btn').forEach(btn => {
  btn.addEventListener('click', function (e) {
    const action = this.closest('form')?.querySelector('input[name="action"]')?.value;
    const msgs = {
      block:  '⚠️ Block this merchant? All transactions will be suspended.',
      refund: '💳 Issue a refund to the customer for this transaction?',
      alert:  '📨 Send an investigation alert to the compliance team?'
    };
    if (action && msgs[action] && !confirm(msgs[action])) e.preventDefault();
  });
});

// Card reveal on scroll
const revObs = new IntersectionObserver(entries => {
  entries.forEach(en => {
    if (en.isIntersecting) { en.target.style.opacity='1'; en.target.style.transform='translateY(0)'; revObs.unobserve(en.target); }
  });
}, { threshold: 0.07 });
document.querySelectorAll('.card, .ms').forEach((el, i) => {
  el.style.opacity = '0'; el.style.transform = 'translateY(14px)';
  el.style.transition = `opacity 0.4s ease ${i * 0.06}s, transform 0.4s ease ${i * 0.06}s`;
  revObs.observe(el);
});

// History table row reveal
document.querySelectorAll('.hist-tbl tbody tr').forEach((row, i) => {
  row.style.opacity = '0';
  row.style.transition = `opacity 0.3s ease ${i * 0.05}s`;
  setTimeout(() => { row.style.opacity = '1'; }, 900 + i * 50);
});
