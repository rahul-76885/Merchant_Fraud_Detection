/* FraudShield AI | login.js */

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const t = document.querySelector(a.getAttribute('href'));
    if (t) { e.preventDefault(); t.scrollIntoView({ behavior: 'smooth' }); }
  });
});

// Staggered reveal on scroll
const obs = new IntersectionObserver(entries => {
  entries.forEach(en => {
    if (en.isIntersecting) { en.target.style.opacity='1'; en.target.style.transform='translateY(0)'; obs.unobserve(en.target); }
  });
}, { threshold: 0.1 });
document.querySelectorAll('.fc, .sc, .ps').forEach((el, i) => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(18px)';
  el.style.transition = `opacity 0.45s ease ${i * 0.055}s, transform 0.45s ease ${i * 0.055}s`;
  obs.observe(el);
});

// Form validation
const form = document.querySelector('form');
if (form) {
  form.addEventListener('submit', e => {
    let ok = true;
    form.querySelectorAll('.finp').forEach(inp => {
      inp.classList.remove('err');
      if (!inp.value.trim()) { inp.classList.add('err'); ok = false; }
    });
    if (!ok) e.preventDefault();
  });
  form.querySelectorAll('.finp').forEach(inp => inp.addEventListener('input', () => inp.classList.remove('err')));
}
