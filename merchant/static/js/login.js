const inputs = document.querySelectorAll('.login-form input');

inputs.forEach((input) => {
  input.addEventListener('focus', () => input.parentElement?.classList.add('focus'));
  input.addEventListener('blur', () => input.parentElement?.classList.remove('focus'));
});

window.addEventListener('load', () => {
  document.querySelectorAll('.stat-card').forEach((card, index) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(12px)';
    setTimeout(() => {
      card.style.transition = 'opacity 420ms ease, transform 420ms ease';
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    }, 120 + index * 80);
  });
});
