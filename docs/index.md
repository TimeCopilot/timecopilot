---
hide:
  - toc
  - navigation
---


<style>
  .md-typeset h1,
  .md-content__button {
    display: none;
  }
</style>

--8<-- "README.md"

<div class="newsletter-signup">
  <h2>Stay Updated</h2>
  <p>Get updates on new features and forecasting insights. No spam, unsubscribe anytime.</p>
  <form class="newsletter-form" id="newsletter-form">
    <input type="email" name="email" placeholder="Your email address" required />
    <button type="submit">Subscribe</button>
  </form>
  <p class="newsletter-thanks" id="newsletter-thanks" style="display:none;">Thanks for subscribing!</p>
</div>

<script>
document.getElementById("newsletter-form").addEventListener("submit", function(e) {
  e.preventDefault();
  var form = e.target;
  fetch("https://formspree.io/f/xvzbzjkk", {
    method: "POST",
    body: new FormData(form),
    headers: { "Accept": "application/json" }
  }).then(function(response) {
    if (response.ok) {
      form.style.display = "none";
      document.getElementById("newsletter-thanks").style.display = "block";
    }
  });
});
</script>
