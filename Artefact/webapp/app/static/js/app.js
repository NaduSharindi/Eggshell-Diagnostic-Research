/* Minimal progressive enhancement. The application is fully usable with JavaScript disabled;
   this only improves feedback during the upload, which on a large tray can take a minute. */
(function () {
  "use strict";

  document.querySelectorAll("form[data-busy]").forEach(function (form) {
    form.addEventListener("submit", function () {
      var btn = form.querySelector("button[type=submit]");
      if (!btn) return;
      var files = form.querySelector("input[type=file]");
      var n = files && files.files ? files.files.length : 0;
      btn.disabled = true;
      btn.textContent = n > 1 ? "Analysing " + n + " images…" : "Analysing…";
    });
  });

  // show how many files were chosen, so a mis-selected folder is obvious before submitting
  document.querySelectorAll('input[type=file][multiple]').forEach(function (inp) {
    inp.addEventListener("change", function () {
      var small = inp.parentElement.querySelector("small");
      if (small && inp.files) {
        small.textContent = inp.files.length + " image" +
          (inp.files.length === 1 ? "" : "s") + " selected.";
      }
    });
  });
})();
