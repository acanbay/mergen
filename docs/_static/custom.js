/* Click-to-zoom for the figures on example pages (medium-zoom, MIT).
   ESC, a click, or a scroll closes the zoomed view. */
document.addEventListener("DOMContentLoaded", function () {
    if (typeof mediumZoom === "function") {
        mediumZoom(".sphx-glr-single-img", { margin: 24 });
    }
});
