// Mol* overlay: wild-type (amber) + design (magma), superposed.
// Each .viewer-box carries data-wt and data-design URLs (PDB).
(async function () {
  const boxes = document.querySelectorAll(".viewer-box");
  for (const box of boxes) {
    const wtUrl = box.dataset.wt, designUrl = box.dataset.design;
    if (!wtUrl || !designUrl) continue;
    // lazy-init when the accordion panel opens (Mol* needs a visible container)
    const parent = box.closest(".accordion-collapse");
    const init = async () => {
      if (box.dataset.loaded) return;
      box.dataset.loaded = "1";
      const viewer = await molstar.Viewer.create(box, {
        layoutIsExpanded: false, layoutShowControls: false,
        layoutShowSequence: false, layoutShowLog: false,
      });
      // wild-type in amber (reference)
      await viewer.loadStructureFromUrl(wtUrl, "pdb", false, {
        representationParams: { theme: { color: "uniform",
          colorParams: { value: 0xf6b23c } } } });
      // design in magma, superposed onto WT
      await viewer.loadStructureFromUrl(designUrl, "pdb", false, {
        representationParams: { theme: { color: "uniform",
          colorParams: { value: 0x5b1226 } } } });
      // Mol* sizes its canvas from the container; nudge it to recompute now that
      // the accordion panel is laid out, so it fits the box instead of the window.
      try { viewer.plugin.layout.events.updated.next(); } catch (e) {}
      window.dispatchEvent(new Event("resize"));
    };
    if (parent) parent.addEventListener("shown.bs.collapse", init, { once: true });
    else init();
  }
})();
