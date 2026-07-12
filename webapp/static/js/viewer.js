// Mol* overlay: wild-type (steel) + design (rose), superposed.
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
      // wild-type in steel blue
      await viewer.loadStructureFromUrl(wtUrl, "pdb", false, {
        representationParams: { theme: { color: "uniform",
          colorParams: { value: 0x3b6a80 } } } });
      // design in rose, superposed onto WT
      await viewer.loadStructureFromUrl(designUrl, "pdb", false, {
        representationParams: { theme: { color: "uniform",
          colorParams: { value: 0xc34c62 } } } });
    };
    if (parent) parent.addEventListener("shown.bs.collapse", init, { once: true });
    else init();
  }
})();
