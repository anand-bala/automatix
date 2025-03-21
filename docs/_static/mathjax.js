window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {
      "[+]": [
        "color",
        "ams",
        "mathtools",
      ]
    },
    macros: {
      // -- Numeric Sets
      Naturals: "{\\mathbb{N}}",
      Integers: "{\\mathbb{Z}}",
      Rationals: "{\\mathbb{Q}}",
      Reals: "{\\mathbb{R}}",

      // -- Topological Sets
      Universal: "{\\mathbb{U}}",

      // -- Typography
      Ae: "{\\mathbb{A}}",
      Ce: "{\\mathbb{C}}",
      Be: "{\\mathbb{B}}",
      Ee: "{\\mathbb{E}}",
      Fe: "{\\mathbb{F}}",
      Ke: "{\\mathbb{K}}",
      Me: "{\\mathbb{M}}",
      Ne: "{\\mathbb{N}}",
      Oe: "{\\mathbb{O}}",
      Pe: "{\\mathbb{P}}",
      Qe: "{\\mathbb{Q}}",
      Re: "{\\mathbb{R}}",
      Se: "{\\mathbb{S}}",
      Te: "{\\mathbb{T}}",
      Ze: "{\\mathbb{Z}}",
      Ve: "{\\mathbb{V}}",
      Xe: "{\\mathbb{X}}",

      Ac: "{\\mathcal{A}}",
      Bc: "{\\mathcal{B}}",
      Dc: "{\\mathcal{D}}",
      Fc: "{\\mathcal{F}}",
      Ec: "{\\mathcal{E}}",
      Hc: "{\\mathcal{H}}",
      Ic: "{\\mathcal{I}}",
      Jc: "{\\mathcal{J}}",
      Kc: "{\\mathcal{K}}",
      Lc: "{\\mathcal{L}}",
      Mc: "{\\mathcal{M}}",
      Nc: "{\\mathcal{N}}",
      Oc: "{\\mathcal{O}}",
      Pc: "{\\mathcal{P}}",
      Qc: "{\\mathcal{Q}}",
      Rc: "{\\mathcal{R}}",
      Sc: "{\\mathcal{S}}",
      Tc: "{\\mathcal{T}}",
      Uc: "{\\mathcal{U}}",
      Vc: "{\\mathcal{V}}",
      Wc: "{\\mathcal{W}}",
      Xc: "{\\mathcal{X}}",
      Zc: "{\\mathcal{Z}}",

    },
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => { 
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()
  MathJax.typesetPromise()
})
