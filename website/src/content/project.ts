export type ProjectLink = {
  label: string;
  url: string | null;
  meta: string;
};

export type VisualAsset = {
  id: string;
  number: number;
  label: string;
  src: string;
  alt: string;
  caption: string;
  group?: string;
  paperUrl?: string;
};

export const project = {
  title: "TongGuOCR",
  subtitle: "A Layout-Aware and Token-Augmented OCR Framework for Chinese Historical Documents",
  institution: "South China University of Technology · Huawei Technologies Co., Ltd.",
  authors: [
    { name: "Zhongheng Zhou", marks: "†" },
    { name: "Yi Sun", marks: "†" },
    { name: "Huiguo He", marks: "*†" },
    { name: "Yuyi Zhang", marks: "" },
    { name: "Peirong Zhang", marks: "" },
    { name: "Yulin Fang", marks: "" },
    { name: "Dezhi Peng", marks: "" },
    { name: "Minghui Liao", marks: "" },
    { name: "Lianwen Jin", marks: "*" },
  ],
  authorNote: "* Corresponding authors · † Equal contribution",
  links: [
    { label: "Paper", url: "https://arxiv.org/abs/2608.07917", meta: "arXiv" },
    { label: "Demo", url: "https://guji-ocr.com", meta: "Open demo" },
    { label: "GitHub", url: "https://github.com/jzzh2004/TongGuOCR", meta: "Source code" },
    { label: "Hugging Face", url: null, meta: "Coming soon" },
  ] satisfies ProjectLink[],
  abstract:
    "Chinese historical documents preserve valuable cultural heritage, but many collections remain accessible only as scanned page images. This format limits full-text retrieval, collation, and computational analysis. Optical character recognition (OCR) can convert these images into machine-readable text, but accurate transcription remains challenging because historical documents often contain complex layouts, rare characters, and nontrivial reading orders. We propose TongGuOCR, a layout-aware and token-augmented OCR framework for Chinese historical documents. Its Layout-Aware Preprocessing module constructs and refines locally coherent recognition blocks to preserve local context while reducing interference across regions. Its Token-Augmented Recognition module uses character-level vocabulary expansion to represent rare characters and line-to-line transition modeling to provide spatial guidance during decoding. Experiments on two Chinese historical document OCR benchmarks show that TongGuOCR outperforms representative traditional OCR models, multimodal large language models (MLLMs), and OCR-oriented systems. Ablation studies confirm the contribution of each design.",
  metrics: [
    { value: "93.76", label: "Accuracy Rate", short: "AR ↑" },
    { value: "94.46", label: "Correct Rate", short: "CR ↑" },
    { value: "3.49", label: "Reading-order Edit Distance", short: "RO-ED ↓" },
  ],
  highlights: [
    {
      number: "01",
      title: "Layout-Aware Preprocessing",
      text: "Builds ordered, locally coherent recognition blocks that retain useful context while keeping historical characters legible.",
    },
    {
      number: "02",
      title: "Token-Augmented Recognition",
      text: "Expands rare characters into single tokens and encodes line-to-line transitions to guide complex reading paths.",
    },
    {
      number: "03",
      title: "Three-Stage Adaptation",
      text: "Progressively aligns historical-domain knowledge, high-quality supervision, and block-level inference granularity.",
    },
  ],
} as const;

export const figures: VisualAsset[] = [
  {
    id: "figure-1",
    number: 1,
    label: "Challenge & recognition comparison",
    src: "paper-assets/figures/figure-1.webp",
    alt: "Historical Chinese document page with reading-order paths and OCR transcriptions from multiple systems.",
    caption:
      "Recognition results illustrating the challenges of Chinese historical document OCR. The example contains a table-like layout, a nontrivial reading order, and rare historical characters. TongGuOCR follows the natural reading sequence and accurately transcribes the selected content in this case.",
  },
  {
    id: "figure-2",
    number: 2,
    label: "Framework overview",
    src: "paper-assets/figures/figure-2.png",
    alt: "TongGuOCR pipeline from document image through layout-aware preprocessing and token-augmented recognition.",
    caption:
      "Overview of TongGuOCR. Layout-Aware Preprocessing obtains text-line boxes, constructs recognition blocks, and refines them into OCR-friendly crops. Token-Augmented Recognition then combines character-level vocabulary expansion with line-to-line transition modeling.",
  },
  {
    id: "figure-3",
    number: 3,
    label: "Token target construction",
    src: "paper-assets/figures/figure-3.png",
    alt: "Construction of an augmented OCR target sequence using ordinary text, expanded character, and transition tokens.",
    caption:
      "Illustration of the Token-Augmented Recognition target construction. The transcription contains ordinary text tokens, expanded character tokens, and line-to-line transition tokens that encode coarse spatial displacement.",
  },
  {
    id: "figure-4",
    number: 4,
    label: "Qualitative comparison",
    src: "paper-assets/figures/figure-4.webp",
    alt: "Qualitative OCR comparison on a dense table-like M5HisDoc page, including reading paths and transcriptions.",
    caption:
      "Qualitative comparison on a challenging M5HisDoc page with a dense table-like layout and nontrivial transitions. TongGuOCR recovers all target text lines, follows the natural reading sequence, and accurately transcribes the corresponding content.",
  },
];

export const tables: VisualAsset[] = [
  {
    id: "table-1",
    number: 1,
    label: "Dataset annotation & split usage",
    group: "Dataset & Training",
    src: "paper-assets/tables/table-1.png",
    alt: "Dataset annotation sources and train, validation, and test split usage for HisDoc1B, MTHv2, and M5HisDoc.",
    caption: "Dataset annotation source and split usage. SFT, Select, and Eval denote supervised fine-tuning, checkpoint selection, and final evaluation.",
  },
  {
    id: "table-2",
    number: 2,
    label: "Three-stage training settings",
    group: "Dataset & Training",
    src: "paper-assets/tables/table-2.png",
    alt: "Training data, input granularity, sequence length, learning rate, and epochs for the three SFT stages.",
    caption: "Implementation settings of the three-stage training strategy. MTH denotes MTHv2, M5H denotes M5HisDoc, and H1B denotes HisDoc1B.",
  },
  {
    id: "table-3",
    number: 3,
    label: "Performance on MTHv2",
    group: "Main Results",
    src: "paper-assets/tables/table-3.png",
    alt: "Performance comparison between TongGuOCR and traditional, multimodal, end-to-end, and pipeline OCR models on MTHv2.",
    caption: "Performance comparison on MTHv2 across accuracy, edit distance, F1, precision, recall, BLEU, and reading-order metrics.",
    paperUrl: "https://arxiv.org/abs/2007.06890",
  },
  {
    id: "table-4",
    number: 4,
    label: "Performance on M5HisDoc",
    group: "Main Results",
    src: "paper-assets/tables/table-4.png",
    alt: "Performance comparison between TongGuOCR and representative OCR systems on the M5HisDoc benchmark.",
    caption: "Performance comparison on M5HisDoc across character recognition, sequence fidelity, and reading-order preservation metrics.",
    paperUrl: "https://papers.neurips.cc/paper_files/paper/2023/hash/f7b424d242cc6bb7708cff241367334d-Abstract-Datasets_and_Benchmarks.html",
  },
  {
    id: "table-5",
    number: 5,
    label: "Input-granularity alignment",
    group: "Ablation Studies",
    src: "paper-assets/tables/table-5.png",
    alt: "Ablation comparing page and block granularity during supervised fine-tuning and inference.",
    caption: "Effect of input-granularity alignment between supervised fine-tuning and inference on the M5HisDoc test split.",
  },
  {
    id: "table-6",
    number: 6,
    label: "Penalty-weight sensitivity",
    group: "Ablation Studies",
    src: "paper-assets/tables/table-6.png",
    alt: "Sensitivity analysis for block, adjacency, and page penalty weights in recognition block construction.",
    caption: "Sensitivity analysis of penalty weights in Recognition Block Construction on M5HisDoc.",
  },
  {
    id: "table-7",
    number: 7,
    label: "Token-augmentation ablation",
    group: "Ablation Studies",
    src: "paper-assets/tables/table-7.png",
    alt: "Ablation of character-level vocabulary expansion and line-to-line transition modeling.",
    caption: "Ablation of character-level vocabulary expansion and line-to-line transition modeling on the M5HisDoc test split.",
  },
  {
    id: "table-8",
    number: 8,
    label: "Character-alignment analysis",
    group: "Ablation Studies",
    src: "paper-assets/tables/table-8.png",
    alt: "Occurrence-level precision, recall, F1, false positive, and false negative analysis for added and non-added characters.",
    caption: "Occurrence-level character alignment analysis of token-augmented recognition for added and non-added character subsets.",
  },
  {
    id: "table-9",
    number: 9,
    label: "Staged SFT effectiveness",
    group: "Ablation Studies",
    src: "paper-assets/tables/table-9.png",
    alt: "Effect of coarse domain adaptation, high-quality refinement, and block-level inference alignment training stages.",
    caption: "Effect of staged supervised fine-tuning: coarse domain adaptation, high-quality page-level refinement, and block-level inference alignment.",
  },
];
