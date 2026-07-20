import { useEffect, useMemo, useRef, useState } from "react";
import { figures, project, tables, type VisualAsset } from "./content/project";

const navItems = [
  { id: "overview", label: "Overview" },
  { id: "method", label: "Method" },
  { id: "results", label: "Results" },
  { id: "tables", label: "Tables" },
];

function AssetImage({
  asset,
  onOpen,
  className = "",
}: {
  asset: VisualAsset;
  onOpen: (asset: VisualAsset, trigger: HTMLButtonElement) => void;
  className?: string;
}) {
  return (
    <figure className={`visual-card ${className}`}>
      <button
        className="visual-card__button"
        type="button"
        onClick={(event) => onOpen(asset, event.currentTarget)}
        aria-label={`Open ${asset.label} at full resolution`}
      >
        <img src={asset.src} alt={asset.alt} loading="lazy" />
        <span className="visual-card__hint" aria-hidden="true">
          View full resolution <span>↗</span>
        </span>
      </button>
      <figcaption>
        <span className="caption-label">{asset.id.replace("-", " ")}</span>
        <p>{asset.caption}</p>
      </figcaption>
    </figure>
  );
}

function ResourceLinks() {
  const iconFor = (label: string) => {
    if (label === "GitHub") return <img src="github-mark.svg" alt="" />;
    if (label === "Hugging Face") return <span className="resource-link__emoji">🤗</span>;
    if (label === "Paper") return <span className="resource-link__glyph">P</span>;
    return <span className="resource-link__glyph">↗</span>;
  };

  return (
    <div className="resource-links" aria-label="Project resources">
      {project.links.map((link) =>
        link.url ? (
          <a key={link.label} href={link.url} target="_blank" rel="noreferrer noopener" className="resource-link">
            <span className="resource-link__icon" aria-hidden="true">
              {iconFor(link.label)}
            </span>
            <span>
              <strong>{link.label}</strong>
              <small>{link.meta}</small>
            </span>
          </a>
        ) : (
          <button key={link.label} type="button" className="resource-link is-unavailable" title={`${link.label} — ${link.meta}`}>
            <span className="resource-link__icon" aria-hidden="true">
              {iconFor(link.label)}
            </span>
            <span>
              <strong>{link.label}</strong>
              <small>{link.meta}</small>
            </span>
          </button>
        ),
      )}
    </div>
  );
}

function Lightbox({ asset, onClose }: { asset: VisualAsset; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    const bodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "+" || event.key === "=") setZoom((value) => Math.min(2.5, value + 0.25));
      if (event.key === "-") setZoom((value) => Math.max(1, value - 0.25));
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = bodyOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [onClose]);

  return (
    <div className={`lightbox ${asset.id.startsWith("table-") ? "lightbox--table" : ""}`} role="dialog" aria-modal="true" aria-label={`${asset.label} full-resolution viewer`} onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <div className="lightbox__topbar">
        <div>
          <span>{asset.id.replace("-", " ")}</span>
          <strong>{asset.label}</strong>
        </div>
        <div className="lightbox__controls">
          <button type="button" onClick={() => setZoom((value) => Math.max(1, value - 0.25))} aria-label="Zoom out">
            −
          </button>
          <output aria-live="polite">{zoom === 1 ? "Fit" : `${Math.round(zoom * 100)}%`}</output>
          <button type="button" onClick={() => setZoom((value) => Math.min(2.5, value + 0.25))} aria-label="Zoom in">
            +
          </button>
          <button ref={closeRef} type="button" onClick={onClose} aria-label="Close image viewer" className="lightbox__close">
            Close ×
          </button>
        </div>
      </div>
      <div className={`lightbox__stage ${zoom > 1 ? "is-zoomed" : "is-fitted"}`}>
        <img
          src={asset.src}
          alt={asset.alt}
          className={zoom > 1 ? "is-zoomed" : "is-fitted"}
          style={zoom > 1 ? { width: `${zoom * 100}%` } : undefined}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [activeSection, setActiveSection] = useState("overview");
  const [lightboxAsset, setLightboxAsset] = useState<VisualAsset | null>(null);
  const [selectedAblationId, setSelectedAblationId] = useState("table-5");
  const lastTrigger = useRef<HTMLButtonElement | null>(null);

  const datasetTables = useMemo(() => tables.filter((table) => table.group === "Dataset & Training"), []);
  const mainResultTables = useMemo(() => tables.filter((table) => table.group === "Main Results"), []);
  const ablationTables = useMemo(() => tables.filter((table) => table.group === "Ablation Studies"), []);
  const selectedAblation = ablationTables.find((table) => table.id === selectedAblationId) ?? ablationTables[0];

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible?.target.id) setActiveSection(visible.target.id);
      },
      { rootMargin: "-28% 0px -58%", threshold: [0.05, 0.25, 0.5] },
    );
    navItems.forEach(({ id }) => {
      const section = document.getElementById(id);
      if (section) observer.observe(section);
    });
    return () => observer.disconnect();
  }, []);

  const openLightbox = (asset: VisualAsset, trigger: HTMLButtonElement) => {
    lastTrigger.current = trigger;
    setLightboxAsset(asset);
  };

  const closeLightbox = () => {
    setLightboxAsset(null);
    window.setTimeout(() => lastTrigger.current?.focus(), 0);
  };

  return (
    <>
      <a className="skip-link" href="#overview">
        Skip to content
      </a>
      <header className="site-header">
        <a className="brand" href="#overview" aria-label="TongGuOCR home">
          <img className="brand__logo" src="tonggu-logo.png" alt="" />
          <span className="brand__wordmark">TongGuOCR</span>
        </a>
        <nav aria-label="Primary navigation">
          {navItems.map((item) => (
            <a key={item.id} href={`#${item.id}`} aria-current={activeSection === item.id ? "location" : undefined}>
              {item.label}
            </a>
          ))}
        </nav>
        <span className="header-tag">Research project · 2026</span>
      </header>

      <main>
        <section className="hero section-shell" id="overview">
          <div className="hero__copy">
            <p className="eyebrow"><span>通古 OCR</span> Chinese Historical Document OCR</p>
            <h1>
              <span>{project.title}</span>
              {project.subtitle}
            </h1>
            <p className="hero__lede">
              Preserving <em>layout</em>, rare characters, and the natural reading order of Chinese historical documents.
            </p>
            <div className="authors" aria-label="Paper authors">
              {project.authors.map((author) => (
                <span key={author.name}>
                  {author.name}<sup>{author.marks}</sup>
                </span>
              ))}
            </div>
            <p className="institution">{project.institution}</p>
            <p className="author-note">{project.authorNote}</p>
            <ResourceLinks />
          </div>

          <div className="hero__visual">
            <AssetImage asset={figures[1]} onOpen={openLightbox} className="visual-card--hero" />
          </div>
        </section>

        <section className="overview-band section-shell" aria-labelledby="abstract-title">
          <div className="overview-band__abstract">
            <p className="kicker">Research premise</p>
            <h2 id="abstract-title">Reading the page as it was written.</h2>
            <p>{project.abstract}</p>
          </div>
          <div className="overview-band__highlights" aria-label="Three coordinated ideas">
            <p className="kicker">Three coordinated ideas</p>
            {project.highlights.map((highlight) => (
              <article key={highlight.number}>
                <span>{highlight.number}</span>
                <div>
                  <h3>{highlight.title}</h3>
                  <p>{highlight.text}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="challenge section-shell" aria-labelledby="challenge-title">
          <div className="challenge__copy">
            <p className="kicker">Problem setting</p>
            <h2 id="challenge-title">Historical pages resist flat OCR.</h2>
            <p>
              Dense layouts, rare characters, and nontrivial reading paths make page-level recognition a structural problem before they make it a decoding problem.
            </p>
          </div>
          <AssetImage asset={figures[0]} onOpen={openLightbox} className="visual-card--challenge" />
        </section>

        <section className="method section-shell" id="method" aria-labelledby="method-title">
          <div className="section-heading section-heading--dark">
            <div>
              <p className="kicker">Method overview</p>
              <h2 id="method-title">From a complex page to an ordered transcription.</h2>
            </div>
            <p>Recognition blocks preserve local context; augmented tokens make rare glyphs and spatial transitions explicit.</p>
          </div>
          <div className="method__rail" aria-label="TongGuOCR stages">
            <span>Page image</span><i>→</i><span>Ordered lines</span><i>→</i><span>Recognition blocks</span><i>→</i><span>Augmented sequence</span>
          </div>
          <div className="method__single">
            <AssetImage asset={figures[2]} onOpen={openLightbox} className="visual-card--dark" />
          </div>
        </section>

        <section className="results section-shell" id="results" aria-labelledby="results-title">
          <div className="section-heading">
            <div>
              <p className="kicker">Evaluation</p>
              <h2 id="results-title">Recognition accuracy and reading order improve together.</h2>
            </div>
            <p>Reported on the M5HisDoc test split using the full TongGuOCR model.</p>
          </div>
          <div className="results__dashboard">
            <div className="results__summary">
              <div className="metrics" aria-label="Key performance metrics">
                {project.metrics.map((metric) => (
                  <article key={metric.short}>
                    <span>{metric.short}</span>
                    <strong>{metric.value}</strong>
                    <p>{metric.label}</p>
                  </article>
                ))}
              </div>
              <div className="results__note">
              <p className="kicker">A page that resists shortcuts</p>
              <h3>Dense structure, rare symbols, nontrivial transitions.</h3>
              <p>
                In the qualitative example, TongGuOCR recovers every target line and follows the page’s natural sequence while competing systems accumulate ordering and character errors.
              </p>
              </div>
            </div>
            <AssetImage asset={figures[3]} onOpen={openLightbox} />
          </div>
        </section>

        <section className="table-section section-shell" id="tables" aria-labelledby="tables-title">
          <div className="section-heading">
            <div>
              <p className="kicker">Complete evidence</p>
              <h2 id="tables-title">All experiments, from setup to ablation.</h2>
            </div>
            <p>Nine tables reproduced directly from the paper. Select any table to inspect the original high-resolution rendering.</p>
          </div>
          <div className="table-group">
            <div className="table-group__title">
              <span>01</span>
              <h3>Dataset &amp; Training</h3>
              <p>2 setup tables</p>
            </div>
            <div className="table-grid table-grid--setup">
              {datasetTables.map((table) => (
                <AssetImage key={table.id} asset={table} onOpen={openLightbox} className="visual-card--table" />
              ))}
            </div>
          </div>

          <div className="table-group table-group--main">
            <div className="table-group__title">
              <span>02</span>
              <h3>Main Results</h3>
              <p>Both benchmark comparisons shown in full</p>
            </div>
            <div className="table-grid table-grid--main">
              {mainResultTables.map((table) => (
                <article className="main-result-card" key={table.id}>
                  <div className="dataset-paper-link">
                    <span>Dataset paper</span>
                    <a href={table.paperUrl} target="_blank" rel="noreferrer noopener">
                      {table.number === 3 ? "MTHv2" : "M5HisDoc"} <span aria-hidden="true">↗</span>
                    </a>
                  </div>
                  <AssetImage asset={table} onOpen={openLightbox} className="visual-card--table visual-card--main-table" />
                </article>
              ))}
            </div>
          </div>

          <div className="table-group table-group--ablation">
            <div className="table-group__title">
              <span>03</span>
              <h3>Ablation Studies</h3>
              <p>Choose one analysis to inspect</p>
            </div>
            <div className="ablation-browser">
              <div className="ablation-picker" role="tablist" aria-label="Choose an ablation study">
                {ablationTables.map((table) => (
                  <button
                    key={table.id}
                    type="button"
                    role="tab"
                    aria-selected={selectedAblation.id === table.id}
                    aria-controls="ablation-panel"
                    onClick={() => setSelectedAblationId(table.id)}
                  >
                    <span>Table {table.number}</span>
                    <strong>{table.label}</strong>
                  </button>
                ))}
              </div>
              <div className="ablation-panel" id="ablation-panel" role="tabpanel" aria-live="polite">
                <AssetImage key={selectedAblation.id} asset={selectedAblation} onOpen={openLightbox} className="visual-card--table" />
              </div>
            </div>
          </div>
        </section>

        <section className="closing section-shell" aria-labelledby="closing-title">
          <img className="closing__logo" src="tonggu-logo.png" alt="" />
          <div>
            <p className="kicker">Conclusion</p>
            <h2 id="closing-title">Turning scanned heritage into searchable knowledge.</h2>
            <p>
              TongGuOCR shows that faithful historical-document recognition benefits from coordinating layout structure, rare-character representation, and spatial transitions during transcription.
            </p>
          </div>
          <a href="#overview">Back to top ↑</a>
        </section>
      </main>

      <footer>
        <div className="footer__brand">
          <img className="brand__logo" src="tonggu-logo.png" alt="" />
          <strong>TongGuOCR</strong>
        </div>
        <p>School of Electronic and Information Engineering · South China University of Technology</p>
        <p>
          Project page inspired by <a href="https://dream-xyz.github.io/dreamstory" target="_blank" rel="noreferrer">DreamStory</a> and the <a href="https://github.com/eliahuhorwitz/Academic-project-page-template" target="_blank" rel="noreferrer">Academic Project Page Template</a>.
        </p>
      </footer>

      {lightboxAsset ? <Lightbox asset={lightboxAsset} onClose={closeLightbox} /> : null}
    </>
  );
}
