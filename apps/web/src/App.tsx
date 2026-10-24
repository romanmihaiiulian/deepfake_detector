import { AnalysisPage } from "./pages/AnalysisPage";
import { UploadPage } from "./pages/UploadPage";
import { navigate, parseRoute, usePathname } from "./router";

export function App() {
  const route = parseRoute(usePathname());
  return (
    <>
      <header className="site-header">
        <a
          href="/"
          className="brand"
          onClick={(event) => {
            event.preventDefault();
            navigate("/");
          }}
        >
          <img src="/favicon.svg" alt="" width={28} height={28} />
          Deepfake Detector
        </a>
      </header>
      {route.name === "upload" && <UploadPage />}
      {route.name === "analysis" && <AnalysisPage key={route.id} id={route.id} />}
      {route.name === "notFound" && (
        <main className="page">
          <h1>Page not found</h1>
          <button type="button" className="button primary" onClick={() => navigate("/")}>
            Go to the upload page
          </button>
        </main>
      )}
      <footer className="site-footer">
        Probabilistic assessment only. Open source (Apache-2.0).
      </footer>
    </>
  );
}
