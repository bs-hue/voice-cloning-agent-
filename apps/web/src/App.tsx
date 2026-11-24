import { useEffect, useState } from "react";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { AudioLines, Mic2, History, Gauge } from "lucide-react";
import { SwaraMark } from "./components/SwaraMark";
import { api, type HealthStatus } from "./lib/api";
import { StudioPage } from "./pages/StudioPage";
import { VoicesPage } from "./pages/VoicesPage";
import { HistoryPage } from "./pages/HistoryPage";
import { SetupPage } from "./pages/SetupPage";

export function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    api.getHealth().then(setHealth).catch(() => {});
  }, []);

  return (
    <BrowserRouter>
      <div className="app-shell">
        <header className="topbar">
          <div className="topbar-inner">
            <NavLink to="/" className="brand-lockup">
              <SwaraMark />
              <div>
                <div className="brand-title">SwaraLab</div>
                <div className="brand-sub">Offline English &amp; Hindi Voice Studio</div>
              </div>
            </NavLink>

            <nav className="nav-tabs" aria-label="Primary sections">
              <NavLink to="/" end className={({ isActive }) => `nav-tab${isActive ? " active" : ""}`}>
                <AudioLines size={15} aria-hidden="true" />
                Studio
              </NavLink>
              <NavLink to="/voices" className={({ isActive }) => `nav-tab${isActive ? " active" : ""}`}>
                <Mic2 size={15} aria-hidden="true" />
                Voice Profiles
              </NavLink>
              <NavLink to="/history" className={({ isActive }) => `nav-tab${isActive ? " active" : ""}`}>
                <History size={15} aria-hidden="true" />
                History
              </NavLink>
              <NavLink to="/setup" className={({ isActive }) => `nav-tab${isActive ? " active" : ""}`}>
                <Gauge size={15} aria-hidden="true" />
                Engine &amp; Storage
              </NavLink>
            </nav>

            <div className="status-pill-bar">
              <span className={`badge ${health?.model.cached_offline ? "badge-success" : ""}`}>
                {health?.model.cached_offline ? "V2 Offline Ready" : "Checking Engine"}
              </span>
              {health?.storage.e_drive && (
                <span className="badge mono">
                  E: {health.storage.e_drive.free_gb} GB free
                </span>
              )}
            </div>
          </div>
        </header>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<StudioPage />} />
            <Route path="/voices" element={<VoicesPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/setup" element={<SetupPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;

