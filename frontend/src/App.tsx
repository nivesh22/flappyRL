import React, { useEffect, useState } from 'react';
import {
  Play,
  Trophy,
  BarChart3,
  Cpu,
  BookOpen,
  Sun,
  Moon,
  Monitor,
  Github,
} from 'lucide-react';
import { PlaygroundPage } from './pages/PlaygroundPage';
import { ModelArenaPage } from './pages/ModelArenaPage';
import { TrainingAnalyticsPage } from './pages/TrainingAnalyticsPage';
import { ModelExplorerPage } from './pages/ModelExplorerPage';
import { LearnRlPage } from './pages/LearnRlPage';

type TabId = 'playground' | 'arena' | 'analytics' | 'explorer' | 'learn';
type ThemeMode = 'light' | 'dark' | 'system';

const NAV_TABS: {
  id: TabId;
  label: string;
  icon: React.FC<{ className?: string }>;
}[] = [
  { id: 'playground', label: 'Playground', icon: Play },
  { id: 'arena', label: 'Arena', icon: Trophy },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'explorer', label: 'Explorer', icon: Cpu },
  { id: 'learn', label: 'Learn RL', icon: BookOpen },
];

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('playground');
  const [themeMode, setThemeMode] = useState<ThemeMode>('light');

  useEffect(() => {
    const applyResolvedTheme = (mode: ThemeMode) => {
      if (mode === 'system') {
        const prefersDark =
          typeof window !== 'undefined' &&
          window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.documentElement.setAttribute(
          'data-theme',
          prefersDark ? 'dark' : 'light'
        );
      } else {
        document.documentElement.setAttribute('data-theme', mode);
      }
    };

    applyResolvedTheme(themeMode);

    if (themeMode === 'system' && typeof window !== 'undefined') {
      const mql = window.matchMedia('(prefers-color-scheme: dark)');
      const handler = () => applyResolvedTheme('system');
      mql.addEventListener('change', handler);
      return () => mql.removeEventListener('change', handler);
    }
  }, [themeMode]);

  return (
    <div className="min-h-screen w-full flex flex-col bg-[var(--bg-page)] text-[var(--text-primary)]">
      {/* Clean Modern AI Platform Header (Compact h-12 to maximize simultaneous viewport space) */}
      <header className="sticky top-0 z-30 w-full bg-[var(--bg-card)]/95 backdrop-blur border-b border-[var(--border-subtle)]">
        <div className="max-w-[1680px] mx-auto px-4 sm:px-6 h-12 flex items-center justify-between gap-4">
          {/* Brand Identity + Subtle Tagline */}
          <div className="flex items-center gap-2.5 shrink-0">
            <img
              src="/favicon.svg"
              alt="FlappyRL logo"
              className="w-7 h-7 rounded-md select-none"
            />
            <div className="flex items-baseline gap-2">
              <span className="text-base font-bold tracking-tight text-[var(--text-primary)]">
                Flappy<span className="text-[var(--accent-primary)]">RL</span>
              </span>
              <span className="hidden xl:inline-block text-xs font-medium text-[var(--text-secondary)]">
                From random to mastery
              </span>
            </div>
          </div>

          {/* Primary Navigation (5 Clean Tabs) */}
          <nav
            aria-label="Primary navigation"
            className="flex items-center gap-1 overflow-x-auto py-1"
          >
            {NAV_TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-colors cursor-pointer whitespace-nowrap ${
                    isActive
                      ? 'bg-[var(--accent-primary-soft)] text-[var(--accent-primary)] font-semibold'
                      : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Right Controls: Compact Light / Dark / System Theme Switch + Standard GitHub Link */}
          <div className="flex items-center gap-2.5 shrink-0">
            <div
              className="ui-segmented"
              role="group"
              aria-label="Theme preference"
            >
              <button
                type="button"
                onClick={() => setThemeMode('light')}
                title="Light theme"
                aria-label="Light theme"
                className={`ui-segmented-item h-7 px-2 ${
                  themeMode === 'light' ? 'ui-segmented-item-active' : ''
                }`}
              >
                <Sun className="w-3.5 h-3.5" />
                <span className="hidden sm:inline text-xs">Light</span>
              </button>
              <button
                type="button"
                onClick={() => setThemeMode('dark')}
                title="Dark theme"
                aria-label="Dark theme"
                className={`ui-segmented-item h-7 px-2 ${
                  themeMode === 'dark' ? 'ui-segmented-item-active' : ''
                }`}
              >
                <Moon className="w-3.5 h-3.5" />
                <span className="hidden sm:inline text-xs">Dark</span>
              </button>
              <button
                type="button"
                onClick={() => setThemeMode('system')}
                title="System theme"
                aria-label="System theme"
                className={`ui-segmented-item h-7 px-2 ${
                  themeMode === 'system' ? 'ui-segmented-item-active' : ''
                }`}
              >
                <Monitor className="w-3.5 h-3.5" />
                <span className="hidden md:inline text-xs">System</span>
              </button>
            </div>

            <a
              href="https://github.com/nivesh22/flappyRL"
              target="_blank"
              rel="noreferrer"
              aria-label="View FlappyRL source on GitHub"
              title="View source on GitHub"
              className="ui-btn ui-btn-secondary h-8 px-2.5"
            >
              <Github className="w-4 h-4" />
              <span className="hidden lg:inline text-xs font-medium">GitHub</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 w-full max-w-[1680px] mx-auto px-3 sm:px-5 py-3">
        {activeTab === 'playground' && <PlaygroundPage />}
        {activeTab === 'arena' && <ModelArenaPage />}
        {activeTab === 'analytics' && <TrainingAnalyticsPage />}
        {activeTab === 'explorer' && <ModelExplorerPage />}
        {activeTab === 'learn' && <LearnRlPage />}
      </main>
    </div>
  );
};

export default App;
