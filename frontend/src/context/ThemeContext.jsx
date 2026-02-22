/**
 * ThemeContext - Light theme only. Dark mode removed.
 */

import React, { createContext, useContext } from 'react';

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  // Ensure the dark class is never on <html>
  if (typeof document !== 'undefined') {
    document.documentElement.classList.remove('dark');
  }

  return (
    <ThemeContext.Provider value={{ darkMode: false, loadThemeFromServer: async () => {} }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within ThemeProvider');
  return context;
}
