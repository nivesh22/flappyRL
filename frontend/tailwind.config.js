/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter"', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
        pixel: ['"Press Start 2P"', 'monospace'],
      },
      colors: {
        brand: {
          DEFAULT: '#F97316',
          dark: '#FB923C',
        },
      },
      keyframes: {
        'flappy-bob': {
          '0%, 100%': { transform: 'translateY(-2px)' },
          '50%': { transform: 'translateY(2px)' },
        },
      },
      animation: {
        'flappy-bob': 'flappy-bob 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
