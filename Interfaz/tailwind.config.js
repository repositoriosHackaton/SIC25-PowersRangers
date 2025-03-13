/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'primary-100': 'var(--primary-100)',
        'primary-200': 'var(--primary-200)',
        'primary-300': 'var(--primary-300)',
        'accent-100': 'var(--accent-100)',
        'accent-200': 'var(--accent-200)',
        'text-100': 'var(--text-100)',
        'text-200': 'var(--text-200)',
        'bg-100': 'var(--bg-100)',
        'bg-200': 'var(--bg-200)',
        'bg-300': 'var(--bg-300)',
      },
      boxShadow: {
        'md': '0 4px 10px var(--shadow-color)',
        'lg': '0 10px 20px var(--shadow-color)',
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
      },
    },
  },
  plugins: [],
};