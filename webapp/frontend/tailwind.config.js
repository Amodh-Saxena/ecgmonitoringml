/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#F6F8FA",
        surface: "#FFFFFF",
        border: "#D8DEE4",
        ink: "#0F1B2D",
        muted: "#5A6878",
        primary: {
          DEFAULT: "#0B5C8F",
          hover: "#084770",
          light: "#E7F2F9",
        },
        status: {
          confident: "#136B4B",
          "confident-bg": "#E8F5EE",
          uncertain: "#8A5300",
          "uncertain-bg": "#FEF7E6",
          rejected: "#A32A20",
          "rejected-bg": "#FDF2F0",
        },
        ecg: {
          minor: "#F4DCDC",
          major: "#E9B8B8",
        },
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
      borderRadius: {
        lg: "8px",
        xl: "12px",
      },
    },
  },
  plugins: [],
}
