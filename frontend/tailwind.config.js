/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        court: "#0b0e14",
        panel: "#141922",
        panel2: "#1b2230",
        teamA: "#28b4eb",
        teamB: "#eb3c3c",
      },
    },
  },
  plugins: [],
};
