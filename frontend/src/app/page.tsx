import ChartingDashboard from './components/ChartingDashboard'; // Make sure this path is correct

// Main page for the route / in any Next.js (App Router) project. Next.js automatically wires this page to the URL http://localhost:3000/
// The component runs server-side by default unless  mark it as "use client".
// Didn't mark it "use client" → that’s fine.
// Why? It's a mostly static layout; only child components that need client-side interactivity must include "use client" themselves.
// My ChartingDashboard does include "use client" at its top. So everything is perfect.

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-10 bg-gray-50">
      <header className="w-full max-w-7xl mb-10">
        <h1 className="text-4xl font-extrabold text-gray-900">Trading Agent Dashboard</h1>
        <p className="text-lg text-gray-600">Visualization of Market Data and Trading Policy</p>
      </header>

      <div className="w-full max-w-7xl">
        <ChartingDashboard />
        

      </div>
    </main>
  );
}