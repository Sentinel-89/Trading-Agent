import ChartingDashboard from './components/ChartingDashboard'; // Make sure this path is correct

// Main page for the route / in any Next.js (App Router) project. Next.js automatically wires this page to the URL http://localhost:3000/
// The component runs server-side by default unless you mark it as "use client".
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
        
        {/* ... (Other placeholder sections) ... */}
        <section className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-4 bg-white shadow rounded-lg">
            <h3 className="font-semibold text-lg">Agent Status</h3>
            <p className="text-gray-500">Ready for simulation...</p>
          </div>
          <div className="p-4 bg-white shadow rounded-lg">
            <h3 className="font-semibold text-lg">Portfolio Value</h3>
            <p className="text-xl font-bold">$100,000.00</p>
          </div>
          <div className="p-4 bg-white shadow rounded-lg">
            <h3 className="font-semibold text-lg">Model Used</h3>
            <p className="text-gray-500">GRU + DQN/PPO (Not Trained)</p>
          </div>
        </section>
      </div>
    </main>
  );
}