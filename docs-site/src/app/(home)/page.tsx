import Link from 'next/link';

const features = [
  {
    title: 'Local first',
    body: 'Index a folder and search it with a local model. No API key and no external services to run.',
  },
  {
    title: 'Scales up',
    body: 'Move to Qdrant, pgvector, or TurboPuffer with real embeddings when your application needs them.',
  },
  {
    title: 'Cited context',
    body: 'Get prompt-safe context with citations, or raw ranked hits when you want to build the prompt yourself.',
  },
  {
    title: 'Embeddable',
    body: 'A retrieval core you call from your code. Your application keeps auth, tenancy, and generation.',
  },
];

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col">
      <section className="mx-auto flex w-full max-w-3xl flex-col items-center px-4 py-24 text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">rag-core</h1>
        <p className="mt-5 max-w-2xl text-lg text-fd-muted-foreground sm:text-xl">
          An embeddable retrieval engine for RAG. Index a folder, ask a question, and
          get back ranked context with citations.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/docs"
            className="rounded-lg bg-fd-primary px-5 py-2.5 text-sm font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
          >
            Get started
          </Link>
          <Link
            href="https://github.com/kaanarici/rag-core"
            className="rounded-lg border px-5 py-2.5 text-sm font-medium transition-colors hover:bg-fd-accent"
          >
            View on GitHub
          </Link>
        </div>
        <pre className="mt-10 w-full max-w-md overflow-x-auto rounded-xl border bg-fd-card p-4 text-left text-sm shadow-sm">
          <code>{`import rag_core

rag = rag_core.index("./docs")
print(rag.ask("How can invoices be paid?"))`}</code>
        </pre>
      </section>

      <section className="mx-auto grid w-full max-w-4xl grid-cols-1 gap-4 px-4 pb-24 sm:grid-cols-2">
        {features.map((feature) => (
          <div key={feature.title} className="rounded-xl border bg-fd-card p-5 text-left">
            <h2 className="font-semibold">{feature.title}</h2>
            <p className="mt-2 text-sm text-fd-muted-foreground">{feature.body}</p>
          </div>
        ))}
      </section>
    </main>
  );
}
