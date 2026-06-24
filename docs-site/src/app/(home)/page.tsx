import Link from 'next/link';
import { codeToHtml } from 'shiki';
import { Blocks, Check, Cpu, Database, Quote } from 'lucide-react';

const LOCAL_CODE = `import rag_core

# Index a folder with a local model. No API key, no services.
idx = rag_core.index("./docs")

# Prompt-ready context with citations, for any LLM.
print(idx.context("How can invoices be paid?"))`;

const CONFIGURED_CODE = `from rag_core import Engine, Config

# Same calls, now backed by Qdrant and OpenAI embeddings.
config = Config.qdrant(
    url="http://127.0.0.1:6333",
    embedding_provider="openai",
    model="text-embedding-3-small",
)

async with Engine(config) as core:
    await core.add("./docs", collection="help")
    pack = await core.context(query="How can invoices be paid?", collection="help")
    print(pack.as_prompt_text())`;

const features = [
  {
    icon: Cpu,
    title: 'Local first',
    body: 'Index a folder and search it with an on-device model. No API key and no services to run.',
  },
  {
    icon: Database,
    title: 'Scales up',
    body: 'Move to Qdrant, pgvector, or TurboPuffer with real embeddings when you need them.',
  },
  {
    icon: Quote,
    title: 'Cited context',
    body: 'Get prompt-safe context with citations, or raw ranked hits to build the prompt yourself.',
  },
  {
    icon: Blocks,
    title: 'Embeddable',
    body: 'A retrieval core you call from code. Your app keeps auth, tenancy, and generation.',
  },
];

const coreOwns = [
  'Ingest and parsing',
  'Chunking',
  'Indexing and search',
  'Context assembly and citations',
  'Traces and eval hooks',
];

const appOwns = [
  'Authentication and tenancy',
  'Connectors and sync',
  'Chat and generation',
  'Product UI',
  'Rate limits and billing',
];

async function highlight(code: string) {
  return codeToHtml(code, {
    lang: 'python',
    themes: { light: 'github-light', dark: 'github-dark' },
  });
}

export default async function HomePage() {
  const [localHtml, configuredHtml] = await Promise.all([
    highlight(LOCAL_CODE),
    highlight(CONFIGURED_CODE),
  ]);

  return (
    <main className="flex flex-1 flex-col">
      <section className="mx-auto grid w-full max-w-6xl items-center gap-12 px-6 py-20 lg:grid-cols-2 lg:py-28">
        <div className="flex flex-col items-start">
          <span className="mb-5 rounded-full border px-3 py-1 text-xs font-medium text-fd-muted-foreground">
            Python retrieval engine for RAG
          </span>
          <h1 className="text-balance text-4xl font-semibold tracking-tight sm:text-5xl">
            Retrieval for RAG, in a few lines of Python.
          </h1>
          <p className="mt-5 max-w-md text-lg leading-relaxed text-fd-muted-foreground">
            rag-core is an embeddable engine. Point it at a folder, ask a
            question, and get back ranked context with citations. It runs locally
            with no API key and scales to your own vector store.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link
              href="/docs"
              className="rounded-lg bg-fd-primary px-5 py-2.5 text-sm font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
            >
              Get started
            </Link>
            <Link
              href="/docs/quickstart"
              className="rounded-lg border px-5 py-2.5 text-sm font-medium transition-colors hover:bg-fd-accent"
            >
              Quickstart
            </Link>
            <Link
              href="https://github.com/kaanarici/rag-core"
              className="rounded-lg border px-5 py-2.5 text-sm font-medium transition-colors hover:bg-fd-accent"
            >
              GitHub
            </Link>
          </div>
          <code className="mt-6 rounded-md border bg-fd-muted/40 px-3 py-1.5 font-mono text-sm text-fd-muted-foreground">
            pip install rag-core
          </code>
        </div>

        <div className="hero-code w-full overflow-x-auto rounded-xl border bg-fd-card p-5 shadow-sm">
          <div dangerouslySetInnerHTML={{ __html: localHtml }} />
        </div>
      </section>

      <section className="mx-auto w-full max-w-6xl px-6 pb-20">
        <div className="grid grid-cols-1 gap-px overflow-hidden rounded-xl border bg-fd-border sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature) => (
            <div key={feature.title} className="bg-fd-background p-6">
              <div className="flex size-9 items-center justify-center rounded-lg border bg-fd-muted/40 text-fd-primary">
                <feature.icon className="size-[18px]" strokeWidth={2} />
              </div>
              <h2 className="mt-4 text-sm font-semibold">{feature.title}</h2>
              <p className="mt-2 text-sm leading-relaxed text-fd-muted-foreground">
                {feature.body}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto w-full max-w-6xl px-6 pb-20">
        <div className="max-w-2xl">
          <h2 className="text-2xl font-semibold tracking-tight">
            Start local, scale when you need to
          </h2>
          <p className="mt-3 leading-relaxed text-fd-muted-foreground">
            The first run uses an on-device model so you can index and search
            without an API key. The same calls move to a configured vector store
            and hosted embeddings by swapping the config factory.
          </p>
        </div>
        <div className="hero-code mt-8 w-full overflow-x-auto rounded-xl border bg-fd-card p-5 shadow-sm">
          <div dangerouslySetInnerHTML={{ __html: configuredHtml }} />
        </div>
      </section>

      <section className="mx-auto w-full max-w-6xl px-6 pb-24">
        <div className="max-w-2xl">
          <h2 className="text-2xl font-semibold tracking-tight">
            A retrieval core, not a platform
          </h2>
          <p className="mt-3 leading-relaxed text-fd-muted-foreground">
            rag-core stays a library you embed. It owns the retrieval path and
            leaves the application concerns to your code.
          </p>
        </div>
        <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="rounded-xl border bg-fd-card p-6">
            <h3 className="text-sm font-semibold">rag-core owns</h3>
            <ul className="mt-4 divide-y divide-fd-border text-sm">
              {coreOwns.map((item) => (
                <li key={item} className="flex items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                  <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-fd-primary/10 text-fd-primary">
                    <Check className="size-3.5" strokeWidth={2.5} />
                  </span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
          <div className="rounded-xl border bg-fd-card p-6">
            <h3 className="text-sm font-semibold">Your application owns</h3>
            <ul className="mt-4 divide-y divide-fd-border text-sm text-fd-muted-foreground">
              {appOwns.map((item) => (
                <li key={item} className="flex items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                  <span className="flex size-5 shrink-0 items-center justify-center rounded-full border bg-fd-muted/40">
                    <Check className="size-3.5" strokeWidth={2.5} />
                  </span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <footer className="mt-auto border-t">
        <div className="mx-auto flex w-full max-w-6xl flex-col items-center justify-between gap-3 px-6 py-8 text-sm text-fd-muted-foreground sm:flex-row">
          <span>rag-core, an embeddable retrieval engine for RAG.</span>
          <nav className="flex items-center gap-5">
            <Link href="/docs" className="transition-colors hover:text-fd-foreground">
              Docs
            </Link>
            <Link
              href="https://github.com/kaanarici/rag-core"
              className="transition-colors hover:text-fd-foreground"
            >
              GitHub
            </Link>
            <Link
              href="https://pypi.org/project/rag-core/"
              className="transition-colors hover:text-fd-foreground"
            >
              PyPI
            </Link>
          </nav>
        </div>
      </footer>
    </main>
  );
}
