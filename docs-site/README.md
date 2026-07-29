# rag-core documentation

The documentation site is a statically exported Next.js application. Product
pages live in `content/docs`; the landing page lives in `src/app/(home)`.

From this directory:

```bash
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://localhost:3000`. Run `pnpm build` before changing deployment or
routing behavior.
