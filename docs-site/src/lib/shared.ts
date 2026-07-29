export const appName = 'rag-core';
export const docsRoute = '/docs';
export const docsImageRoute = '/og/docs';
export const docsContentRoute = '/llms.mdx/docs';

// Next prefixes the base path onto <Link> and assets automatically, but not
// onto raw URL strings that a client component fetches (the markdown copy
// button) or that metadata resolves against metadataBase (OG images). Prepend
// it explicitly for those.
export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export const gitConfig = {
  user: 'kaanarici',
  repo: 'rag-core',
  branch: 'main',
};
