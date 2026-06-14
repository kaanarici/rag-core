import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

// Set NEXT_PUBLIC_BASE_PATH=/rag-core for the GitHub Pages project-site build;
// left empty for local dev and root-domain hosts.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  reactStrictMode: true,
  basePath,
  images: { unoptimized: true },
};

export default withMDX(config);
