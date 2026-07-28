# Release readiness

Use this page when a maintainer wants to treat the current tree as public-beta
ready. These checks prove package shape, local no-key journeys, optional HTTP
runtime wiring, and public-checkout reproducibility. They do not prove semantic
retrieval quality on arbitrary corpora or paid-provider behavior.

## Local gate

Run the full landing gate from a clean checkout:

```bash
./scripts/landing_check.sh
```

This runs sync, lint, typecheck, the full test suite, no-key journey smoke,
Vercel AI SDK example typecheck, self-host HTTP smoke, optional integration
smoke, build, artifact checks, and wheel install smoke.

For iteration, use:

```bash
./scripts/landing_check.sh --quick
```

## Public checkout gate

After pushing, prove that a fresh clone can validate itself without local-only
files:

```bash
./scripts/public_checkout_smoke.sh --package
./scripts/github_install_smoke.sh https://github.com/kaanarici/rag-core.git main
```

By default the script clones `origin` and checks out the current checkout's
`HEAD` SHA in a temporary directory. Set a remote URL and ref explicitly when
checking a branch or tag:

```bash
./scripts/public_checkout_smoke.sh --package https://github.com/kaanarici/rag-core.git main
```

Use `--quick` when you only need the first-run developer path:

```bash
./scripts/public_checkout_smoke.sh --quick
```

Use `github_install_smoke.sh` with an explicit tag when checking installability
for a release candidate:

```bash
./scripts/github_install_smoke.sh https://github.com/kaanarici/rag-core.git v0.1.0
```

## Remote CI

For the public GitHub repository, the release-ready signal is a successful
`ci` workflow on the pushed SHA:

```bash
gh run list --repo kaanarici/rag-core --branch main --limit 3
gh run watch <run-id> --repo kaanarici/rag-core --exit-status
```

## Release artifacts

The `release-artifacts` workflow runs on `v*` tags or manual dispatch. It runs
the quick landing gate, builds the sdist and wheel, checks the artifact contents,
runs the wheel smoke, and uploads `dist/*` as a GitHub Actions artifact. It does
not publish to a registry.

Registry publishing, tagging, and GitHub Releases are separate release actions.
Do not treat a green checkout as proof that a package was published.
