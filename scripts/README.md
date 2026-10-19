# Local verification scripts

Start with [development and release](../docs/release.md). These scripts run
locally and do not publish or push. Live provider checks are separate.

| Script | Purpose |
| --- | --- |
| `landing_check.sh` | Run lint, typing, local tests, and installed-package checks. `--quick` selects iteration checks. |
| `dx_smoke.sh` | Exercise the no-key journey with demo embeddings. This proves plumbing. |
| `check_dist_artifacts.py` | Check public source archives, exact wheel source, typing files, and retired-module exclusions. |
| `wheel_smoke.py` | Install the wheel in a fresh environment and use real local models. `--keep-temp` preserves evidence. |
| `github_install_smoke.sh` | Install an explicit Git remote/ref and check the installed API. |
| `public_checkout_smoke.sh` | Clone an explicit remote/ref and run its local gate. |
| `validate_provider_fixtures.sh` | Replay provider SDK fixtures without live credentials. |
| `verify_optional_integrations.sh` | Check the retained optional MCP entry point. |

Models may download on first use. FastEmbed and Hugging Face cache variables are
respected. Do not confuse cached model reuse with a fresh offline-install test.
