# Lessons

- Keep heavy vision compute behind manifest-driven SLURM jobs; local Mac workflows should remain config and smoke-test only.
- Preserve the existing repo contracts while migrating architecture; additive interfaces reduce breakage during phased upgrades.
- Separate registry metadata from runtime download logic so gated model handling stays explicit and auditable.
- For verifier quality, data shape matters more than classifier complexity; richer weak-alignment features improve the usefulness of the fusion dataset without requiring immediate model changes.
- Generated split files should stay out of code commits unless the split policy itself changed; otherwise they add noise to review.
- If verifier training text includes the target label or inference uses a different feature schema than training, the fusion metric becomes inflated and the deployed behavior becomes unstable.
