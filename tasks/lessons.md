# Lessons

- Keep heavy vision compute behind manifest-driven SLURM jobs; local Mac workflows should remain config and smoke-test only.
- Preserve the existing repo contracts while migrating architecture; additive interfaces reduce breakage during phased upgrades.
- Separate registry metadata from runtime download logic so gated model handling stays explicit and auditable.
