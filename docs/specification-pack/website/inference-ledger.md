# Inference Ledger

## Selected inference level

- **Level:** low

## Explicit facts used

- the first public beta prerelease is now live as `v0.1.0-beta.1`
- the primary verified environment is Arch Linux + Ryzen 7 7800X3D + Radeon RX 7900 GRE
- preview distro packages exist but are not broad compatibility proof
- GitHub Releases is the download backend for the current beta
- the user wants an anti-slop website design phase after release publication

## Inferred decisions

| Decision | Why it was inferred | Confidence |
| --- | --- | --- |
| GitHub Pages should be the default first host | it matches the previously discussed free GitHub path and keeps the first website simple | medium |
| The first version should be static and low-maintenance | the project is still release-focused and should not absorb unnecessary web-stack overhead | high |
| The minimum viable page set is Home + Downloads + Proof / Validation + Docs / Contribute | this is the smallest set that can carry the current truth cleanly | high |
| The website should prioritize download safety and support clarity over broad marketing polish | the repo's current credibility depends on visible proof and narrow claims | high |
| Manual release metadata is the safest first integration path | it is simpler and less fragile than adding a dynamic release-fetch workflow immediately | medium |

## Deferred decisions

- exact visual style and palette details
- final site source location (`website/` vs `docs/site/`)
- whether screenshots are required for version one
- whether to add build-time GitHub API sync later
- whether to use a custom domain

## Guardrail

Any later implementation that wants to widen claims beyond this pack must first update the release truth surface and validation evidence.
