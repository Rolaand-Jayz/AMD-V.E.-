# Technical Specification

## Recommended first implementation

Build the first website version as a **static site** that links to GitHub Releases for downloads and to the repo docs for deep technical detail.

## Hosting model

### Recommended default

- **Host:** GitHub Pages
- **Downloads backend:** GitHub Releases
- **Source of truth:** repo markdown docs plus a small site-specific content layer

This keeps the site aligned with the project's current free-tier launch strategy and avoids introducing a second binary hosting system before it is needed.

## Implementation recommendation

### First version

Use a lightweight static implementation with:

- semantic HTML
- custom CSS
- minimal JavaScript only where it improves usability
- no backend dependency
- no mandatory CMS

This is the safest first implementation because it:

- reduces maintenance burden
- minimizes toolchain drift from the main repo
- makes GitHub Pages deployment straightforward
- supports a credibility-first launch better than a complex stack would

### Optional future upgrade path

If content complexity grows, a small static-site framework can be introduced later. That is not required for the first version.

## Suggested content sources

### Canonical repo docs

- `README.md`
- `docs/RELEASE_STATUS.md`
- `docs/SUPPORT_TIERS.md`
- `docs/LIMITATIONS.md`
- `docs/VALIDATION_AND_EVIDENCE.md`
- `docs/BENCHMARKS.md`

### Site-local content

Add a website-specific content layer for:

- homepage copy blocks
- downloads asset descriptions
- concise proof summaries
- CTA text
- visual assets and screenshots when available

## Release integration options

### Option A — manual release metadata file (recommended first)

Maintain a checked-in data file for:

- current release tag
- primary download asset labels
- preview package labels
- checksum links

Pros:

- simple
- reviewable
- no runtime API dependency

Cons:

- requires a small content update when releases change

### Option B — build-time GitHub API fetch (future)

At site build time, fetch release metadata from GitHub and render the current release automatically.

Pros:

- less manual metadata maintenance

Cons:

- higher complexity
- more moving parts in the deployment flow

## Content model requirements

The site should model assets with fields like:

- label
- file name
- type
- support tier
- short description
- checksum availability
- host requirement notes

## Accessibility and performance

The first version should:

- remain readable on mobile and desktop
- use accessible heading hierarchy
- maintain strong contrast
- keep the downloads table usable with keyboard navigation
- avoid heavy animation on the critical path
- keep page weight restrained, especially before real screenshots/video are available

## Deployment workflow

Recommended first workflow:

1. update site content when a new release is published
2. verify that download links and support labels match the repo docs
3. deploy static site to GitHub Pages
4. smoke-test homepage, downloads page, and proof page

## File placement options

### Preferred options

- `website/` if the site becomes a first-class code artifact
- `docs/site/` if the first version stays tightly documentation-adjacent

This pack does not force one choice yet. That decision remains open until implementation begins.
