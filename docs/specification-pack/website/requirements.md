# Requirements

## Functional requirements

### 1. Homepage

The site must provide a homepage that:

- explains AMD Video Enhancer in plain language
- communicates the AMD-first / Linux-first angle without sounding tribal or sloppy
- states that the current public release is a beta prerelease
- identifies the verified primary environment explicitly
- provides a primary CTA to the current downloads surface
- provides a secondary CTA to proof / validation content or docs

### 2. Downloads page

The site must provide a downloads page that:

- highlights the current beta tag
- links to GitHub release assets instead of hosting a separate binary store initially
- distinguishes the primary verified asset path from preview package targets
- shows checksum availability clearly
- explains host-side GPU / driver requirements that are not bundled
- links to support tiers, limitations, and release status near the downloads themselves

### 3. Proof / validation surface

The site must provide a page or section that:

- summarizes what is actually verified today
- links to benchmark and validation evidence
- makes preview package targets visibly different from verified support
- explains that packaging reach does not equal compatibility proof
- surfaces current limitations instead of hiding them

### 4. Docs / contribute path

The site must provide a clean path to:

- canonical repo docs
- contribution guidance
- issue reporting / beta feedback flow
- release notes or changelog context

### 5. Truth-surface governance

The website must:

- separate current truth from roadmap or ambition
- avoid unsupported platform or performance claims
- avoid screenshots, numbers, or endorsements that the repo cannot support yet
- use wording consistent with `README.md`, `docs/RELEASE_STATUS.md`, `docs/SUPPORT_TIERS.md`, `docs/LIMITATIONS.md`, and `docs/VALIDATION_AND_EVIDENCE.md`

## Non-functional requirements

The first implementation should be:

- static-host friendly
- cheap or free to host
- fast to load on desktop and mobile
- easy to update when a new GitHub release ships
- accessible enough to avoid obvious keyboard / contrast / hierarchy failures
- maintainable by one person without a heavy web stack

## Content requirements

The site should include or link out to:

- current release tag
- verified primary environment
- preview distro list
- support tiers
- limitations
- benchmark / validation evidence
- release notes or changelog

## Prohibited outcomes

The first version must not:

- imply Windows, macOS, or NVIDIA support
- bury the beta / preview status in fine print
- present preview packages as guaranteed compatibility
- rely on vague AI marketing language where a specific claim would do
