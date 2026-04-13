# Constraints and Assumptions

## Explicit constraints

### Public truth constraints

- The site must reflect the currently published beta prerelease, not a future ideal state.
- The verified primary environment remains **Arch Linux + Ryzen 7 7800X3D + Radeon RX 7900 GRE**.
- Other distro packages are preview targets and must stay labeled that way.
- The current project is still beta software, not a general-availability release.
- Host-side AMD GPU / driver requirements cannot be bundled away and must remain visible.

### Product / repo constraints

- The website should align with the current repo docs rather than inventing a parallel truth surface.
- The site should send visitors back into canonical repo docs for deep technical detail.
- The site should use GitHub Releases as the initial download backend.

### Operational constraints

- Prefer a free or near-free hosting model.
- Avoid introducing a web stack that is heavy relative to the current team size.
- Keep release updates easy enough that publishing a new beta does not become a website-maintenance tax.

## Assumptions

### Safe assumptions used in this pack

- GitHub Pages is the default first-hosting target because it matches the project's current zero-dollar launch path.
- The first site version should be static and mostly content-driven.
- The first site version should prioritize homepage + downloads + proof / validation over breadth.
- Initial screenshots or motion assets may be limited, so the design should work even if the first version ships with restrained visuals.
- The website should be visually better than a raw repo page, but credibility matters more than decorative flash.

### Assumptions intentionally not treated as facts

- a custom domain is not assumed
- a specific frontend framework is not assumed as mandatory
- analytics are not assumed
- automated release metadata sync is not assumed yet

## Boundaries

The website may:

- summarize the repo truth surface
- reorganize it into clearer public information architecture
- create stronger landing-page copy

The website may not:

- widen support claims beyond the repo evidence
- promise validation that has not happened
- imply a broader roadmap is already delivered
