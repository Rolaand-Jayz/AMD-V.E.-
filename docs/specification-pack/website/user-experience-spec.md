# User Experience Specification

## Primary user journeys

### Journey 1: "Can I trust this enough to try it?"

1. Visitor lands on homepage.
2. Visitor understands the AMD-first / Linux-first purpose quickly.
3. Visitor sees that the current release is a public beta, not a vague future promise.
4. Visitor finds proof, support boundaries, and downloads without digging.

### Journey 2: "Which download is actually meant for me?"

1. Visitor opens downloads page.
2. Visitor sees the current tag and asset categories.
3. Visitor can distinguish:
   - verified-primary path
   - preview distro packages
   - checksum files
   - supporting assets like AUR handoff bundles
4. Visitor understands host requirements before downloading.

### Journey 3: "I like this, but I need details first."

1. Visitor clicks into proof / validation or docs.
2. Visitor sees benchmarks, support tiers, limitations, and release status.
3. Visitor can continue to repo docs or contribution guidance.

## Information architecture

### Recommended first-version navigation

- Home
- Downloads
- Proof / Validation
- Docs
- Contribute

## Homepage structure

1. **Hero**
   - clear description of the app
   - explicit beta status
   - primary CTA: download beta
   - secondary CTA: see proof / support surface
2. **Why it matters**
   - AMD-first / Linux-first rationale
   - why the project exists in public
3. **What is real today**
   - current beta tag
   - verified primary environment
   - supported proof boundaries
4. **How it works at a glance**
   - short execution-path explanation
5. **Downloads preview**
   - short asset overview with link to full downloads page
6. **Proof and limitations teaser**
   - benchmark scope, support tiers, and known limits
7. **Contribute / docs CTA**

## Downloads page structure

1. current beta release banner
2. primary verified download path
3. preview package matrix
4. checksums section
5. host requirements section
6. support tiers and limitations links
7. issue / feedback CTA

## Proof / validation page structure

1. current truth summary
2. verified primary environment
3. benchmark scope and evidence links
4. support-tier table
5. limitations summary
6. release-status and packaging-boundary links

## Tone requirements

- direct
- technical
- grounded
- slightly bold, never inflated
- friendly to beginners without dumbing down the truth

## UX failure conditions

The experience fails if:

- a visitor can miss that this is still beta software
- a visitor downloads a preview package believing it is broadly validated
- a skeptical visitor cannot find evidence quickly
- the homepage sounds more impressive than the proof surface supports
