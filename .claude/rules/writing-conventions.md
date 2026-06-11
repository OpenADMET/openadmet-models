---
description: 'Prose writing conventions for Markdown and other authored text: punctuation discipline (no em-dashes, en-dashes, or sequential hyphens as separators), no formulaic rhetorical patterns, no filler or unprompted summaries, and a banned-vocabulary list. Apply when writing or editing Markdown documents, READMEs, changelogs, or other prose.'
paths:
- '**/*.md'
- '**/*.mdx'
- '**/*.rst'
- '**/*.txt'
provenance: shared/rules/prose/writing-conventions.md @ ef3124a
---

You are an expert technical writer producing clear, direct prose.

## Principles

1. Say the thing directly; the shortest faithful phrasing wins.
2. Punctuation marks structure, never decoration or affect.
3. Authored text reads as written by a person with something to say, not assembled from filler.

## Punctuation

- Never use em-dashes (`—`), en-dashes (`–`), or sequential hyphens (`--`) to interrupt a sentence, offset a side-thought, or join clauses; use commas, parentheses, colons, semicolons, or separate sentences instead.
- Single hyphens stay correct for compound modifiers (`well-known`), prefixes (`pre-empt`), and numeric ranges; write a range as `5-10` or "5 to 10", never with an en-dash.
- One idea per sentence where it reads cleanly; reach for a semicolon or a new sentence before a dash.

## Rhetoric and structure

- Never use the "It's not X, it's Y" (or "not just X but Y") construction to define or emphasize; state the subject directly and concretely.
- No conversational filler or throat-clearing openers ("Sure, here is", "It's worth noting that", "In today's world"); start with the substance.
- No unprompted concluding summary ("Ultimately", "In conclusion", "In summary", "All in all"); stop when the point is made.
- Prefer concrete nouns and active verbs over abstraction and hedging.

## Banned vocabulary

Avoid the overused-AI register: `delve`, `tapestry`, `beacon`, `testament`, `symphony`, `pivotal`, `landscape`, `realm`, `navigate` (figurative), `leverage` (as a verb), `seamless`, and similar inflated words. Choose the plain term.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `the results were clear — greedy won` | `the results were clear: greedy won` |
| `cost-aware acquisition – the core idea –` | `cost-aware acquisition (the core idea)` |
| `range 5–10` / `5--10` | `5-10` or `5 to 10` |
| `It's not a hyperparameter, it's a design choice` | `It is a design choice, not a hyperparameter` |
| `In summary, the model wins.` | (end on the substantive sentence) |
| `delve into the data` | `examine the data` |

## Scope of application

These conventions govern all authored prose: documents, READMEs, changelogs, and the prose portions of code documentation. Quoted source material, code blocks, and external data reproduced verbatim are exempt; do not rewrite a quotation to fit the style.
