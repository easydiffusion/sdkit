# AGENTS.md

Guidance for AI coding agents working in this repository.

## Read the knowledge base first

Before making any changes, read the knowledge base in the `kb/` folder, starting
with the index page:

- [kb/README.md](kb/README.md) - reading order and page index.

Specifically, read [kb/development.md](kb/development.md) carefully: it describes
the day-to-day workflow, how to set up a local environment, how to build and run
the server locally, the coding conventions used in the repo, and an end-to-end
walkthrough for adding a new feature. Note in particular its Windows path length
note about the `subst` drive letter substitution trick, which is used to bypass
the MAX_PATH (260-character) limitation when the checkout or build directory
paths get too deep for CMake.

Other useful pages, depending on the task:

- [kb/architecture.md](kb/architecture.md) - big-picture internal design.
- [kb/server-and-api.md](kb/server-and-api.md) - the HTTP layer and routes.
- [kb/image-generation.md](kb/image-generation.md) - ImageGenerator internals.
- [kb/building-and-platforms.md](kb/building-and-platforms.md) - build and platforms.

## Conventions

- Follow the coding conventions in [kb/development.md](kb/development.md)
  (header/source pairs, naming, logging macros, HTTP route style).
- Update the relevant page in `kb/` when your change affects documented behavior.
