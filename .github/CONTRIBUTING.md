# How to Contribute

We welcome contributions from external contributors. This document describes the process for merging code changes into this repo.

## Important: Contribution Quality & AI Policy

As a small team, we must prioritize Pull Requests (PRs) that require minimal review. To maintain high project standards:

* **Human Validation:** While AI-assisted contributions are not strictly discouraged, they **must** undergo significant manual, human validation and verification.
* **Submission State:** All PRs must be submitted in a "ready-to-merge" state. We reserve the right to close any PR without review if it does not meet these quality standards.
* **Verification:** If you have pending PRs, please audit them manually and leave a comment on the PR once you have personally verified the changes. We will defer reviewing these submissions until this confirmation is provided.
* **Final Authority:** All final decisions regarding the acceptance of contributions remain at the sole discretion of the maintainers.

---

## Getting Started

* Make sure you have a [GitHub account](https://github.com/signup/free).
* [Fork](https://help.github.com/articles/fork-a-repo/) this repository on GitHub.
* On your local machine, [clone](https://help.github.com/articles/cloning-a-repository/) your fork of the repository.

## Making Changes

* **Scoped Issues:** Before starting work, ensure the issue you are contributing to is **well-scoped**. We prefer focused, incremental improvements over broad, ambiguous changes. If you are proposing a new feature, please discuss it in an issue first to ensure it aligns with the project's direction.
* **Branching:** Add your code to your local fork. It is recommended to make changes on a [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) with a name relating to the feature you are adding.
* **Opening a PR:** When ready for feedback, navigate to your fork on GitHub and open a [pull request](https://help.github.com/articles/using-pull-requests/). Subsequent commits to that branch will be added to the PR automatically and validated for mergeability and test suite compliance.
* **Requirements:** If you are providing a new feature, you **must** add corresponding test cases and documentation.
* **Testing:** Before final submission, ensure you run the test suite locally using `pytest`.
* **Final Submission:** When the code is ready to be considered for merging, check the **"Ready to go"** box on the PR page. The code will not be merged until this box is checked, continuous integration passes, and core developers provide "Approved" reviews.

## Additional Resources

* [General GitHub documentation](https://help.github.com/)
* [PR best practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
* [A guide to contributing to software packages](http://www.contribution-guide.org)
