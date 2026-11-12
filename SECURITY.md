# Security policy

## Supported versions

pouw is a research prototype. Only the latest commit on `main` receives
security fixes.

## Reporting a vulnerability

Do not report security vulnerabilities through public GitHub issues. Report
them privately through the repository's **Security** tab, using **Report a
vulnerability**.

Include the following:

1. A description of the issue and its impact.
2. The affected component and commit.
3. Step-by-step instructions to reproduce it, including any proofs, messages,
   or scripts needed.
4. The expected and the observed behavior.

A maintainer acknowledges the report within one week and keeps you informed
until it is resolved.

## Scope

In scope:

- Obtaining credit for work the GPU did not perform, for example a proof that
  passes all five checks without the multiplication it claims, a proof counted
  twice, or a proof made with weights other than the declared ones.
- Verifier bugs that a remote party can trigger, such as a crash from a
  crafted message.

Out of scope, and open for discussion in public issues:

- The verifier trusts every client that connects to its socket.
- A proof does not show that its activations came from real requests.
- The GPU reports its own ID, and nothing signs it.
- The open questions in [NEXT_STEPS.md](NEXT_STEPS.md).

Report weaknesses in upstream's proof scheme or kernels to
[upstream](https://github.com/pearl-research-labs/pearl/security).
