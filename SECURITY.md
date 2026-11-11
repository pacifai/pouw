# Security policy

pouw is a research prototype. Only the latest commit on `main` is supported.

## Report a vulnerability

Report privately through GitHub: open the repository's **Security** tab and
select **Report a vulnerability**. Don't open a public issue for it.

Include what you found, how to reproduce it, and what it lets an attacker do.
A maintainer replies within a week.

## What counts

- **A way to get credit for work the GPU didn't do.** For example, a proof that
  passes all five checks without the multiplication it claims, a proof counted
  twice, or a proof made with weights other than the declared ones.
- **A bug in the verifier** that a remote party can trigger, such as a crash
  from a crafted message.

Some limits are already known. These aren't vulnerabilities, and you can
discuss them in public issues:

- The verifier trusts whoever connects to its socket.
- A proof doesn't show that its activations came from real requests.
- The GPU reports its own ID, and nothing signs it.
- The open questions in [NEXT_STEPS.md](NEXT_STEPS.md).

Weaknesses in upstream's proof scheme or kernels belong with
[upstream](https://github.com/pearl-research-labs/pearl).
