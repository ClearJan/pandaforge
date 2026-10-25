# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.3.x   | yes |
| < 1.3   | no  |

## Scope

PandaForge is a local desktop application. It runs no network services, no
telemetry, and no automatic updates — after `setup.cmd` everything works
offline. Attack surface considerations are therefore limited to:

- **Untrusted scene files** (`scenes/*.json`, loaded via the Scene tab's
  Open): these are plain data parsed with `json.loads` and validated through
  dataclasses; no code execution path. Do not add features that eval scene
  content.
- **Model assets** (`assets/robots/panda/`): vendored MJCF/meshes from
  mujoco_menagerie at a pinned revision; treat any replacement the same way
  you would treat model files from any source.

## Reporting

Report vulnerabilities privately via GitHub's "Report a vulnerability"
security advisory rather than a public issue. Expect a response within 7
days.
