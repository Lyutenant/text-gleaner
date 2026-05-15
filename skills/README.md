# Skills

Agent skills for generating textgleaner-compatible extraction schemas through reasoning, without invoking the textgleaner library or CLI.

Both frameworks follow the [Agent Skills](https://agentskills.io) open standard: a named directory containing `SKILL.md` with YAML frontmatter. The skill content is identical across frameworks — only the frontmatter fields differ.

## Available skills

| Skill | What it does |
|---|---|
| `generate-schema` | Analyzes plain-text sample documents and produces a JSON extraction schema |

## Framework versions

```
skills/
  openclaw/
    generate-schema/
      SKILL.md        ← name + description required (hyphen-case)
  claude-code/
    generate-schema/
      SKILL.md        ← description recommended; supports $ARGUMENTS, argument-hint, disable-model-invocation
```

### Claude Code

Copy the `claude-code/generate-schema/` directory to your project's `.claude/skills/` or your personal `~/.claude/skills/`. Invoke with `/generate-schema sample.txt "what to extract"`.

```bash
cp -r skills/claude-code/generate-schema ~/.claude/skills/
```

### OpenClaw

Copy the `openclaw/generate-schema/` directory to `~/.openclaw/workspace/skills/`, then run `/new` or restart the gateway.

```bash
cp -r skills/openclaw/generate-schema ~/.openclaw/workspace/skills/
```

## Adding a new framework

Create a new subfolder under `skills/` named after the framework. Adapt the frontmatter to that framework's conventions — the two-pass methodology and schema format rules in the body stay identical.
