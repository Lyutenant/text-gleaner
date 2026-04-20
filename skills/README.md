# Skills

Agent skills for generating textgleaner-compatible extraction schemas through reasoning, without invoking the textgleaner library or CLI.

Each subfolder contains a version of the same skill adapted to a specific agent framework's conventions.

## Available skills

| Skill | What it does |
|---|---|
| `generate-schema` | Analyzes plain-text sample documents and produces a JSON extraction schema |

## Framework versions

```
skills/
  openclaw/
    generate-schema/
      SKILL.md        ← OpenClaw format (YAML frontmatter + markdown)
  claude-code/
    generate-schema.md  ← Claude Code slash command (copy to .claude/commands/)
```

### OpenClaw

Copy the `generate-schema/` directory to `~/.openclaw/workspace/skills/`, then run `/new` or restart the gateway.

### Claude Code

Copy `claude-code/generate-schema.md` to your project's `.claude/commands/` directory. Invoke with `/generate-schema <samples and description>`.
