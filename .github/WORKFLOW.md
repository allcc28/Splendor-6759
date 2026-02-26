# AI-Assisted Agile Development Workflow

Quick reference for working with AI assistant on this project.

## Starting a Work Session

```
"Hey, I want to continue working on [task/feature]. 
Show me the current sprint status."
```

## During Development

### Implementing a Feature
```
"Let's implement [feature name] from the sprint backlog."
"Show me the relevant code first."
"Now let's modify [file] to add [functionality]."
```

### Making Decisions
```
"We need to decide between [option A] and [option B].
Help me create an ADR to document this decision."
```

### Documentation
```
"Create a dev log for today's session."
"Update the sprint planning doc - mark [task] as done."
"Generate a technical spec for [component]."
```

### Debugging
```
"This code isn't working. Help me debug [issue]."
"Run [test/script] and analyze the error."
```

## Ending a Session

```
"Summarize what we accomplished today and create a dev log."
"Update the sprint backlog with our progress."
"What should I work on next session?"
```

## Sprint Management

### Starting a Sprint
```
"Let's start Sprint [N]. Create the planning doc."
```

### Mid-Sprint
```
"Update today's daily log."
"What's our progress on Sprint [N] goals?"
```

### Ending a Sprint
```
"Generate a retrospective for Sprint [N]."
"What did we accomplish? What's left for next sprint?"
```

## Best Practices

1. **Be Specific**: Reference exact file names, function names, or line numbers
2. **Provide Context**: Mention what you're trying to achieve
3. **Review Output**: Always check AI-generated code before running
4. **Document Decisions**: Ask for ADRs when making important choices
5. **Iterate**: If AI's first attempt isn't perfect, ask for refinements

## Common Tasks

### Creating New Components
```
"Create a new [agent/module/script] in [directory] that does [X]."
```

### Refactoring
```
"Refactor [file/function] to improve [performance/readability]."
```

### Testing
```
"Create unit tests for [component]."
"Run the tests and fix any failures."
```

### Experiments
```
"Set up an experiment to compare [A] vs [B]."
"Create a config file for [experiment name]."
"Run [experiment] and analyze the results."
```

## Documentation Hierarchy

- **Dev Logs**: What happened each session (chronological)
- **ADRs**: Why we made important decisions (by decision)
- **Specs**: How components work (by component)
- **Sprint Docs**: What we're building (by time period)
- **Meeting Notes**: Team discussions (by meeting)

## File Naming Conventions

- Dev Logs: `YYYY-MM-DD_sessionN_brief_description.md`
- ADRs: `ADR-NNN-short-title.md` (sequential numbering)
- Specs: `component_name_spec.md`
- Meeting Notes: `YYYY-MM-DD_meeting_type.md`
- Daily Logs: `YYYY-MM-DD.md`

## Git Commit Messages

Use conventional commits:
```
feat: Add score-based evaluator
fix: Correct reward calculation in event-based agent
docs: Update sprint 1 planning
refactor: Simplify state representation
test: Add unit tests for MCTS node
```

## Emergency Commands

```
"Explain what this code does: [paste code]"
"Find all uses of [function/class] in the codebase."
"What's causing this error: [paste error]"
"Rollback the last change to [file]."
```

---

Remember: The AI is your pair programmer. Communicate clearly, ask questions, and iterate together!
