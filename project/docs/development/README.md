# Development Documentation

This folder contains all development-related documentation for the IFT6759 Splendor RL project.

## Structure

### `/sprints/`
Sprint planning, backlog, and retrospectives organized by sprint number.
- `sprint_N/` - Each sprint folder contains:
  - `planning.md` - Sprint goals, user stories, and task breakdown
  - `daily_logs/` - Brief daily progress updates
  - `retrospective.md` - What went well, what didn't, action items

### `/dev_logs/`
Chronological development logs for each work session.
- Format: `YYYY-MM-DD_session_description.md`
- Documents what was done, decisions made, and next steps

### `/decisions/`
Architecture Decision Records (ADRs) for important technical choices.
- Format: `ADR-NNN-short-title.md`
- Captures context, decision, alternatives, and consequences

### `/specs/`
Detailed technical specifications and design documents.
- Agent architectures
- Neural network designs
- MCTS implementations
- Reward function specifications

### `/meeting_notes/`
Notes from team meetings, advisor discussions, and planning sessions.

## Workflow with AI Assistant

### Starting a Session
1. Review current sprint backlog: `sprints/sprint_N/planning.md`
2. Check latest dev log to understand context
3. Tell AI: "I want to work on [task from backlog]"

### During Development
- AI will help implement features, debug, and explain code
- Ask AI to document decisions in ADRs when making architectural choices
- Request AI to create dev logs summarizing what was accomplished

### Ending a Session
1. Ask AI to create/update dev log for the session
2. Update sprint planning doc with completed tasks
3. Commit changes with meaningful messages

### Best Practices
- **Be Specific**: Tell AI exactly which task you're tackling
- **Request Documentation**: Ask AI to "document this in a dev log" or "create an ADR for this decision"
- **Iterative Refinement**: Review AI-generated docs and request edits
- **Link Context**: Reference previous logs/ADRs when continuing work

## Example Commands

```
"Let's start Sprint 2. Create the planning doc and help me break down the tasks."

"I just implemented the Event-Based evaluator. Create a dev log documenting what we did."

"We need to decide between DQN and PPO for Phase 1. Help me create an ADR."

"Update today's daily log with progress on the MCTS implementation."

"Generate a sprint 1 retrospective based on our dev logs."
```

## Current Sprint Status

**Active Sprint**: Sprint 1 (Phase 1 - Reward Shaping Baseline)
**Sprint Goal**: Implement Score-based and Event-based agents, run comparison experiments
**Start Date**: 2026-02-24
**End Date**: 2026-03-10 (2 weeks)

See `sprints/sprint_01/planning.md` for details.
