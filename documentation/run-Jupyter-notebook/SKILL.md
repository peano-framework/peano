---
name: peano:run-Jupyter-notebook
description: Execute a Jupyter notebook (e.g. a tutorial under tutorials/) with nbconvert and afterwards rebuild the Doxygen documentation. Runs automatically as a PostToolUse hook after every modification of a .ipynb file. Use explicitly when the user asks to run/execute/test a notebook, or after notebook changes the hook does not see.
allowed-tools: [Bash, Read]
---

# Execute a notebook, then rebuild the docs

After every modification of a notebook, we execute it end to end with
`jupyter nbconvert --execute` (this generates, compiles and runs the tutorial's
code) and afterwards run Doxygen. Both steps live in
`.claude/skills/run-notebook/run-notebook.sh`, which this skill and the hooks share.

## Steps

1. Run the script for the notebook:
   ```sh
   .claude/skills/run-notebook/run-notebook.sh tutorials/exahype2/13_euler-with-amr.ipynb
   ```
   Run it in the background (Bash `run_in_background`) and wait for its completion
   notification: the execution takes minutes, the Doxygen rebuild about 9 more.
2. The notebook is executed in its own directory. The source notebook is not
   modified. The executed copy and the log go to
   `<root>/.claude/notebook-runs/<name>.ipynb` and `<name>.log`. All cells are run,
   even if some fail (`--allow-errors`). The end of the log lists the failing
   cells: cells raising a Python error, and shell cells (`!make`, ...) whose output
   contains compiler or make errors. The last line reports nbconvert's exit code.
3. Afterwards the script calls `.claude/skills/regen-docs/regen-docs.sh`, which
   converts the notebooks into .dox files and runs Doxygen. Its output goes to
   `<root>/.doxygen-rebuild.log`. Report warnings about the notebook's page.
4. Tell the user whether the notebook ran through and whether its page has warnings.

## Automatic runs

`.claude/settings.json` calls `run-notebook.sh --hook` after every
`Write|Edit|MultiEdit|NotebookEdit`. If the edited file is a `.ipynb`, the script
starts the run detached and returns at once. After every `Bash` call,
`run-notebook.sh --hook-bash` reruns all tracked notebooks of the command's
repository or worktree that are newer than their last run, e.g. after a merge or
an edit via sed or Python. Only notebooks that have been run before are picked up
this way, so a notebook created or first changed via Bash needs one explicit run.

Runs for one root are serialised, and a notebook whose current version has
already been run is skipped. Hence, a series of quick edits results in at most
two runs. The Doxygen rebuild waits for a rebuild that is already running.

Limitations:
- The generated code has to be built against a configured Peano. In a tree
  without `config.status` or `build/` (e.g. a fresh git worktree), the script skips
  the execution and says so in the log. Run the notebook in the main checkout then.
- The kernel is whatever `jupyter` finds. Cells which need packages that are
  missing there (e.g. `ipywidgets` for the interactive viewer) fail in the log.
- Executing a notebook writes its output directory (e.g.
  `tutorials/exahype2/13_euler-with-amr/`) next to it, exactly as a manual run.
