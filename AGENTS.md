# AGENTS.md: standing rules (read first, always apply)

You are helping revise the experiments of a GPR buried-object detection research paper.
Read `PAPER_CONTEXT.md` (facts), then `TASK.md` (what to do), and tick items in `CHECKLIST.md` as you finish them.

## Ask before acting (mandatory)
Follow the plan in `TASK.md` exactly. Do nothing outside it without my approval.

Stop and ask me, and wait for my answer, whenever:
1. You have any doubt, or something is unclear, missing, or contradicts what you find in the repo (a file, label format, class name, path, or metadata field that does not match the docs).
2. You want to do something that is not in the plan: add or remove a model, change a hyperparameter, change the split ratio, change augmentation logic, change library versions, delete or rename files, or skip a step.
3. Something fails or cannot be done as written (a model will not train, a dataset has no labels, a package conflicts). Do not silently work around it or substitute something else.
4. You are about to make an assumption that affects a result or a number in the paper.
5. An action is hard to undo (deleting data, overwriting results, force-push, rewriting git history).

How to ask:
- Stop that step and write a message starting with `QUESTION:`.
- Give the problem in 1-2 plain sentences, list options (A / B / C), say which you recommend and why, and say what you will do for each answer.
- One clear question at a time, in simple language.
- Do not continue that step until I reply. You may work on other steps that do not depend on the answer, if you tell me you are doing so.

Do not ask about trivial things (file names inside the planned folders, plot colors).

## Golden rules
1. Never fabricate numbers, references, or results. Every number in a table must come from a saved file in `results/`. If something cannot be run, say so and say why.
2. Split by SOURCE image first; augment only the training part. Validation and test sets are never augmented.
3. Same split, same image size (224), same augmented training set, same epochs and seeds for every model compared.
4. Use seeds 0, 1, 2 for every main result; report mean ± std.
5. Never delete or overwrite existing code or results. Work on branch `paper-revision`.
6. After each phase, stop and give a short summary: what was done, files created, key numbers. Ask before starting the next phase.
7. Never edit the paper's text directly. Only write suggested changes to `results/PAPER_UPDATES.md`.
8. Never drop a model silently, never reuse the old inflated results as new ones, never invent or estimate a number.
