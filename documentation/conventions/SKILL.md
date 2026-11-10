---
name: peano:format-conventions
description: Ensure that all files follow similar coding conventions where appropriate, and that certain best practices are followed.
allowed-tools: [Bash, Read]
---

# Ever only alter files that have seen significant changes

Only apply this skill to files which have seen significant modififiations.
Significant means actual text has changed. 
Fixing single typos or references does not mean significant.
If a current merge request would already involve more than 10 files, then only focus on the 10 files that have already most of the changes.
Ignore the others.
Ensure that the formatter never changes more than 10 percent of a file by means of lines of code.



## Formatting steps

Read the file conventions.dox and implement the rules from there.


## Code quality

### Search for flaws

Search for some common flaws in lots of projects:

1. Inheritance hierarchy of more than two. Analyse if you can propose a design which avoids deep inheritance.
2. Peano has a strict layering: application code uses things like ExaGRyPE. They use ExaHyPE or MGHyPE or ExaParty. They use Peano. Peano uses the tarch.
   Ensure that there are no calls the other way around in generated code, and flag if a piece of code "jumps" over one layer.
3. As much code as possible should end up in C++ files rather than be hidden in Python strings. Try to push code parts into C++ therefore.
   Where a solver of an extension defines PDE terms (flux, source, eigenvalue, initial condition, ...)
   in C++ snippets, the bodies belong into a `Kernels.h`/`Kernels.cpp` pair (or `<Solver>Kernels.h/.cpp`)
   in a subdirectory of the extension named after the solver, e.g. `src/exagrype/ccz4/Kernels.h`
   or `src/exagrype/euler/EulerKernels.h`. The pair is part of the extension's library and sits in
   the namespace `<extension>::<solver>`. The Python class only emits one-line calls into these kernels
   and adds the include. The docu of the kernels sits in the header.
4. Python strings that represent C++ code snippets should, where possible, not rely on naming conventions, but instead directly get the Python object that represents what a C++ code should couple to and ask this Python object what C++ names it will generate. The logic which names to pick sits within the emitting Python code, and all other places read from there rather than assuming certain names.
5. All C++ docu is in the header files next to the method declaration.
6. Minimise in-code documentation, both in Python and C++. Try to move descriptions into the method documentations.

If we find such flaws, report them to the user.


### Quality control over documentation

Most documentation is very much focused on what a code does. Point out if a
documentation lacks descriptions why things are done the way they are done.
Also recommend to embed AI work plans and failed attempts into the docu.

If documentation explains workflows or dependencies, use dot to create 
cartoons or illustrations of these facts rather than relying solely on text.



